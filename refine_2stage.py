import numpy as np
from scipy.sparse import coo_matrix
import pyamg
from pyamg.krylov import cg as pyamg_cg
import hashlib
from collections import OrderedDict

# === Global constants and cache ===
F32 = np.float32
_MAX_AMG_CACHE = 8
_PYAMG_CACHE = OrderedDict()   # sig -> ml (AMG hierarchy)
_PREV_SOL     = {}             # sig -> last solution (float32, free-space)

_PREV_SOL_2STAGE = {}             # sig -> last solution (float32, free-space)

# ======================= Utilities =======================


def _make_rays(K: np.ndarray, H: int, W: int) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(H, dtype=F32),
                         np.arange(W, dtype=F32), indexing="ij")
    pix = np.stack([xx, yy, np.ones_like(xx, dtype=F32)], axis=-1).astype(F32)
    K = K.astype(F32, copy=False)
    Kinv = np.linalg.inv(K).astype(F32)
    rays = pix @ Kinv.T
    nrm = np.linalg.norm(rays, axis=2, keepdims=True).astype(F32)
    rays = rays / np.clip(nrm, F32(1e-6), None)
    return rays.astype(F32)


def _binary_dilate(mask: np.ndarray, radius: int = 1, include_diag: bool = True) -> np.ndarray:
    """Simple dilation: 3x3 neighbors repeated radius times."""
    if radius is None or radius <= 0:
        return mask
    out = mask.copy()
    for _ in range(int(radius)):
        pad = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        nb = [
            pad[1:-1, 2:],   # E
            pad[1:-1, :-2],  # W
            pad[2:, 1:-1],   # S
            pad[:-2, 1:-1],  # N
        ]
        if include_diag:
            nb += [pad[2:, 2:], pad[2:, :-2], pad[:-2, 2:], pad[:-2, :-2]]
        out = np.logical_or(out, np.logical_or.reduce(nb))
    return out


def _conv2d_same(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """3x3 kernel SAME convolution (padding=edge)."""
    H, W = img.shape
    pad = np.pad(img, ((1, 1), (1, 1)), mode="edge")
    out = np.empty_like(img, dtype=img.dtype)
    # 3x3 expansion
    out[:, :] = (
        pad[0:H,   0:W]   * k[0,0] + pad[0:H,   1:W+1] * k[0,1] + pad[0:H,   2:W+2] * k[0,2] +
        pad[1:H+1, 0:W]   * k[1,0] + pad[1:H+1, 1:W+1] * k[1,1] + pad[1:H+1, 2:W+2] * k[1,2] +
        pad[2:H+2, 0:W]   * k[2,0] + pad[2:H+2, 1:W+1] * k[2,1] + pad[2:H+2, 2:W+2] * k[2,2]
    )
    return out


def detect_discontinuities(
    ir_in: np.ndarray,                 # 0~255 IR/grayscale
    *,
    # Threshold settings: tau_abs_norm for absolute threshold (after normalization), otherwise percentile
    tau_abs_norm: float | None = None, # [0,1] scale absolute threshold (e.g. 0.25). None uses percentile
    tau_percentile: float = 85.0,      # gradient percentile threshold (default 85%)
    band_radius: int | None = None,    # if provided, apply dilation and return result
    include_diag: bool = True          # include diagonal in dilation
) -> np.ndarray:
    """
    IR(0~255) → Sobel edge → discontinuity map generation.
    - Default: threshold by 'tau_percentile' percentile of gradient magnitude.
    - If tau_abs_norm is given (0~1), normalize grad to 0~1 then use absolute threshold.
    - If band_radius is given, apply dilation and return result.
    Returns: disc (H,W) bool
    """
    g = ir_in.astype(F32, copy=False) / F32(255.0)  # 0~1 normalization

    # Sobel kernels
    sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=F32)
    sy = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=F32)

    gx = _conv2d_same(g, sx)
    gy = _conv2d_same(g, sy)
    grad = np.sqrt(gx*gx + gy*gy).astype(F32)  # gradient magnitude (0~roughly numerical)

    # Threshold calculation
    eps = F32(1e-6)
    if tau_abs_norm is not None:
        # Stable normalization of grad to 0~1 (scaled by 99th percentile)
        scale = F32(np.percentile(grad, 99.0))
        if not np.isfinite(scale) or scale < eps:
            scale = grad.max().astype(F32)
        if scale < eps:
            return np.zeros_like(g, dtype=bool)
        grad_n = np.clip(grad / scale, 0.0, 1.0)
        thr = F32(tau_abs_norm)
        disc = grad_n >= thr
    else:
        # Percentile-based threshold
        thr = F32(np.percentile(grad, float(tau_percentile)))
        disc = grad >= thr

    # Band option: if provided, apply dilation and return
    if band_radius is not None and band_radius > 0:
        disc = _binary_dilate(disc, radius=band_radius, include_diag=include_diag)

    return disc


def gates_from_disc_1d(disc: np.ndarray) -> dict:
    H, W = disc.shape
    pair_disc_h = disc[:, :-1] | disc[:, 1:]
    pair_disc_v = disc[:-1, :] | disc[1:, :]

    mask_h = ~pair_disc_h
    mask_v = ~pair_disc_v
    pix_disc = disc

    candL_ok = np.zeros((H, W), dtype=bool); candL_ok[:, 1:]  = ~pair_disc_h
    candR_ok = np.zeros((H, W), dtype=bool); candR_ok[:, :-1] = ~pair_disc_h
    candU_ok = np.zeros((H, W), dtype=bool); candU_ok[1:, :]  = ~pair_disc_v
    candD_ok = np.zeros((H, W), dtype=bool); candD_ok[:-1, :] = ~pair_disc_v

    return dict(mask_h=mask_h, mask_v=mask_v,
                candL_ok=candL_ok, candR_ok=candR_ok, candU_ok=candU_ok, candD_ok=candD_ok,
                pix_disc=pix_disc)

# =================== Pattern hash based AMG cache ===================


def _pattern_sig(rows_i32: np.ndarray, cols_i32: np.ndarray, shape_tuple: tuple[int, int]) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(rows_i32.tobytes())
    h.update(cols_i32.tobytes())
    h.update(np.asarray(shape_tuple, dtype=np.int64).tobytes())
    return h.hexdigest()


def _get_pyamg_solver_by_sig_float32(AtA_csr32, sig: str):
    if sig in _PYAMG_CACHE:
        ml = _PYAMG_CACHE.pop(sig)
        _PYAMG_CACHE[sig] = ml
        return ml
    ml = pyamg.smoothed_aggregation_solver(
        AtA_csr32,
        symmetry='hermitian',
        strength='symmetric',
        aggregate='standard',
        smooth='energy',
        presmoother=('chebyshev', {'degree': 2}),
        postsmoother=('chebyshev', {'degree': 2}),
        max_levels=10,
        max_coarse=500,
        B=np.ones((AtA_csr32.shape[0], 1), dtype=np.float32),
    )
    _PYAMG_CACHE[sig] = ml
    while len(_PYAMG_CACHE) > _MAX_AMG_CACHE:
        _PYAMG_CACHE.popitem(last=False)
    return ml


def solve_with_pyamg32(A_csr32, b32, *, maxiter=2, x0=None, cycle="V", sig: str):
    A = A_csr32.tocoo(); A.sum_duplicates(); A = A.tocsr()
    AtA = (A.T @ A).tocsr().astype(np.float32, copy=False)
    Atb = (A.T @ b32).astype(np.float32, copy=False)
    ml = _get_pyamg_solver_by_sig_float32(AtA, sig)
    x0_ = None if x0 is None else x0.astype(np.float32, copy=False)
    z32 = ml.solve(Atb, x0=x0_, tol=1e-4, maxiter=int(maxiter), cycle=cycle)
    return z32


def solve_with_pcg_pyamg32(A_csr32, b32, *, cg_maxiter=10, x0=None, sig: str):
    A = A_csr32.tocoo(); A.sum_duplicates(); A = A.tocsr()
    AtA = (A.T @ A).tocsr().astype(np.float32, copy=False)
    Atb = (A.T @ b32).astype(np.float32, copy=False)
    ml = _get_pyamg_solver_by_sig_float32(AtA, sig)
    M = ml.aspreconditioner()
    x0_ = None if x0 is None else x0.astype(np.float32, copy=False)
    z32, info = pyamg_cg(AtA, Atb, tol=1e-4, maxiter=int(cg_maxiter), M=M, x0=x0_, callback=None, residuals=None)
    return z32

# =================== Soft-Top (Normal similarity) utils ===================


def _group_softmax_by_p(p_all: np.ndarray, score_all: np.ndarray,
                        *, tau: float, topk: int | None = None) -> np.ndarray:
    """Softmax by p for score. If topk specified, only top K per p get softmax, rest 0."""
    if score_all.size == 0:
        return np.zeros_like(score_all, dtype=F32)
    order = np.argsort(p_all)
    p_sorted = p_all[order]
    s_sorted = score_all[order]
    cuts = np.flatnonzero(np.r_[True, p_sorted[1:] != p_sorted[:-1], True])
    w_sorted = np.zeros_like(s_sorted, dtype=F32)
    for s, e in zip(cuts[:-1], cuts[1:]):
        seg = s_sorted[s:e]
        if seg.size == 0:
            continue
        if topk is not None and seg.size > topk:
            idx_top = np.argpartition(-seg, topk-1)[:topk]
            mask = np.zeros(seg.size, dtype=bool); mask[idx_top] = True
            seg_use = seg[mask]
            m = np.max(seg_use)
            w = np.exp((seg_use - m) / F32(tau)).astype(F32)
            w /= np.sum(w) + F32(1e-12)
            w_full = np.zeros_like(seg, dtype=F32); w_full[mask] = w
            w_sorted[s:e] = w_full
        else:
            m = np.max(seg)
            w = np.exp((seg - m) / F32(tau)).astype(F32)
            w /= np.sum(w) + F32(1e-12)
            w_sorted[s:e] = w
    w_all = np.zeros_like(w_sorted, dtype=F32)
    w_all[order] = w_sorted
    return w_all


def _normal_similarity_score(n_p_blk: np.ndarray, n_q_blk: np.ndarray, *, beta: float) -> np.ndarray:
    """|dot(n_p, n_q)|^beta (assuming normalized normals)"""
    cos = np.abs(np.sum(n_p_blk * n_q_blk, axis=-1)).astype(F32)
    return np.power(np.clip(cos, F32(0.0), F32(1.0)), F32(beta)).astype(F32)

# ================= Main (float32 / Soft-Top normal / AMG cache) =================

def refine_depth_normal_alignment(
    depth_in: np.ndarray,
    known_mask: np.ndarray,
    hole_mask: np.ndarray,
    n_guide: np.ndarray | None,
    K: np.ndarray | None,
    discontinuity_maps: np.ndarray | None = None,  # (H,W) bool
    # Lambda parameters (E/P 제거)
    lambda_normal: float = 3.0,     # (N) normal alignment outside boundaries
    lambda_screen: float = 1e-3,    # (R) sparse anchor (drift prevention)
    lambda_keep: float = 30.0,      # (K) keep known (경계 제외)
    # AMG/PCG
    use_pcg: bool = False,          # True for PCG+AMG, False for AMG.solve
    pyamg_cycle: str = "V",         # "V"|"W"
    maxiter: int = 2,               # AMG V-cycle count
    cg_maxiter: int = 8,            # PCG iteration count
) -> np.ndarray:
    """
    Equal/Plane 항 제거 버전.
    구성 항: (N) 노멀 정합, (R) 스크린 앵커, (K) 킵-노운
    자유변수: 경계 픽셀 ∪ 홀 픽셀
    """
    # ==== Input preprocessing ====
    depth_in = depth_in.astype(F32, copy=False)
    known_mask = known_mask.astype(bool, copy=False)
    hole_mask  = hole_mask.astype(bool,  copy=False)
    H, W = depth_in.shape
    Npix = H * W

    if K is None:
        K = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=F32)
    else:
        K = K.astype(F32, copy=False)
    if n_guide is None:
        n = np.zeros((H, W, 3), dtype=F32); n[..., 2] = F32(1.0)
    else:
        n = n_guide.astype(F32, copy=False)
    rays = _make_rays(K, H, W)

    # Normal normalization & flip toward camera
    n_norm = np.linalg.norm(n, axis=2, keepdims=True).astype(F32)
    bad = (n_norm < F32(1e-6))
    if bad.any():
        n[bad[..., 0]] = np.array([0, 0, 1], dtype=F32)
        n_norm = np.linalg.norm(n, axis=2, keepdims=True).astype(F32)
    n = n / np.clip(n_norm, F32(1e-6), None)
    dot = np.sum(n * rays, axis=2).astype(F32)
    n[dot > F32(0.0)] *= F32(-1.0)
    n = np.nan_to_num(n, nan=0.0, posinf=1.0, neginf=-1.0)
    n = n / np.clip(np.linalg.norm(n, axis=2, keepdims=True).astype(F32), F32(1e-6), None)

    # Boundary & gates
    disc = detect_discontinuities(depth_in) if (discontinuity_maps is None) else discontinuity_maps
    fused = gates_from_disc_1d(disc)
    mask_h, mask_v = fused["mask_h"], fused["mask_v"]
    pix_disc = fused["pix_disc"]

    # Free variables: boundary ∪ hole
    free = (pix_disc | hole_mask)
    free_vec = free.reshape(-1)
    map_full2free = -np.ones(Npix, dtype=np.int32)
    free_ids = np.where(free_vec)[0].astype(np.int32)
    map_full2free[free_ids] = np.arange(free_ids.size, dtype=np.int32)

    data_buf = []; row_buf = []; col_buf = []; b_buf = []
    row_ofs = 0

    def _append_rows_2var(c_p: np.ndarray, c_q: np.ndarray,
                          p_full: np.ndarray, q_full: np.ndarray,
                          rhs: np.ndarray, z_p_snap: np.ndarray, z_q_snap: np.ndarray, w: np.ndarray):
        nonlocal row_ofs
        if c_p.size == 0:
            return
        p_free = map_full2free[p_full]
        q_free = map_full2free[q_full]
        both = (p_free >= 0) & (q_free >= 0)
        only_p = (p_free >= 0) & (q_free < 0)
        only_q = (p_free < 0) & (q_free >= 0)

        if np.any(both):
            Kk = int(np.count_nonzero(both))
            rr = np.arange(Kk, dtype=np.int32) + row_ofs
            data_buf.append((w[both] * c_p[both]).astype(F32)); row_buf.append(rr); col_buf.append(p_free[both])
            data_buf.append((w[both] * c_q[both]).astype(F32)); row_buf.append(rr); col_buf.append(q_free[both])
            b_buf.append((w[both] * rhs[both]).astype(F32))
            row_ofs += Kk

        if np.any(only_p):
            Kk = int(np.count_nonzero(only_p))
            rr = np.arange(Kk, dtype=np.int32) + row_ofs
            data_buf.append((w[only_p] * c_p[only_p]).astype(F32)); row_buf.append(rr); col_buf.append(p_free[only_p])
            rhs_corr = (rhs[only_p] - c_q[only_p] * z_q_snap[only_p]).astype(F32)
            b_buf.append((w[only_p] * rhs_corr).astype(F32))
            row_ofs += Kk

        if np.any(only_q):
            Kk = int(np.count_nonzero(only_q))
            rr = np.arange(Kk, dtype=np.int32) + row_ofs
            data_buf.append((w[only_q] * c_q[only_q]).astype(F32)); row_buf.append(rr); col_buf.append(q_free[only_q])
            rhs_corr = (rhs[only_q] - c_p[only_q] * z_p_snap[only_q]).astype(F32)
            b_buf.append((w[only_q] * rhs_corr).astype(F32))
            row_ofs += Kk

    # Geometric diffs
    rx = (rays[:, 1:, :] - rays[:, :-1, :]).astype(F32)
    ry = (rays[1:, :, :] - rays[:-1, :, :]).astype(F32)

    # (N) Normal alignment: exclude boundary pairs
    if lambda_normal > 0:
        lamN = F32(np.sqrt(F32(lambda_normal)))

        # Horizontal pairs
        ok_h = mask_h.reshape(-1)
        if np.any(ok_h):
            n_p = n[:, :-1, :].reshape(-1, 3)
            r_p = rays[:, :-1, :].reshape(-1, 3)
            rdx = rx.reshape(-1, 3)
            p_idx = (np.arange(H*W, dtype=np.int32).reshape(H, W)[:, :-1]).reshape(-1)
            q_idx = (np.arange(H*W, dtype=np.int32).reshape(H, W)[:,  1:]).reshape(-1)

            a = np.sum(n_p * r_p, axis=1).astype(F32)
            b = np.sum(n_p * rdx, axis=1).astype(F32)
            a = np.nan_to_num(a).astype(F32); b = np.nan_to_num(b).astype(F32)
            c_p = (b - a).astype(F32); c_q = a
            den = np.sqrt(c_p * c_p + c_q * c_q).astype(F32) + F32(1e-6)
            c_p = (c_p / den).astype(F32); c_q = (c_q / den).astype(F32)

            if np.any(ok_h):
                z_p = depth_in.reshape(-1)[p_idx]
                z_q = depth_in.reshape(-1)[q_idx]
                w = lamN * np.ones(np.count_nonzero(ok_h), dtype=F32)
                _append_rows_2var(c_p[ok_h], c_q[ok_h],
                                  p_idx[ok_h], q_idx[ok_h],
                                  np.zeros(np.count_nonzero(ok_h), dtype=F32),
                                  z_p[ok_h], z_q[ok_h], w)

        # Vertical pairs
        ok_v = mask_v.reshape(-1)
        if np.any(ok_v):
            n_p = n[:-1, :, :].reshape(-1, 3)
            r_p = rays[:-1, :, :].reshape(-1, 3)
            rdy = ry.reshape(-1, 3)
            p_idx = (np.arange(H*W, dtype=np.int32).reshape(H, W)[:-1, :]).reshape(-1)
            q_idx = (np.arange(H*W, dtype=np.int32).reshape(H, W)[ 1:, :]).reshape(-1)

            a = np.sum(n_p * r_p, axis=1).astype(F32)
            b = np.sum(n_p * rdy, axis=1).astype(F32)
            a = np.nan_to_num(a).astype(F32); b = np.nan_to_num(b).astype(F32)
            c_p = (b - a).astype(F32); c_q = a
            den = np.sqrt(c_p * c_p + c_q * c_q).astype(F32) + F32(1e-6)
            c_p = (c_p / den).astype(F32); c_q = (c_q / den).astype(F32)

            if np.any(ok_v):
                z_p = depth_in.reshape(-1)[p_idx]
                z_q = depth_in.reshape(-1)[q_idx]
                w = lamN * np.ones(np.count_nonzero(ok_v), dtype=F32)
                _append_rows_2var(c_p[ok_v], c_q[ok_v],
                                  p_idx[ok_v], q_idx[ok_v],
                                  np.zeros(np.count_nonzero(ok_v), dtype=F32),
                                  z_p[ok_v], z_q[ok_v], w)

    # (R) Screen: non-boundary holes only + stride(2,2) anchor (weak)
    if lambda_screen > 0:
        lamR = F32(np.sqrt(F32(lambda_screen)))
        base = hole_mask & (~pix_disc)
        sy, sx = 2, 2
        oy, ox = 0, 0
        mask_sub = np.zeros_like(base, dtype=bool)
        mask_sub[oy::sy, ox::sx] = True
        use = base & mask_sub
        if np.any(use):
            ids_full = np.arange(Npix, dtype=np.int32).reshape(H, W)[use].reshape(-1)
            cols_free = map_full2free[ids_full]
            ok = cols_free >= 0
            if np.any(ok):
                Kk = int(np.count_nonzero(ok))
                rr = np.arange(Kk, dtype=np.int32)
                ww = lamR * np.ones(Kk, dtype=F32)
                data_buf.append(ww.astype(F32))
                row_buf.append(rr + row_ofs)
                col_buf.append(cols_free[ok])
                b_buf.append((ww * depth_in.reshape(-1)[ids_full[ok]].astype(F32)))
                row_ofs += Kk

    # (K) Keep-known: strongly maintain known, BUT EXCLUDE discontinuities
    if lambda_keep > 0:
        lamK = F32(np.sqrt(F32(lambda_keep)))
        keep_candidates = known_mask & (~pix_disc)
        if np.any(keep_candidates):
            ids_full = np.arange(Npix, dtype=np.int32).reshape(H, W)[keep_candidates].reshape(-1)
            cols_free = map_full2free[ids_full]
            ok = cols_free >= 0
            if np.any(ok):
                Kk = int(np.count_nonzero(ok))
                rr = np.arange(Kk, dtype=np.int32)
                ww = lamK * np.ones(Kk, dtype=F32)
                data_buf.append(ww.astype(F32))
                row_buf.append(rr + row_ofs)
                col_buf.append(cols_free[ok])
                b_buf.append((ww * depth_in.reshape(-1)[ids_full[ok]].astype(F32)))
                row_ofs += Kk

    # ===== Assembly & Solve =====
    if len(data_buf) == 0:
        return depth_in.copy().astype(F32)

    data = np.concatenate(data_buf).astype(np.float32, copy=False)
    rows = np.concatenate(row_buf).astype(np.int32,   copy=False)
    cols = np.concatenate(col_buf).astype(np.int32,   copy=False)
    b    = np.concatenate(b_buf).astype(np.float32,   copy=False)

    R = int(rows.max()) + 1
    Ncols = int(free_ids.size)
    sig = _pattern_sig(rows, cols, (R, Ncols))
    A = coo_matrix((data, (rows, cols)), shape=(R, Ncols), dtype=np.float32).tocsr()

    x0 = _PREV_SOL.get(sig, depth_in.reshape(-1)[free_ids].astype(np.float32, copy=False))
    if use_pcg:
        z_free32 = solve_with_pcg_pyamg32(A, b, cg_maxiter=cg_maxiter, x0=x0, sig=sig)
    else:
        z_free32 = solve_with_pyamg32(A, b, maxiter=maxiter, x0=x0, cycle=pyamg_cycle, sig=sig)

    out = depth_in.copy().astype(F32, copy=False)
    out_flat = out.reshape(-1)
    out_flat[free_ids] = z_free32.astype(F32, copy=False)
    _PREV_SOL[sig] = out_flat[free_ids].copy()

    mask_h_full = np.zeros((H, W), dtype=bool)
    mask_v_full = np.zeros((H, W), dtype=bool)
    mask_h_full[:, 1:] = ~mask_h
    mask_v_full[1:, :] = ~mask_v

    refine_next = (mask_v_full & mask_h_full).astype(np.bool) | disc
    return out, refine_next

####

def _normals_from_depth(depth_in: np.ndarray,
                        rays: np.ndarray,
                        valid: np.ndarray | None = None,
                        *,
                        eps: float = 1e-6) -> np.ndarray:
    """
    입력 depth와 rays로부터 카메라좌표계 법선맵 계산.
    - X = z*r 로 3D 복원 후, dX/dx × dX/dy
    - 경계/결측 안전 처리, 방향은 카메라를 향하게(n·r <= 0)
    """
    z = depth_in.astype(F32, copy=False)
    H, W = z.shape
    if valid is None:
        valid = np.isfinite(z)
    valid = valid & (z > 0)

    # 3D 포인트
    X = (rays * z[..., None]).astype(F32)  # (H,W,3)

    # 이산 차분(전방/후방 혼합: Sobel보다 간단하고 빠름)
    # 중앙차분 유사: 좌우/상하가 유효하면 평균, 아니면 가능한 한쪽 사용
    def _diff_h(A):
        # (H,W,3)
        left  = A[:, :-1, :]
        right = A[:, 1:,  :]
        out = np.zeros_like(A, dtype=F32)
        out[:, 1:-1, :] = (A[:, 2:, :] - A[:, :-2, :]) * F32(0.5)
        out[:, 0,    :] = (right[:, 0, :] - A[:, 0, :])
        out[:, -1,   :] = (A[:, -1, :] - left[:, -1, :])
        return out

    def _diff_v(A):
        up    = A[:-1, :, :]
        down  = A[1:,  :, :]
        out = np.zeros_like(A, dtype=F32)
        out[1:-1, :, :] = (A[2:, :, :] - A[:-2, :, :]) * F32(0.5)
        out[0,    :, :] = (down[0, :, :] - A[0, :, :])
        out[-1,   :, :] = (A[-1, :, :] - up[-1, :, :])
        return out

    dXdx = _diff_h(X)
    dXdy = _diff_v(X)

    # 법선 = dX/dx × dX/dy
    n = np.cross(dXdx, dXdy, axis=-1).astype(F32)
    # 안정화: 크기 0 대비
    nrm = np.linalg.norm(n, axis=-1, keepdims=True).astype(F32)
    bad = (nrm < F32(eps)) | (~valid[..., None])
    # 최소 대체: z>0인 곳은 카메라 향 기본값(0,0,1) 적용
    n[bad[..., 0]] = np.array([0, 0, 1], dtype=F32)
    nrm = np.linalg.norm(n, axis=-1, keepdims=True).astype(F32)
    n = n / np.clip(nrm, F32(eps), None)

    # 방향 통일: 카메라를 향하게 (n·r <= 0)
    dot = np.sum(n * rays, axis=-1).astype(F32)
    flip = (dot > F32(0.0))[..., None]
    n = np.where(flip, -n, n)

    return n.astype(F32)


def _softmax_per_p(p_all: np.ndarray, scores: np.ndarray, *, tau: float, topk: int | None) -> np.ndarray:
    if scores.size == 0:
        return np.zeros_like(scores, dtype=F32)
    order = np.argsort(p_all)
    p_sorted = p_all[order]; s_sorted = scores[order]
    cuts = np.flatnonzero(np.r_[True, p_sorted[1:] != p_sorted[:-1], True])
    w_sorted = np.zeros_like(s_sorted, dtype=F32)
    for s, e in zip(cuts[:-1], cuts[1:]):
        seg = s_sorted[s:e]
        if seg.size == 0:
            continue
        if topk is not None and seg.size > topk:
            idx_top = np.argpartition(-seg, topk-1)[:topk]
            mask = np.zeros(seg.size, dtype=bool); mask[idx_top] = True
            seg_use = seg[mask]
            m = np.max(seg_use)
            w = np.exp((seg_use - m) / F32(tau)).astype(F32)
            w /= np.sum(w) + F32(1e-12)
            w_full = np.zeros_like(seg, dtype=F32); w_full[mask] = w
            w_sorted[s:e] = w_full
        else:
            m = np.max(seg)
            w = np.exp((seg - m) / F32(tau)).astype(F32)
            w /= np.sum(w) + F32(1e-12)
            w_sorted[s:e] = w
    w_all = np.zeros_like(w_sorted, dtype=F32)
    w_all[order] = w_sorted
    return w_all


def _cos_score(n_p: np.ndarray, n_q: np.ndarray, beta: float) -> np.ndarray:
    cos = np.abs(np.sum(n_p * n_q, axis=-1)).astype(F32)
    return np.power(np.clip(cos, F32(0.0), F32(1.0)), F32(beta)).astype(F32)


def refine_mask_equal_plane_from_depth(
    depth_in: np.ndarray,                # (H,W) 입력 depth (Equal/Plane 모두 이 값 기준)
    target_mask: np.ndarray,             # (H,W) True=refine할 영역(자유변수)
    known_mask: np.ndarray | None = None,# (H,W) True=신뢰가능(없으면 finite(depth)로 대체)
    K: np.ndarray | None = None,
    *,
    # 이웃 설정
    use_8nb: bool = False,               # False=4-방향, True=8-방향
    # 람다
    lambda_equal: float = 1.0,
    lambda_plane: float = 1.0,
    # 게인: 마스크 밖 후보/마스크 안 후보 각각 다르게
    gain_outside: float = 1000.0,        # mask 밖 q 후보에 크게 주면 “밖으로 스냅”이 우선
    gain_inside: float  = 1.0,           # 모두 mask 안이면 이 값으로 soft-top 결합
    # Soft-Top
    soft_tau: float = 0.01,
    soft_beta: float = 2.0,
    soft_topk: int | None = 2,
    # 안정성 게이트 (Plane)
    plane_eps: float = 1e-6,             # |alpha|,|gamma| 게이트
    # AMG/PCG
    use_pcg: bool = False,
    pyamg_cycle: str = "V",
    maxiter: int = 2,
    cg_maxiter: int = 8,
) -> np.ndarray:
    """
    규칙:
      - 자유변수는 target_mask 내부 픽셀 p만.
      - p의 이웃 중 target_mask 밖 q가 있으면 = '법선 유사도 top-1' q만 선택하여 E/P 스냅.
      - 모든 이웃이 target_mask 안이면 = '안 후보들'에 대해 노멀 유사도 soft-top 가중으로 E/P 결합.
      - Equal/Plane의 z_q, 그리고 Plane의 n_q 모두 '입력 depth'에서 계산/유도.
    """
    z = depth_in.astype(F32, copy=False)
    H, W = z.shape
    N = H * W
    tmask = target_mask.astype(bool, copy=False)

    if known_mask is None:
        known_mask = np.isfinite(z)
    else:
        known_mask = known_mask.astype(bool, copy=False) & np.isfinite(z)

    # 레이 & 입력 depth 기반 노멀 계산
    if K is None:
        K = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=F32)
    else:
        K = K.astype(F32, copy=False)
    rays = _make_rays(K, H, W).astype(F32)
    rays = np.nan_to_num(rays, nan=0.0, posinf=0.0, neginf=0.0)

    # 유효 영역을 넓게 잡기 위해 known ∪ target 사용
    n = _normals_from_depth(z, rays, valid=(known_mask | tmask))
    n = np.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)

    # 인덱스/이웃
    idx = np.arange(N, dtype=np.int32).reshape(H, W)
    NB = [(-1,0),(1,0),(0,-1),(0,1)] if not use_8nb else \
         [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]

    # 자유변수는 target_mask 내부만
    free_vec = tmask.reshape(-1)
    free_ids = np.where(free_vec)[0].astype(np.int32)
    map_full2free = -np.ones(N, dtype=np.int32)
    map_full2free[free_ids] = np.arange(free_ids.size, dtype=np.int32)

    # 선형계 버퍼
    data_buf = []; row_buf = []; col_buf = []; b_buf = []
    row_ofs = 0

    def _append_equal_rows(p_cols, zq, w):
        nonlocal row_ofs
        if p_cols.size == 0:
            return
        r = np.arange(p_cols.size, dtype=np.int32) + row_ofs
        data_buf.append(w.astype(F32)); row_buf.append(r); col_buf.append(p_cols)
        b_buf.append((w * zq.astype(F32)).astype(F32))
        row_ofs += p_cols.size

    def _append_plane_rows(p_cols, zq, alpha, gamma, w):
        nonlocal row_ofs
        if p_cols.size == 0:
            return
        r = np.arange(p_cols.size, dtype=np.int32) + row_ofs
        data_buf.append((w * alpha.astype(F32)).astype(F32)); row_buf.append(r); col_buf.append(p_cols)
        b_buf.append((w * gamma.astype(F32) * zq.astype(F32)).astype(F32))
        row_ofs += p_cols.size

    # 타겟 픽셀 루프
    yy, xx = np.where(tmask)
    for cy, cx in zip(yy, xx):
        p_full = idx[cy, cx]
        p_free = map_full2free[p_full]
        if p_free < 0:
            continue

        # 이웃 후보 수집
        out_ids = []
        in_ids  = []
        for dy, dx in NB:
            y = cy + dy; x = cx + dx
            if y < 0 or y >= H or x < 0 or x >= W:
                continue
            q_full = idx[y, x]
            zq_val = z.reshape(-1)[q_full]
            if not np.isfinite(zq_val):
                continue
            if not tmask[y, x]:
                out_ids.append(q_full)
            else:
                in_ids.append(q_full)

        # ===== 선택 규칙 =====
        if len(out_ids) > 0:
            # 밖 이웃 존재 → 법선 유사도 top-1
            gain = F32(gain_outside)
            cands = np.array(out_ids, dtype=np.int32)

            # Equal: top-1 한 개만
            n_p = n[cy, cx][None, :].astype(F32).repeat(cands.size, axis=0)
            n_q = n.reshape(-1, 3)[cands]
            score = _cos_score(n_p, n_q, beta=soft_beta)  # |n_p·n_q|^beta
            best = int(np.argmax(score))
            use_ids_equal = cands[best:best+1]
            wA_equal = np.ones(1, dtype=F32)  # 가중치=1 (top-1)

            # Plane: 안정성(|α|,|γ|) 통과하는 top-1을 순서대로 탐색
            order = np.argsort(-score)  # 고득점 우선
            use_id_plane = None
            for k in order:
                qid = cands[k]
                r_p = rays[cy, cx].astype(F32)
                r_q = rays.reshape(-1,3)[qid]
                n_qk = n.reshape(-1,3)[qid]
                alpha = F32(np.dot(n_qk, r_p))
                gamma = F32(np.dot(n_qk, r_q))
                if (abs(alpha) > F32(plane_eps)) and (abs(gamma) > F32(plane_eps)):
                    use_id_plane = qid
                    break
            if use_id_plane is not None:
                use_ids_plane = np.array([use_id_plane], dtype=np.int32)
                wA_plane = np.ones(1, dtype=F32)
            else:
                use_ids_plane = None
                wA_plane = None

        else:
            # 안쪽만 존재 → Soft-Top
            if len(in_ids) == 0:
                continue
            gain = F32(gain_inside)
            cands = np.array(in_ids, dtype=np.int32)

            n_p = n[cy, cx][None, :].astype(F32).repeat(cands.size, axis=0)
            n_q = n.reshape(-1,3)[cands]
            score = _cos_score(n_p, n_q, beta=soft_beta)
            p_all = np.full(cands.size, p_full, dtype=np.int32)
            wA = _softmax_per_p(p_all, score, tau=F32(soft_tau), topk=soft_topk)
            # 안전장치: 합이 0이면 균등 분배
            s = float(np.sum(wA))
            if not np.isfinite(s) or s <= 1e-12:
                wA = np.ones_like(wA, dtype=F32) / F32(max(1, wA.size))

            use_ids_equal = cands
            use_ids_plane = cands
            wA_equal = wA
            wA_plane = wA

        # ===== Equal 항 =====
        if lambda_equal > 0 and use_ids_equal is not None and use_ids_equal.size > 0:
            lamE = F32(np.sqrt(F32(lambda_equal)))
            zq = z.reshape(-1)[use_ids_equal]
            w  = lamE * gain * wA_equal
            _append_equal_rows(np.full(use_ids_equal.size, p_free, dtype=np.int32), zq, w)

        # ===== Plane 항 =====
        if lambda_plane > 0 and (use_ids_plane is not None) and (use_ids_plane.size > 0):
            lamP = F32(np.sqrt(F32(lambda_plane)))
            r_p  = rays[cy, cx].astype(F32)[None, :].repeat(use_ids_plane.size, axis=0)
            r_q  = rays.reshape(-1,3)[use_ids_plane]
            n_qs = n.reshape(-1,3)[use_ids_plane]

            alpha = np.sum(n_qs * r_p, axis=-1).astype(F32)
            gamma = np.sum(n_qs * r_q, axis=-1).astype(F32)
            alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
            gamma = np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)

            stable = (np.abs(alpha) > F32(plane_eps)) & (np.abs(gamma) > F32(plane_eps))
            if np.any(stable):
                zq_s = z.reshape(-1)[use_ids_plane][stable]
                a_s  = alpha[stable]
                g_s  = gamma[stable]
                if wA_plane is None:
                    wA_s = np.ones(np.count_nonzero(stable), dtype=F32)
                else:
                    wA_s = wA_plane[stable] if wA_plane.size == stable.size else np.ones(np.count_nonzero(stable), dtype=F32)
                w = lamP * gain * wA_s
                _append_plane_rows(np.full(zq_s.size, p_free, dtype=np.int32), zq_s, a_s, g_s, w)

    # ===== 조립 & 풀이 =====
    if len(data_buf) == 0:
        return depth_in.copy().astype(F32)

    data = np.concatenate(data_buf).astype(np.float32, copy=False)
    rows = np.concatenate(row_buf).astype(np.int32,   copy=False)
    cols = np.concatenate(col_buf).astype(np.int32,   copy=False)
    bvec = np.concatenate(b_buf).astype(np.float32,   copy=False)

    R = int(rows.max()) + 1
    A = coo_matrix((data, (rows, cols)), shape=(R, free_ids.size), dtype=np.float32).tocsr()
    sig = _pattern_sig(rows, cols, (R, int(free_ids.size)))

    # 초기해/캐시 (전역 _PREV_SOL 사용)
    x0 = _PREV_SOL_2STAGE.get(sig, depth_in.reshape(-1)[free_ids].astype(np.float32, copy=False))
    if use_pcg:
        z_free32 = solve_with_pcg_pyamg32(A, bvec, cg_maxiter=cg_maxiter, x0=x0, sig=sig)
    else:
        z_free32 = solve_with_pyamg32(A, bvec, maxiter=maxiter, x0=x0, cycle=pyamg_cycle, sig=sig)

    out = depth_in.copy().astype(F32, copy=False)
    out_flat = out.reshape(-1)
    out_flat[free_ids] = z_free32.astype(F32, copy=False)
    _PREV_SOL_2STAGE[sig] = out_flat[free_ids].copy()
    return out
