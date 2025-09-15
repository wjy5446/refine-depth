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
    # Lambda parameters
    lambda_normal: float = 3.0,     # (N) normal alignment outside boundaries
    lambda_smooth: float = 0.0,     # (S) unused (slot maintained)
    lambda_equal: float = 1.0,      # (E) value snap
    lambda_plane: float = 1.0,      # (P) plane projection
    lambda_screen: float = 1e-3,    # (R) sparse anchor (drift prevention)
    lambda_keep: float = 30.0,      # (K) keep known
    # Gains
    gain_known: float = 1000.0,
    gain_hole: float  = 1.0,
    # Soft-Top (normal based)
    use_softA: bool = True,
    softA_tau: float = 0.01,        # softmax temperature
    softA_beta: float = 2.0,        # |cos|^beta
    softA_topk: int = 2,     # top K per p for softmax (None for all)
    # AMG/PCG
    use_pcg: bool = False,          # True for PCG+AMG, False for AMG.solve
    pyamg_cycle: str = "V",         # "V"|"W"
    maxiter: int = 2,               # AMG V-cycle count
    cg_maxiter: int = 8,            # PCG iteration count
) -> np.ndarray:
    """
    Distance(dz) based weighting/selection 'complete removal'.
    Equal/Plane terms weighted only by 'normal similarity Soft-Top'.
    """
    # ==== Input preprocessing ====
    depth_in = depth_in.astype(F32, copy=False)
    known_mask = known_mask.astype(bool, copy=False)
    hole_mask  = hole_mask.astype(bool,  copy=False)
    H, W = depth_in.shape
    Npix = H * W
    eps = F32(1e-6)

    if K is None:
        K = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=F32)
    else:
        K = K.astype(F32, copy=False)
    if n_guide is None:
        n = np.zeros((H, W, 3), dtype=F32); n[..., 2] = F32(1.0)
    else:
        n = n_guide.astype(F32, copy=False)
    rays = _make_rays(K, H, W)

    # Normal normalization and direction unification (toward camera)
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

    # Boundary and gates
    disc = detect_discontinuities(depth_in) if (discontinuity_maps is None) else discontinuity_maps
    fused = gates_from_disc_1d(disc)
    mask_h, mask_v = fused["mask_h"], fused["mask_v"]
    pix_disc = fused["pix_disc"]
    candL_ok = fused["candL_ok"]; candR_ok = fused["candR_ok"]
    candU_ok = fused["candU_ok"]; candD_ok = fused["candD_ok"]

    # Free variables: boundary ∪ hole
    free = (pix_disc | hole_mask)
    free_vec = free.reshape(-1)
    map_full2free = -np.ones(Npix, dtype=np.int32)
    free_ids = np.where(free_vec)[0].astype(np.int32)
    map_full2free[free_ids] = np.arange(free_ids.size, dtype=np.int32)

    data_buf = []; row_buf = []; col_buf = []; b_buf = []
    row_ofs = 0

    def _append_rows_2var(c_p: np.ndarray, c_q: np.ndarray, ridx_local: np.ndarray,
                          p_full: np.ndarray, q_full: np.ndarray, rhs: np.ndarray,
                          z_p_snap: np.ndarray, z_q_snap: np.ndarray, w: np.ndarray):
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

    # Geometric quantities
    rx = (rays[:, 1:, :] - rays[:, :-1, :]).astype(F32)
    ry = (rays[1:, :, :] - rays[:-1, :, :]).astype(F32)

    # (N) Normal alignment: exclude boundary pairs
    if lambda_normal > 0:
        lamN = F32(np.sqrt(F32(lambda_normal)))
        # Horizontal
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
            ok = ok_h
            if np.any(ok):
                z_p = depth_in.reshape(-1)[p_idx]
                z_q = depth_in.reshape(-1)[q_idx]
                w = lamN * np.ones(np.count_nonzero(ok), dtype=F32)
                _append_rows_2var(c_p[ok], c_q[ok],
                                  np.arange(np.count_nonzero(ok), dtype=np.int32),
                                  p_idx[ok], q_idx[ok],
                                  np.zeros(np.count_nonzero(ok), dtype=F32),
                                  z_p[ok], z_q[ok], w)
        # Vertical
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
            ok = ok_v
            if np.any(ok):
                z_p = depth_in.reshape(-1)[p_idx]
                z_q = depth_in.reshape(-1)[q_idx]
                w = lamN * np.ones(np.count_nonzero(ok), dtype=F32)
                _append_rows_2var(c_p[ok], c_q[ok],
                                  np.arange(np.count_nonzero(ok), dtype=np.int32),
                                  p_idx[ok], q_idx[ok],
                                  np.zeros(np.count_nonzero(ok), dtype=F32),
                                  z_p[ok], z_q[ok], w)

    # ===== (E,P) Common: candidate collection (using boundary gates) =====
    idx_map = np.arange(H * W, dtype=np.int32).reshape(H, W)
    dirs = [
        ((slice(None), slice(1,   W)), (slice(None), slice(0,   W-1)), candL_ok),
        ((slice(None), slice(0,   W-1)), (slice(None), slice(1,   W)), candR_ok),
        ((slice(1,   H), slice(None)), (slice(0,   H-1), slice(None)), candU_ok),
        ((slice(0,   H-1), slice(None)), (slice(1,   H),   slice(None)), candD_ok),
    ]

    p_blocks = []      # p ids
    dz_blocks = []     # dz: collection only (for debug/analysis), not used for weighting/selection
    zp_blocks = []     # z_p snap
    zq_blocks = []     # z_q snap
    r_p_blocks = []
    r_q_blocks = []
    n_q_blocks = []
    q_known_blocks = []
    q_hole_blocks  = []
    pfree_blocks   = []
    stable_alpha_gamma_blocks = []  # Plane stability gate

    for p_sl, q_sl, cand_ok_full in dirs:
        pmask = pix_disc[p_sl] & cand_ok_full[p_sl]
        if not np.any(pmask):
            p_blocks.append(np.zeros(0, dtype=np.int32))
            dz_blocks.append(np.zeros(0, dtype=F32))
            zp_blocks.append(np.zeros(0, dtype=F32))
            zq_blocks.append(np.zeros(0, dtype=F32))
            r_p_blocks.append(np.zeros((0,3), dtype=F32))
            r_q_blocks.append(np.zeros((0,3), dtype=F32))
            n_q_blocks.append(np.zeros((0,3), dtype=F32))
            q_known_blocks.append(np.zeros(0, dtype=bool))
            q_hole_blocks.append(np.zeros(0, dtype=bool))
            pfree_blocks.append(np.zeros(0, dtype=np.int32))
            stable_alpha_gamma_blocks.append(np.zeros(0, dtype=bool))
            continue

        p_ids_full = idx_map[p_sl][pmask]
        zp    = depth_in[p_sl][pmask].astype(F32)
        zq    = depth_in[q_sl][pmask].astype(F32)
        finite_zq = np.isfinite(zq)
        q_known = (known_mask[q_sl][pmask]) & finite_zq
        q_hole  = (~known_mask[q_sl][pmask])
        dz = np.abs(zp - zq).astype(F32)  # unused (for recording)

        r_p = rays[p_sl][pmask].astype(F32)
        r_q = rays[q_sl][pmask].astype(F32)
        n_q = n[q_sl][pmask].astype(F32)

        p_free = map_full2free[p_ids_full]
        alpha = np.sum(n_q * r_p, axis=-1).astype(F32)
        gamma = np.sum(n_q * r_q, axis=-1).astype(F32)
        stable = (np.abs(alpha) > eps) & (np.abs(gamma) > eps)

        p_blocks.append(p_ids_full)
        dz_blocks.append(dz)
        zp_blocks.append(zp)
        zq_blocks.append(zq)
        r_p_blocks.append(r_p)
        r_q_blocks.append(r_q)
        n_q_blocks.append(n_q)
        q_known_blocks.append(q_known)
        q_hole_blocks.append(q_hole)
        pfree_blocks.append(p_free)
        stable_alpha_gamma_blocks.append(stable)

    # ===== (Soft-Top by Normal Similarity) =====
    if use_softA:
        p_all = np.concatenate(p_blocks) if len(p_blocks) else np.zeros(0, dtype=np.int32)
        if p_all.size > 0:
            # Collect n_p corresponding to block p positions
            n_p_blocks = []
            for p_sl, q_sl, cand_ok_full in dirs:
                pmask = pix_disc[p_sl] & cand_ok_full[p_sl]
                if not np.any(pmask):
                    n_p_blocks.append(np.zeros((0,3), dtype=F32))
                else:
                    n_p_blocks.append(n[p_sl][pmask].astype(F32))
            n_p_all = np.concatenate(n_p_blocks, axis=0)
            n_q_all = np.concatenate(n_q_blocks, axis=0)

            score_all = _normal_similarity_score(n_p_all, n_q_all, beta=softA_beta)
            wA_all = _group_softmax_by_p(p_all, score_all, tau=F32(softA_tau), topk=softA_topk)

            # Block-wise decomposition
            wA_blocks = []
            ofs = 0
            for p_ids_full in p_blocks:
                L = int(p_ids_full.size)
                wA_blocks.append(wA_all[ofs:ofs+L] if L>0 else np.zeros(0, dtype=F32))
                ofs += L
        else:
            wA_blocks = [np.zeros(0, dtype=F32) for _ in p_blocks]
    else:
        wA_blocks = [np.ones(int(pb.size), dtype=F32) if pb.size>0 else np.zeros(0, dtype=F32)
                     for pb in p_blocks]

    # ===== (E) Equal: w_final = wA × gain × sqrt(lambda_equal) =====
    if lambda_equal > 0:
        lamE = F32(np.sqrt(F32(lambda_equal)))
        for bi, (p_ids_full, zq, q_known, q_hole, p_free, wA) in enumerate(
            zip(p_blocks, zq_blocks, q_known_blocks, q_hole_blocks, pfree_blocks, wA_blocks)
        ):
            if p_ids_full.size == 0:
                continue
            w_final = wA.astype(F32)
            for sel, gain in ((q_known, F32(gain_known)), (q_hole, F32(gain_hole))):
                ok = sel & (p_free >= 0)
                if not np.any(ok):
                    continue
                cols = p_free[ok]
                ww = (lamE * gain * w_final[ok]).astype(F32)
                rr = np.arange(cols.size, dtype=np.int32)
                rhs = (ww * zq[ok]).astype(F32)  # z_p = z_q
                data_buf.append(ww.astype(F32)); row_buf.append(rr + row_ofs); col_buf.append(cols)
                b_buf.append(rhs); row_ofs += cols.size

    # ===== (P) Plane: (w*alpha) z_p = (w*gamma) z_q, w = wA × gain × sqrt(lambda_plane) =====
    if lambda_plane > 0:
        lamP = F32(np.sqrt(F32(lambda_plane)))
        for bi, (p_ids_full, zq, r_p, r_q, n_q, q_known, q_hole, p_free, stable, wA) in enumerate(
            zip(p_blocks, zq_blocks, r_p_blocks, r_q_blocks, n_q_blocks,
                q_known_blocks, q_hole_blocks, pfree_blocks, stable_alpha_gamma_blocks, wA_blocks)
        ):
            if p_ids_full.size == 0:
                continue
            alpha = np.sum(n_q * r_p, axis=-1).astype(F32)
            gamma = np.sum(n_q * r_q, axis=-1).astype(F32)
            w_final = wA.astype(F32)

            for sel, gain in ((q_known, F32(gain_known)), (q_hole, F32(gain_hole))):
                ok = sel & (p_free >= 0) & stable
                if not np.any(ok):
                    continue
                cols = p_free[ok]
                ww = (lamP * gain * w_final[ok]).astype(F32)
                rr = np.arange(cols.size, dtype=np.int32)
                rhs = (ww * gamma[ok] * zq[ok]).astype(F32)
                coef = (ww * alpha[ok]).astype(F32)
                data_buf.append(coef.astype(F32)); row_buf.append(rr + row_ofs); col_buf.append(cols)
                b_buf.append(rhs.astype(F32)); row_ofs += cols.size

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

    # (K) Keep-known: strongly maintain known (only for boundary-included free variables)
    if lambda_keep > 0:
        lamK = F32(np.sqrt(F32(lambda_keep)))
        ids_full = np.arange(Npix, dtype=np.int32)[known_mask.reshape(-1)]
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

    # ===== Assembly and solve =====
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
    return out
