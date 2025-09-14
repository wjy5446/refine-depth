import numpy as np
from scipy.sparse import coo_matrix
import pyamg
from pyamg.krylov import cg as pyamg_cg
import hashlib
from collections import OrderedDict

# === 전역 상수 및 캐시 ===
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


def detect_discontinuities(
    depth_in: np.ndarray,
    *,
    n: np.ndarray | None = None,     # (H,W,3)
    tau_rel: float = 0.05,
    tau_abs: float | None = None,
    use_normals: bool = False,
    tau_n_cos: float = 0.94,
    normal_logic: str = "or",
) -> np.ndarray:
    depth_in = depth_in.astype(F32, copy=False)
    H, W = depth_in.shape

    dz_h = np.abs(depth_in[:, 1:] - depth_in[:, :-1]).astype(F32)
    dz_v = np.abs(depth_in[1:, :] - depth_in[:-1, :]).astype(F32)
    thr_h_rel = F32(tau_rel) * np.maximum(depth_in[:, 1:], depth_in[:, :-1]).astype(F32)
    thr_v_rel = F32(tau_rel) * np.maximum(depth_in[1:, :], depth_in[:-1, :]).astype(F32)
    if tau_abs is None:
        thr_h, thr_v = thr_h_rel, thr_v_rel
    else:
        thr_h = np.maximum(thr_h_rel, F32(tau_abs)).astype(F32)
        thr_v = np.maximum(thr_v_rel, F32(tau_abs)).astype(F32)
    disc_h_d = dz_h > thr_h
    disc_v_d = dz_v > thr_v

    if use_normals and (n is not None):
        nn = n.astype(F32, copy=False)
        nn = nn / np.clip(np.linalg.norm(nn, axis=2, keepdims=True).astype(F32), F32(1e-6), None)
        cos_h = np.abs(np.sum(nn[:, :-1, :] * nn[:, 1:, :], axis=2)).astype(F32)
        cos_v = np.abs(np.sum(nn[:-1, :, :] * nn[1:, :, :], axis=2)).astype(F32)
        disc_h_n = cos_h < F32(tau_n_cos)
        disc_v_n = cos_v < F32(tau_n_cos)
    else:
        disc_h_n = np.zeros_like(disc_h_d, dtype=bool)
        disc_v_n = np.zeros_like(disc_v_d, dtype=bool)

    logic = normal_logic.lower()
    if logic == "or":
        disc_h = disc_h_d | disc_h_n
        disc_v = disc_v_d | disc_v_n
    elif logic == "and":
        disc_h = disc_h_d & disc_h_n
        disc_v = disc_v_d & disc_v_n
    elif logic == "only":
        disc_h = disc_h_n
        disc_v = disc_v_n
    else:
        raise ValueError("normal_logic must be 'or'|'and'|'only'")

    H, W = depth_in.shape
    disc = np.zeros((H, W), dtype=bool)
    disc[:, 1:]  |= disc_h
    disc[:, :-1] |= disc_h
    disc[1:, :]  |= disc_v
    disc[:-1, :] |= disc_v
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

# =================== 패턴 해시 기반 AMG 캐시 ===================

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
    A = A_csr32
    b = b32
    A = A.tocoo(); A.sum_duplicates(); A = A.tocsr()

    AtA = (A.T @ A).tocsr().astype(np.float32, copy=False)
    Atb = (A.T @ b).astype(np.float32, copy=False)

    ml = _get_pyamg_solver_by_sig_float32(AtA, sig)
    x0_ = None if x0 is None else x0.astype(np.float32, copy=False)
    z32 = ml.solve(Atb, x0=x0_, tol=1e-4, maxiter=int(maxiter), cycle=cycle)
    return z32


def solve_with_pcg_pyamg32(A_csr32, b32, *, cg_maxiter=10, x0=None, sig: str):
    A = A_csr32
    b = b32
    A = A.tocoo(); A.sum_duplicates(); A = A.tocsr()

    AtA = (A.T @ A).tocsr().astype(np.float32, copy=False)
    Atb = (A.T @ b).astype(np.float32, copy=False)

    ml = _get_pyamg_solver_by_sig_float32(AtA, sig)
    M = ml.aspreconditioner()
    x0_ = None if x0 is None else x0.astype(np.float32, copy=False)
    z32, info = pyamg_cg(AtA, Atb, tol=1e-4, maxiter=int(cg_maxiter), M=M, x0=x0_, callback=None, residuals=None)
    return z32

# ================= Main (축약/float32/캐시/warm-start) =================

def refine_depth_normal_alignment(
    depth_in: np.ndarray,
    known_mask: np.ndarray,
    hole_mask: np.ndarray,
    n_guide: np.ndarray | None,
    K: np.ndarray | None,
    discontinuity_maps: np.ndarray | None = None,  # (H,W) bool
    lambda_normal: float = 3.0,     # (N)
    lambda_smooth: float = 0.2,     # (S)  # 현재 예시엔 사용 X
    lambda_equal: float = 1.0,      # (E)
    lambda_plane: float = 1.0,      # (P)
    lambda_screen: float = 1e-3,    # (R)
    lambda_keep: float = 30.0,      # (K)
    # 가중/게인
    sigma_equal: float = 0.01,
    sigma_plane: float = 0.01,
    gain_known: float = 2.0,
    gain_hole: float  = 1.0,
    # AMG/PCG
    use_pcg: bool = False,          # True면 PCG+AMG, False면 AMG.solve
    pyamg_cycle: str = "V",         # "V"|"W"
    maxiter: int = 2,               # AMG V-cycle 횟수 (1~2 권장)
    cg_maxiter: int = 8,            # PCG 반복수 (5~10 권장)
) -> np.ndarray:
    """
    [A-조건 하드 컷오프 적용 버전]
    - E/P 후보를 p별로 모아 dz 기반 Top-1(+근처만) 유지.
    - 그 외 구조/수치는 기존과 동일.
    """
    # ==== (A) 하드 컷 파라미터 ====
    kappa_z_equal   = F32(1.5)     # 절대 임계 배수 (E)
    kappa_z_plane   = F32(1.5)     # 절대 임계 배수 (P)
    delta_abs_equal = F32(5e-4)    # min 근처 허용폭 (E)
    delta_abs_plane = F32(5e-4)    # min 근처 허용폭 (P)
    rho_ratio_equal = F32(1.2)     # 2등/1등 비율 임계 (E)
    rho_ratio_plane = F32(1.2)     # 2등/1등 비율 임계 (P)

    def _build_keep_mask(block_lengths, p_all, dz_all, sigma, kappa_z, delta_abs, rho_ratio):
        """
        모든 방향에서 모은 (p, dz) 시퀀스에 대해 p별 Top-1(+근처)만 True.
        block_lengths: 각 방향 블록 길이 리스트 -> 슬라이스로 다시 돌려주기 위해 필요.
        반환: keep_mask_all (전체 길이), 그리고 블록별 슬라이스 인덱스
        """
        total = int(np.sum(block_lengths))
        if total == 0:
            return np.zeros(0, dtype=bool), []

        # p 기준 그룹화
        order = np.argsort(p_all)
        p_sorted  = p_all[order]
        dz_sorted = dz_all[order]
        cuts = np.flatnonzero(np.r_[True, p_sorted[1:] != p_sorted[:-1], True])

        keep_sorted = np.zeros_like(dz_sorted, dtype=bool)

        # p-세그먼트별 컷
        for s, e in zip(cuts[:-1], cuts[1:]):
            seg = dz_sorted[s:e]
            if seg.size == 0:
                continue
            i_min = int(np.argmin(seg))
            dz1 = seg[i_min]
            dz2 = np.partition(seg, 1)[1] if seg.size > 1 else np.inf

            cond_abs  = seg <= (kappa_z * F32(sigma))
            cond_near = seg <= (dz1 + delta_abs)
            keep = cond_abs & cond_near

            if (dz2 / (dz1 + F32(1e-12))) >= rho_ratio:
                keep = np.zeros_like(keep, dtype=bool)
                keep[i_min] = True

            keep_sorted[s:e] = keep

        # 원래 순서 복원
        keep_mask_all = np.zeros_like(keep_sorted, dtype=bool)
        keep_mask_all[order] = keep_sorted

        # 블록 슬라이스 경계
        slices = []
        ofs = 0
        for L in block_lengths:
            slices.append(slice(ofs, ofs + int(L)))
            ofs += int(L)
        return keep_mask_all, slices

    # ==== 입력 전처리 ====
    depth_in = depth_in.astype(F32, copy=False)
    known_mask = known_mask.astype(bool, copy=False)
    hole_mask = hole_mask.astype(bool, copy=False)
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

    disc = detect_discontinuities(depth_in, n=n, use_normals=False) if (discontinuity_maps is None) else discontinuity_maps
    fused = gates_from_disc_1d(disc)
    mask_h, mask_v = fused["mask_h"], fused["mask_v"]
    pix_disc = fused["pix_disc"]
    candL_ok = fused["candL_ok"]; candR_ok = fused["candR_ok"]
    candU_ok = fused["candU_ok"]; candD_ok = fused["candD_ok"]

    # 자유변수: 경계 ∪ hole
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

    # 기하량
    rx = (rays[:, 1:, :] - rays[:, :-1, :]).astype(F32)
    ry = (rays[1:, :, :] - rays[:-1, :, :]).astype(F32)

    # (N) 노멀 정합: 경계쌍 제외
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

    # ===== (E,P) 공통: 먼저 모든 방향 후보를 모아 p별 keep-mask 계산 (A 조건) =====
    idx_map = np.arange(H * W, dtype=np.int32).reshape(H, W)
    dirs = [
        ((slice(None), slice(1,   W)), (slice(None), slice(0,   W-1)), fused["candL_ok"]),
        ((slice(None), slice(0,   W-1)), (slice(None), slice(1,   W)), fused["candR_ok"]),
        ((slice(1,   H), slice(None)), (slice(0,   H-1), slice(None)), fused["candU_ok"]),
        ((slice(0,   H-1), slice(None)), (slice(1,   H),   slice(None)), fused["candD_ok"]),
    ]

    # 공통 p, dz 수집 (한 번만 계산해서 E/P에서 각각 컷 기준만 달리 적용)
    p_blocks = []      # list of (p_ids_full_block)
    dz_blocks = []     # list of (dz block)
    block_lengths = [] # lengths for slicing back
    zp_blocks = []     # zp snapshots (그대로 사용)
    zq_blocks = []     # zq snapshots (그대로 사용)
    r_p_blocks = []    # r_p
    r_q_blocks = []    # r_q
    n_q_blocks = []    # n_q
    q_known_blocks = []# q_known mask
    q_hole_blocks  = []# q_hole  mask
    pfree_blocks   = []# map_full2free[p_ids_full] (미리 캐시)
    stable_alpha_gamma_blocks = []  # |alpha|,|gamma|>eps (P 전용)

    for p_sl, q_sl, cand_ok_full in dirs:
        pmask = pix_disc[p_sl] & cand_ok_full[p_sl]
        if not np.any(pmask):
            # 빈 블록 자리 채우기
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
            block_lengths.append(0)
            continue

        p_ids_full = idx_map[p_sl][pmask]
        zp    = depth_in[p_sl][pmask].astype(F32)
        zq    = depth_in[q_sl][pmask].astype(F32)
        finite_zq = np.isfinite(zq)
        q_known = (known_mask[q_sl][pmask]) & finite_zq
        q_hole  = (~known_mask[q_sl][pmask])
        dz = np.abs(zp - zq).astype(F32)

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
        block_lengths.append(dz.size)

    if (lambda_equal > 0) or (lambda_plane > 0):
        # === 전체 concat (E/P용 keep-mask를 별도로 계산) ===
        p_all  = np.concatenate(p_blocks) if len(p_blocks) else np.zeros(0, dtype=np.int32)
        dz_all = np.concatenate(dz_blocks) if len(dz_blocks) else np.zeros(0, dtype=F32)

        keepE_all = np.ones_like(dz_all, dtype=bool)
        keepP_all = np.ones_like(dz_all, dtype=bool)

        if lambda_equal > 0 and dz_all.size > 0:
            keepE_all, slicesE = _build_keep_mask(
                block_lengths, p_all, dz_all,
                F32(sigma_equal), kappa_z_equal, delta_abs_equal, rho_ratio_equal
            )
        else:
            slicesE = [slice(0,0) for _ in block_lengths]

        if lambda_plane > 0 and dz_all.size > 0:
            keepP_all, slicesP = _build_keep_mask(
                block_lengths, p_all, dz_all,
                F32(sigma_plane), kappa_z_plane, delta_abs_plane, rho_ratio_plane
            )
        else:
            slicesP = [slice(0,0) for _ in block_lengths]

        # ===== (E) Equal 조립: keepE_all 컷 적용 =====
        if lambda_equal > 0:
            lamE = F32(np.sqrt(F32(lambda_equal)))
            se2  = F32(2.0) * (np.maximum(F32(sigma_equal), F32(1e-8)) ** F32(2.0))

            for bi, (p_ids_full, dz, zq, q_known, q_hole, p_free) in enumerate(
                zip(p_blocks, dz_blocks, zq_blocks, q_known_blocks, q_hole_blocks, pfree_blocks)
            ):
                if p_ids_full.size == 0:
                    continue
                wE_base = np.exp(-(dz*dz)/se2).astype(F32)
                keep_slice = keepE_all[slicesE[bi]]
                # known/hole 별로 컷 적용
                for sel, gain in ((q_known, F32(gain_known)), (q_hole, F32(gain_hole))):
                    ok = sel & (p_free >= 0) & keep_slice
                    if not np.any(ok):
                        continue
                    cols = p_free[ok]
                    ww = (lamE * gain * wE_base[ok]).astype(F32)
                    rr = np.arange(cols.size, dtype=np.int32)
                    rhs = (ww * zq[ok]).astype(F32)
                    data_buf.append(ww.astype(F32)); row_buf.append(rr + row_ofs); col_buf.append(cols)
                    b_buf.append(rhs); row_ofs += cols.size

        # ===== (P) Plane 조립: keepP_all 컷 + 안정 게이트 적용 =====
        if lambda_plane > 0:
            lamP = F32(np.sqrt(F32(lambda_plane)))
            sp2  = F32(2.0) * (np.maximum(F32(sigma_plane), F32(1e-8)) ** F32(2.0))

            for bi, (p_ids_full, dz, zq, r_p, r_q, n_q, q_known, q_hole, p_free, stable) in enumerate(
                zip(p_blocks, dz_blocks, zq_blocks, r_p_blocks, r_q_blocks, n_q_blocks,
                    q_known_blocks, q_hole_blocks, pfree_blocks, stable_alpha_gamma_blocks)
            ):
                if p_ids_full.size == 0:
                    continue
                alpha = np.sum(n_q * r_p, axis=-1).astype(F32)
                gamma = np.sum(n_q * r_q, axis=-1).astype(F32)
                wP_base = np.exp(-(dz*dz)/sp2).astype(F32)
                keep_slice = keepP_all[slicesP[bi]]

                for sel, gain in ((q_known, F32(gain_known)), (q_hole, F32(gain_hole))):
                    ok = sel & (p_free >= 0) & keep_slice & stable
                    if not np.any(ok):
                        continue
                    cols = p_free[ok]
                    ww = (lamP * gain * wP_base[ok]).astype(F32)
                    rr = np.arange(cols.size, dtype=np.int32)
                    rhs = (ww * gamma[ok] * zq[ok]).astype(F32)
                    coef = (ww * alpha[ok]).astype(F32)  # (w*alpha) * z_p = (w*gamma) * z_q
                    data_buf.append(coef.astype(F32)); row_buf.append(rr + row_ofs); col_buf.append(cols)
                    b_buf.append(rhs.astype(F32)); row_ofs += cols.size

    # (R) Screen: 비-경계 hole만 + stride(2,2)
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

    # (K) Keep-known
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

    # ===== 조립 및 해 =====
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
