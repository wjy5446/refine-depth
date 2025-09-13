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
    """픽셀 광선 r_p = K^{-1}[x,y,1]^T (HxWx3), L2 정규화. (float32 고정)"""
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
    # float32-friendly smoother 조합
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
        B=np.ones((AtA_csr32.shape[0], 1), dtype=np.float32),  # near-nullspace (상수 모드)
    )
    _PYAMG_CACHE[sig] = ml
    while len(_PYAMG_CACHE) > _MAX_AMG_CACHE:
        _PYAMG_CACHE.popitem(last=False)
    return ml


def solve_with_pyamg32(A_csr32, b32, *, maxiter=2, x0=None, cycle="V", sig: str):
    """정상방정식 형태(AtA z = Atb)를 float32로 AMG V-cycle 1~2회."""
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
    """AMG를 preconditioner로 쓰는 PCG (AtA z=Atb). 종종 더 빠르다."""
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
    lambda_smooth: float = 0.2,     # (S)
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
    자유변수 축소: free = hole ∪ 경계.
    (N) 경계 쌍 제외, 비경계에서 2변수. 단, 한쪽이 비자유면 단항으로 축약.
    (E,P) 경계 p만 변수, q는 스냅샷 (known/hole 모두 허용).
    (R) 비경계 hole 서브샘플 stride 단항.
    (K) known 고정 단항.
    float32로 조립/해결, 패턴 시그니처 캐시 + warm-start 적용.
    """
    depth_in = depth_in.astype(F32, copy=False)
    known_mask = known_mask.astype(bool, copy=False)
    hole_mask = hole_mask.astype(bool, copy=False)
    H, W = depth_in.shape
    Npix = H * W
    eps = F32(1e-6)

    # K / n / rays
    if K is None:
        K = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=F32)
    else:
        K = K.astype(F32, copy=False)
    if n_guide is None:
        n = np.zeros((H, W, 3), dtype=F32); n[..., 2] = F32(1.0)
    else:
        n = n_guide.astype(F32, copy=False)
    rays = _make_rays(K, H, W)

    # 노멀 정규화/플립/NaN 처리
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

    # 경계/게이트
    disc = detect_discontinuities(depth_in, n=n, use_normals=False) if (discontinuity_maps is None) else discontinuity_maps
    fused = gates_from_disc_1d(disc)
    mask_h, mask_v = fused["mask_h"], fused["mask_v"]
    pix_disc = fused["pix_disc"]
    candL_ok = fused["candL_ok"]; candR_ok = fused["candR_ok"]
    candU_ok = fused["candU_ok"]; candD_ok = fused["candD_ok"]

    # ===== 자유변수 축소 =====
    # 경계 또는 hole만 변수. 필요 시 1-ring 확장해도 됨.
    free = (pix_disc | hole_mask)
    free_vec = free.reshape(-1)
    map_full2free = -np.ones(Npix, dtype=np.int32)
    free_ids = np.where(free_vec)[0].astype(np.int32)
    map_full2free[free_ids] = np.arange(free_ids.size, dtype=np.int32)

    # big buffers (float32)
    data_buf = []; row_buf = []; col_buf = []; b_buf = []
    row_ofs = 0

    def _append_rows(vals: np.ndarray, ridx_local: np.ndarray, cidx_full: np.ndarray, rhs: np.ndarray):
        """열 인덱스(full) → free 공간으로 투영. 비자유 열은 버리고 rhs는 그대로."""
        nonlocal row_ofs
        if vals.size == 0:
            return
        cols_free = map_full2free[cidx_full]
        ok = cols_free >= 0
        if not np.any(ok):
            return
        rr = ridx_local + row_ofs
        # 일부 열이 버려지면 행이 듬성일 수 있으므로, 여기선 '각 항이 1열짜리'라고 가정해 추가한다.
        data_buf.append(vals[ok].astype(F32, copy=False))
        row_buf.append(rr[:np.count_nonzero(ok)])
        col_buf.append(cols_free[ok])
        b_buf.append(rhs[ok].astype(F32, copy=False))
        row_ofs += rhs.shape[0]  # 원래 행 수만큼 증가 (각 항 1방정식)

    def _append_rows_2var(c_p: np.ndarray, c_q: np.ndarray, ridx_local: np.ndarray,
                          p_full: np.ndarray, q_full: np.ndarray, rhs: np.ndarray,
                          z_p_snap: np.ndarray, z_q_snap: np.ndarray, w: np.ndarray):
        """
        (N)용: c_p*d_p + c_q*d_q = rhs_form(보통 0). 자유변수 조합에 따라:
        - 둘 다 free: 2열 추가
        - p만 free: 1열(c_p) + rhs에 (-c_q*z_q_snap)
        - q만 free: 1열(c_q) + rhs에 (-c_p*z_p_snap)
        - none: 버림
        """
        nonlocal row_ofs
        if c_p.size == 0:
            return
        p_free = map_full2free[p_full]
        q_free = map_full2free[q_full]

        both = (p_free >= 0) & (q_free >= 0)
        only_p = (p_free >= 0) & (q_free < 0)
        only_q = (p_free < 0) & (q_free >= 0)

        # 둘 다 free → 2개 항을 같은 행 인덱스에 추가
        if np.any(both):
            Kk = int(np.count_nonzero(both))
            rr = np.arange(Kk, dtype=np.int32) + row_ofs
            # p 항
            data_buf.append((w[both] * c_p[both]).astype(F32))
            row_buf.append(rr)
            col_buf.append(p_free[both])
            # q 항
            data_buf.append((w[both] * c_q[both]).astype(F32))
            row_buf.append(rr)
            col_buf.append(q_free[both])
            # rhs
            b_buf.append((w[both] * rhs[both]).astype(F32))
            row_ofs += Kk

        # p만 free → 단항, rhs 보정
        if np.any(only_p):
            Kk = int(np.count_nonzero(only_p))
            rr = np.arange(Kk, dtype=np.int32) + row_ofs
            data_buf.append((w[only_p] * c_p[only_p]).astype(F32))
            row_buf.append(rr)
            col_buf.append(p_free[only_p])
            rhs_corr = (rhs[only_p] - c_q[only_p] * z_q_snap[only_p]).astype(F32)
            b_buf.append((w[only_p] * rhs_corr).astype(F32))
            row_ofs += Kk

        # q만 free → 단항, rhs 보정
        if np.any(only_q):
            Kk = int(np.count_nonzero(only_q))
            rr = np.arange(Kk, dtype=np.int32) + row_ofs
            data_buf.append((w[only_q] * c_q[only_q]).astype(F32))
            row_buf.append(rr)
            col_buf.append(q_free[only_q])
            rhs_corr = (rhs[only_q] - c_p[only_q] * z_p_snap[only_q]).astype(F32)
            b_buf.append((w[only_q] * rhs_corr).astype(F32))
            row_ofs += Kk

    # ==== 기하량 ====
    rx = (rays[:, 1:, :] - rays[:, :-1, :]).astype(F32)
    ry = (rays[1:, :, :] - rays[:-1, :, :]).astype(F32)

    # ===== (N) 노멀 정합: 경계쌍 제외, 2변수 (필요 시 단항 축약) =====
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

    # ===== (E,P): 경계 p만 변수, q는 스냅샷 =====
    if (lambda_equal > 0) or (lambda_plane > 0):
        lamE = F32(np.sqrt(F32(lambda_equal))) if lambda_equal > 0 else F32(0.0)
        lamP = F32(np.sqrt(F32(lambda_plane))) if lambda_plane > 0 else F32(0.0)
        se2  = F32(2.0) * (np.maximum(F32(sigma_equal), F32(1e-8)) ** F32(2.0))
        sp2  = F32(2.0) * (np.maximum(F32(sigma_plane), F32(1e-8)) ** F32(2.0))

        idx_map = np.arange(H * W, dtype=np.int32).reshape(H, W)
        dirs = [
            ((slice(None), slice(1,   W)), (slice(None), slice(0,   W-1)), fused["candL_ok"]),
            ((slice(None), slice(0,   W-1)), (slice(None), slice(1,   W)), fused["candR_ok"]),
            ((slice(1,   H), slice(None)), (slice(0,   H-1), slice(None)), fused["candU_ok"]),
            ((slice(0,   H-1), slice(None)), (slice(1,   H),   slice(None)), fused["candD_ok"]),
        ]

        for p_sl, q_sl, cand_ok_full in dirs:
            pmask = pix_disc[p_sl] & cand_ok_full[p_sl]
            if not np.any(pmask):
                continue

            p_ids_full = idx_map[p_sl][pmask]
            zp    = depth_in[p_sl][pmask].astype(F32)
            zq    = depth_in[q_sl][pmask].astype(F32)
            finite_zq = np.isfinite(zq)
            q_known = (known_mask[q_sl][pmask]) & finite_zq
            q_hole  = (~known_mask[q_sl][pmask])

            n_q = n[q_sl][pmask].astype(F32)
            r_p = rays[p_sl][pmask].astype(F32)
            r_q = rays[q_sl][pmask].astype(F32)

            dz = np.abs(zp - zq).astype(F32)
            wE_base = np.exp(-(dz*dz)/se2).astype(F32) if lambda_equal > 0 else None
            wP_base = np.exp(-(dz*dz)/sp2).astype(F32) if lambda_plane > 0 else None

            if lambda_equal > 0:
                w_known = (lamE * F32(gain_known) * wE_base).astype(F32)
                w_hole  = (lamE * F32(gain_hole)  * wE_base).astype(F32)

                for sel in (q_known, q_hole):
                    ok = sel & (map_full2free[p_ids_full] >= 0)
                    if not np.any(ok):
                        continue
                    p_free = map_full2free[p_ids_full[ok]]
                    ww = (w_known if sel is q_known else w_hole)[ok]
                    rr = np.arange(p_free.size, dtype=np.int32)
                    rhs = (ww * zq[ok]).astype(F32)
                    # 열(자유변수)만 추가
                    data_buf.append(ww.astype(F32))
                    row_buf.append(rr + row_ofs)
                    col_buf.append(p_free)
                    b_buf.append(rhs)
                    row_ofs += p_free.size

            if lambda_plane > 0:
                alpha = np.sum(n_q * r_p, axis=-1).astype(F32)   # (n_q·r_p)
                gamma = np.sum(n_q * r_q, axis=-1).astype(F32)   # (n_q·r_q)
                stable = (np.abs(alpha) > eps) & (np.abs(gamma) > eps)
                w_known = (lamP * F32(gain_known) * wP_base).astype(F32)
                w_hole  = (lamP * F32(gain_hole)  * wP_base).astype(F32)

                for sel, w_base in ((q_known, w_known), (q_hole, w_hole)):
                    ok = sel & stable & (map_full2free[p_ids_full] >= 0)
                    if not np.any(ok):
                        continue
                    p_free = map_full2free[p_ids_full[ok]]
                    ww = (w_base[ok] * alpha[ok]).astype(F32)
                    rr = np.arange(p_free.size, dtype=np.int32)
                    rhs = (w_base[ok] * gamma[ok] * zq[ok]).astype(F32)
                    data_buf.append(ww.astype(F32))
                    row_buf.append(rr + row_ofs)
                    col_buf.append(p_free)
                    b_buf.append(rhs.astype(F32))
                    row_ofs += p_free.size

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

    # (K) Keep-known: known을 강하게 고정
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

    # 패턴 시그니처 (축약 공간 기준)
    sig = _pattern_sig(rows, cols, (R, Ncols))

    # COO → CSR (float32)
    A = coo_matrix((data, (rows, cols)), shape=(R, Ncols), dtype=np.float32).tocsr()

    # warm-start: 직전 해 or depth_in의 free 부분
    x0 = _PREV_SOL.get(sig, depth_in.reshape(-1)[free_ids].astype(np.float32, copy=False))

    # AMG or PCG(+AMG)
    if use_pcg:
        z_free32 = solve_with_pcg_pyamg32(A, b, cg_maxiter=cg_maxiter, x0=x0, sig=sig)
    else:
        z_free32 = solve_with_pyamg32(A, b, maxiter=maxiter, x0=x0, cycle=pyamg_cycle, sig=sig)

    # 결과 합치기
    out = depth_in.copy().astype(F32, copy=False)
    out_flat = out.reshape(-1)
    out_flat[free_ids] = z_free32.astype(F32, copy=False)
    _PREV_SOL[sig] = out_flat[free_ids].copy()  # 다음 프레임 warm-start
    return out
