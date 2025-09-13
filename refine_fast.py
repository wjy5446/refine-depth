import numpy as np
from scipy.sparse import coo_matrix, diags
import pyamg

# === 전역 상수 ===
F32 = np.float32

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
    n: np.ndarray | None = None,     # (H,W,3) 단위 법선(옵션)
    tau_rel: float = 0.05,           # 깊이 상대 임계
    tau_abs: float | None = None,    # 깊이 절대 임계
    use_normals: bool = False,       # 법선 사용 여부 (기본 OFF)
    tau_n_cos: float = 0.94,         # 법선 코사인 임계(≈20°)
    normal_logic: str = "or",        # 'or'|'and'|'only'
) -> np.ndarray:
    """단일 픽셀 불연속 맵 disc(H,W)을 만든다. (전연산 float32)"""
    depth_in = depth_in.astype(F32, copy=False)
    H, W = depth_in.shape

    # 깊이 기반 쌍 경계
    dz_h = np.abs(depth_in[:, 1:] - depth_in[:, :-1]).astype(F32)
    dz_v = np.abs(depth_in[1:, :] - depth_in[:-1, :]).astype(F32)
    thr_h_rel = F32(tau_rel) * np.maximum(depth_in[:, 1:], depth_in[:, :-1]).astype(F32)
    thr_v_rel = F32(tau_rel) * np.maximum(depth_in[1:, :], depth_in[:-1, :]).astype(F32)
    if tau_abs is None:
        thr_h = thr_h_rel
        thr_v = thr_v_rel
    else:
        thr_h = np.maximum(thr_h_rel, F32(tau_abs)).astype(F32)
        thr_v = np.maximum(thr_v_rel, F32(tau_abs)).astype(F32)
    disc_h_d = dz_h > thr_h
    disc_v_d = dz_v > thr_v

    # 법선 기반 쌍 경계(옵션)
    if use_normals and (n is not None):
        nn = n.astype(F32, copy=False)
        nn = nn / np.clip(np.linalg.norm(nn, axis=2, keepdims=True).astype(F32), F32(1e-6), None)
        cos_h = np.abs(np.sum(nn[:, :-1, :] * nn[:, 1:, :], axis=2)).astype(F32)  # (H,W-1)
        cos_v = np.abs(np.sum(nn[:-1, :, :] * nn[1:, :, :], axis=2)).astype(F32)  # (H-1,W)
        disc_h_n = cos_h < F32(tau_n_cos)
        disc_v_n = cos_v < F32(tau_n_cos)
    else:
        disc_h_n = np.zeros_like(disc_h_d, dtype=bool)
        disc_v_n = np.zeros_like(disc_v_d, dtype=bool)

    # 결합
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

    # 픽셀 1D 경계로 승격
    disc = np.zeros((H, W), dtype=bool)
    disc[:, 1:]  |= disc_h
    disc[:, :-1] |= disc_h
    disc[1:, :]  |= disc_v
    disc[:-1, :] |= disc_v
    return disc


def gates_from_disc_1d(disc: np.ndarray) -> dict:
    """
    1D 픽셀 경계맵 disc(H,W)로부터:
      - mask_h, mask_v: 쌍 사용 가능(True=통과)  (경계쌍 제외)
      - candL/R/U/D_ok: 이웃 후보 허용(True=허용)   (경계 넘지 않기)
      - pix_disc: 픽셀 경계
    """
    H, W = disc.shape
    pair_disc_h = disc[:, :-1] | disc[:, 1:]   # (H,W-1)
    pair_disc_v = disc[:-1, :] | disc[1:, :]   # (H-1,W)

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

# =================== PyAMG-only solver ===================

_PYAMG_CACHE = {}

def _get_pyamg_solver(AtA_csr):
    """AtA의 스패스 패턴(shape, nnz)이 같으면 PyAMG 계층을 재사용."""
    key = (AtA_csr.shape, AtA_csr.nnz)
    ml = _PYAMG_CACHE.get(key)
    if ml is None:
        # 라플라시안+대각 구조에 강한 Smoothed Aggregation AMG
        ml = pyamg.smoothed_aggregation_solver(AtA_csr)
        _PYAMG_CACHE[key] = ml
    return ml


def solve_with_pyamg_only(A_csr, b, *, tol=1e-4, maxiter=100, x0=None, cycle="V"):
    """
    외부 CG/LSMR 없이, PyAMG 멀티그리드만으로 정규방정식 해결:
        (A^T A) z = (A^T b)
    """
    A64 = A_csr.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)

    AtA = (A64.T @ A64).tocsr()
    Atb = (A64.T @ b64)

    ml = _get_pyamg_solver(AtA)
    x0_ = None if x0 is None else x0.astype(np.float64, copy=False)
    z64 = ml.solve(Atb, x0=x0_, tol=float(tol), maxiter=int(maxiter), cycle=cycle)
    return z64

# ================= Main (조립 + PyAMG 해) =================

def refine_depth_normal_alignment(
    depth_in: np.ndarray,
    known_mask: np.ndarray,
    hole_mask: np.ndarray,
    n_guide: np.ndarray | None,
    K: np.ndarray | None,
    discontinuity_maps: np.ndarray | None = None,  # (H,W) bool map
    lambda_normal: float = 3.0,
    lambda_smooth: float = 0.2,
    lambda_equal: float = 1.0,
    lambda_plane: float = 1.0,
    lambda_screen: float = 1e-3,
    lambda_keep: float = 30.0,
    tol: float = 1e-4,
    maxiter: int = 100,
    pyamg_cycle: str = "V",          # "V"|"W"
) -> np.ndarray:
    """
    경계 기반 게이팅:
      - (N) 노멀: 경계 쌍 제외 (mask_h/v)
      - (S) 스무딩: 경계 쌍 제외 (mask_h/v)
      - (E) equal: **경계 p만 변수**, q는 스냅샷 깊이로 단항식
      - (P) plane: **경계 p만 변수**, q는 스냅샷 깊이로 단항식
      - (R) screen: 비-경계 hole만
      - (K) keep-known: 관측 고정
    조립은 float32, 해법은 float64(PAMG), 출력은 float32.
    """
    depth_in = depth_in.astype(F32, copy=False)
    known_mask = known_mask.astype(bool, copy=False)
    hole_mask = hole_mask.astype(bool, copy=False)
    H, W = depth_in.shape

    # 파라미터
    sigma_equal  = F32(0.01)
    sigma_plane  = F32(0.01)
    gain_known   = F32(2.0)
    gain_hole    = F32(1.0)
    eps          = F32(1e-6)

    # 변수 인덱스
    idx_map = np.arange(H * W, dtype=np.int32).reshape(H, W)
    N = H * W

    # intrinsics / normals
    if K is None:
        K = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=F32)
    else:
        K = K.astype(F32, copy=False)

    if n_guide is None:
        n = np.zeros((H, W, 3), dtype=F32); n[..., 2] = F32(1.0)
    else:
        n = n_guide.astype(F32, copy=False)

    rays = _make_rays(K, H, W)

    # 노멀 정규화 + 카메라를 향하도록 플립
    n_norm = np.linalg.norm(n, axis=2, keepdims=True).astype(F32)
    bad = (n_norm < F32(1e-6))
    if bad.any():
        n[bad[..., 0]] = np.array([0, 0, 1], dtype=F32)
        n_norm = np.linalg.norm(n, axis=2, keepdims=True).astype(F32)
    n = n / np.clip(n_norm, F32(1e-6), None)
    dot = np.sum(n * rays, axis=2).astype(F32)
    n[dot > F32(0.0)] *= F32(-1.0)

    # Ray differences
    rx = (rays[:, 1:, :] - rays[:, :-1, :]).astype(F32)
    ry = (rays[1:, :, :] - rays[:-1, :, :]).astype(F32)

    # ---- Discontinuity detection ----
    disc = detect_discontinuities(depth_in, n=n, use_normals=False) if (discontinuity_maps is None) else discontinuity_maps
    fused = gates_from_disc_1d(disc)
    mask_h, mask_v = fused["mask_h"], fused["mask_v"]
    pix_disc = fused["pix_disc"]
    candL_ok = fused["candL_ok"]; candR_ok = fused["candR_ok"]
    candU_ok = fused["candU_ok"]; candD_ok = fused["candD_ok"]

    # ----- 빅 COO/벡터 버퍼 (float32) -----
    data_buf = []; row_buf = []; col_buf = []; b_buf = []
    row_ofs = 0
    def _append_block(vals, ridx, cidx, rhs):
        nonlocal row_ofs
        if vals.size == 0:
            return
        row_buf.append(ridx + row_ofs)
        col_buf.append(cidx)
        data_buf.append(vals.astype(F32, copy=False))
        b_buf.append(rhs.astype(F32, copy=False))
        row_ofs += rhs.size

    # ===== (N) 노멀 정합: 경계 쌍 제외 =====
    if lambda_normal > 0:
        lamN = np.sqrt(F32(lambda_normal)).astype(F32)

        # Horizontal (p: left, q: right)
        n_p = n[:, :-1, :].reshape(-1, 3)
        r_p = rays[:, :-1, :].reshape(-1, 3)
        rdx = rx.reshape(-1, 3)
        p_idx = idx_map[:, :-1].reshape(-1)
        q_idx = idx_map[:,  1:].reshape(-1)
        ok_h = mask_h.reshape(-1)

        a = np.sum(n_p * r_p, axis=1).astype(F32)
        b = np.sum(n_p * rdx, axis=1).astype(F32)
        a = np.nan_to_num(a).astype(F32); b = np.nan_to_num(b).astype(F32)
        c_p = (b - a).astype(F32); c_q = a
        den = np.sqrt(c_p * c_p + c_q * c_q).astype(F32) + F32(1e-6)
        c_p = (c_p / den).astype(F32); c_q = (c_q / den).astype(F32)

        if np.any(ok_h):
            Kk = int(np.count_nonzero(ok_h))
            rr = np.arange(Kk, dtype=np.int32)
            ww = (lamN * np.ones(Kk, dtype=F32))
            vals = np.concatenate([(ww * c_p[ok_h]).astype(F32),
                                   (ww * c_q[ok_h]).astype(F32)]).astype(F32)
            ridx = np.concatenate([rr, rr])
            cidx = np.concatenate([p_idx[ok_h], q_idx[ok_h]])
            rhs  = np.zeros(Kk, dtype=F32)
            _append_block(vals, ridx, cidx, rhs)

        # Vertical (p: up, q: down)
        n_p = n[:-1, :, :].reshape(-1, 3)
        r_p = rays[:-1, :, :].reshape(-1, 3)
        rdy = ry.reshape(-1, 3)
        p_idx = idx_map[:-1, :].reshape(-1)
        q_idx = idx_map[ 1:, :].reshape(-1)
        ok_v = mask_v.reshape(-1)

        a = np.sum(n_p * r_p, axis=1).astype(F32)
        b = np.sum(n_p * rdy, axis=1).astype(F32)
        a = np.nan_to_num(a).astype(F32); b = np.nan_to_num(b).astype(F32)
        c_p = (b - a).astype(F32); c_q = a
        den = np.sqrt(c_p * c_p + c_q * c_q).astype(F32) + F32(1e-6)
        c_p = (c_p / den).astype(F32); c_q = (c_q / den).astype(F32)

        if np.any(ok_v):
            Kk = int(np.count_nonzero(ok_v))
            rr = np.arange(Kk, dtype=np.int32)
            ww = (lamN * np.ones(Kk, dtype=F32))
            vals = np.concatenate([(ww * c_p[ok_v]).astype(F32),
                                   (ww * c_q[ok_v]).astype(F32)]).astype(F32)
            ridx = np.concatenate([rr, rr])
            cidx = np.concatenate([p_idx[ok_v], q_idx[ok_v]])
            rhs  = np.zeros(Kk, dtype=F32)
            _append_block(vals, ridx, cidx, rhs)

    # ===== (S) 스무딩: 경계 쌍 제외 =====
    if lambda_smooth > 0:
        lamS = np.sqrt(F32(lambda_smooth)).astype(F32)

        # Horizontal
        if np.any(mask_h):
            p_idx = idx_map[:, :-1][mask_h].reshape(-1)
            q_idx = idx_map[:,  1:][mask_h].reshape(-1)
            Kk = p_idx.size
            rr = np.arange(Kk, dtype=np.int32)
            ww = lamS * np.ones(Kk, dtype=F32)
            vals = np.concatenate([ww, -ww]).astype(F32)
            ridx = np.concatenate([rr, rr])
            cidx = np.concatenate([p_idx, q_idx])
            rhs  = np.zeros(Kk, dtype=F32)
            _append_block(vals, ridx, cidx, rhs)

        # Vertical
        if np.any(mask_v):
            p_idx = idx_map[:-1, :][mask_v].reshape(-1)
            q_idx = idx_map[ 1:, :][mask_v].reshape(-1)
            Kk = p_idx.size
            rr = np.arange(Kk, dtype=np.int32)
            ww = lamS * np.ones(Kk, dtype=F32)
            vals = np.concatenate([ww, -ww]).astype(F32)
            ridx = np.concatenate([rr, rr])
            cidx = np.concatenate([p_idx, q_idx])
            rhs  = np.zeros(Kk, dtype=F32)
            _append_block(vals, ridx, cidx, rhs)

    # ===== (E,P) 경계 p만 변화: q는 값 스냅샷에 고정 =====
    if (lambda_equal > 0) or (lambda_plane > 0):
        lamE = F32(np.sqrt(F32(lambda_equal))) if lambda_equal > 0 else F32(0.0)
        lamP = F32(np.sqrt(F32(lambda_plane))) if lambda_plane > 0 else F32(0.0)
        se2  = F32(2.0) * (np.maximum(sigma_equal, F32(1e-8)) ** F32(2.0))
        sp2  = F32(2.0) * (np.maximum(sigma_plane, F32(1e-8)) ** F32(2.0))

        dirs = [
            ((slice(None), slice(1,   W)), (slice(None), slice(0,   W-1)), fused["candL_ok"]),
            ((slice(None), slice(0,   W-1)), (slice(None), slice(1,   W)), fused["candR_ok"]),
            ((slice(1,   H), slice(None)), (slice(0,   H-1), slice(None)), fused["candU_ok"]),
            ((slice(0,   H-1), slice(None)), (slice(1,   H),   slice(None)), fused["candD_ok"]),
        ]

        for p_sl, q_sl, cand_ok_full in dirs:
            # 경계 p + p↔q가 경계를 가로지르지 않는 이웃만
            pmask = pix_disc[p_sl] & cand_ok_full[p_sl]
            if not np.any(pmask):
                continue

            # 인덱스/깊이 스냅샷
            p_ids = idx_map[p_sl][pmask]
            zp    = depth_in[p_sl][pmask].astype(F32)
            zq    = depth_in[q_sl][pmask].astype(F32)        # q 스냅샷
            finite_zq = np.isfinite(zq)

            # known/hole (q 기준)
            q_known = (known_mask[q_sl][pmask]) & finite_zq
            q_hole  = (~known_mask[q_sl][pmask])

            # 기하량
            n_q = n[q_sl][pmask].astype(F32)
            r_p = rays[p_sl][pmask].astype(F32)
            r_q = rays[q_sl][pmask].astype(F32)

            # 기본 가중(|Δz| 스냅샷)
            dz = np.abs(zp - zq).astype(F32)
            wE_base = np.exp(-(dz*dz)/se2).astype(F32) if lambda_equal > 0 else None
            wP_base = np.exp(-(dz*dz)/sp2).astype(F32) if lambda_plane > 0 else None

            # plane 안정성
            if lambda_plane > 0:
                alpha = np.sum(n_q * r_p, axis=-1).astype(F32)   # (n_q·r_p)
                gamma = np.sum(n_q * r_q, axis=-1).astype(F32)   # (n_q·r_q)
                stable = (np.abs(alpha) > eps) & (np.abs(gamma) > eps)
            else:
                alpha = gamma = None
                stable = None

            # ---------- Equal: d_p ≈ z_q_snap (p만 변수) ----------
            if lambda_equal > 0:
                w_known = (lamE * gain_known * wE_base).astype(F32)
                w_hole  = (lamE * gain_hole  * wE_base).astype(F32)

                # known 쌍
                okA = q_known & (w_known > F32(0))
                if np.any(okA):
                    var_p = p_ids[okA]
                    ww    = w_known[okA]
                    rr    = np.arange(var_p.size, dtype=np.int32)
                    rhs   = (ww * zq[okA]).astype(F32)
                    _append_block(ww, rr, var_p, rhs)

                # hole 쌍(여전히 p만 변화)
                okB = q_hole & (w_hole > F32(0))
                if np.any(okB):
                    var_p = p_ids[okB]
                    ww    = w_hole[okB]
                    rr    = np.arange(var_p.size, dtype=np.int32)
                    rhs   = (ww * zq[okB]).astype(F32)  # q는 스냅샷
                    _append_block(ww, rr, var_p, rhs)

            # ---------- Plane: (n_q·r_p) d_p ≈ (n_q·r_q) z_q_snap (p만 변수) ----------
            if lambda_plane > 0:
                w_known = (lamP * gain_known * wP_base).astype(F32)
                w_hole  = (lamP * gain_hole  * wP_base).astype(F32)

                okPA = q_known & (w_known > F32(0)) & stable
                if np.any(okPA):
                    var_p = p_ids[okPA]
                    rr    = np.arange(var_p.size, dtype=np.int32)
                    ww    = (w_known[okPA] * alpha[okPA]).astype(F32)
                    rhs   = (w_known[okPA] * gamma[okPA] * zq[okPA]).astype(F32)
                    _append_block(ww, rr, var_p, rhs)

                okPB = q_hole & (w_hole > F32(0)) & stable
                if np.any(okPB):
                    var_p = p_ids[okPB]
                    rr    = np.arange(var_p.size, dtype=np.int32)
                    ww    = (w_hole[okPB] * alpha[okPB]).astype(F32)
                    rhs   = (w_hole[okPB] * gamma[okPB] * zq[okPB]).astype(F32)  # q 스냅샷
                    _append_block(ww, rr, var_p, rhs)

    # (R) Screen: 비-경계 hole만
    if lambda_screen > 0:
        lamR = F32(np.sqrt(F32(lambda_screen)))
        ids = idx_map[hole_mask & (~pix_disc)]
        if ids.size > 0:
            Kk = ids.size
            rr = np.arange(Kk, dtype=np.int32)
            ww = lamR * np.ones(Kk, dtype=F32)
            _append_block(ww, rr, ids, (ww * depth_in[hole_mask & (~pix_disc)].astype(F32)))

    # (K) Keep-known
    if lambda_keep > 0:
        lamK = F32(np.sqrt(F32(lambda_keep)))
        ids = idx_map[known_mask]
        if ids.size > 0:
            Kk = ids.size
            rr = np.arange(Kk, dtype=np.int32)
            ww = lamK * np.ones(Kk, dtype=F32)
            _append_block(ww, rr, ids, (ww * depth_in[known_mask].astype(F32)))

    # ===== 시스템 조립/해 (PyAMG only) =====
    if len(data_buf) == 0:
        out = depth_in.copy().astype(F32, copy=False)
        return out

    data = np.concatenate(data_buf).astype(np.float64, copy=False)  # solver는 64로
    rows = np.concatenate(row_buf).astype(np.int32,   copy=False)
    cols = np.concatenate(col_buf).astype(np.int32,   copy=False)
    b    = np.concatenate(b_buf).astype(np.float64,   copy=False)

    A = coo_matrix((data, (rows, cols)), shape=(rows.max()+1, N), dtype=np.float64).tocsr()

    # warm-start: 현재 depth_in을 초기값으로 사용
    x0 = depth_in.reshape(-1).astype(np.float64, copy=False)

    # PyAMG 단독 해법
    z_vec64 = solve_with_pyamg_only(
        A, b,
        tol=tol,
        maxiter=maxiter,
        x0=x0,
        cycle=pyamg_cycle  # "V" 일반적, 필요시 "W"
    )

    out = depth_in.copy().astype(F32, copy=False)
    out.reshape(-1)[:] = z_vec64.astype(F32, copy=False)
    return out
