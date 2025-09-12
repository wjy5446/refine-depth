import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import lsmr, cg


# ---------- Utilities ----------

def _make_rays(K: np.ndarray, H: int, W: int) -> np.ndarray:
    """픽셀 광선 r_p = K^{-1}[x,y,1]^T (HxWx3), L2 정규화."""
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    pix = np.stack([xx, yy, np.ones_like(xx)], axis=-1).astype(np.float32)
    Kinv = np.linalg.inv(K).astype(np.float32)
    rays = pix @ Kinv.T
    rays /= np.linalg.norm(rays, axis=2, keepdims=True).clip(1e-6, None)
    return rays.astype(np.float32)


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
    """
    단일 픽셀 불연속 맵 disc(H,W)을 만든다.
    - 기본: 깊이 차 기반
    - 옵션: 법선 급변도 결합
    """
    H, W = depth_in.shape
    # 깊이 기반 쌍 경계
    dz_h = np.abs(depth_in[:, 1:] - depth_in[:, :-1])
    dz_v = np.abs(depth_in[1:, :] - depth_in[:-1, :])
    thr_h_rel = tau_rel * np.maximum(depth_in[:, 1:], depth_in[:, :-1])
    thr_v_rel = tau_rel * np.maximum(depth_in[1:, :], depth_in[:-1, :])
    thr_h = thr_h_rel if tau_abs is None else np.maximum(thr_h_rel, tau_abs)
    thr_v = thr_v_rel if tau_abs is None else np.maximum(thr_v_rel, tau_abs)
    disc_h_d = dz_h > thr_h         # (H,W-1)
    disc_v_d = dz_v > thr_v         # (H-1,W)

    # 법선 기반 쌍 경계(옵션)
    if use_normals and (n is not None):
        nn = n.astype(np.float32)
        nn = nn / np.clip(np.linalg.norm(nn, axis=2, keepdims=True), 1e-6, None)
        cos_h = np.abs(np.sum(nn[:, :-1, :] * nn[:, 1:, :], axis=2))  # (H,W-1)
        cos_v = np.abs(np.sum(nn[:-1, :, :] * nn[1:, :, :], axis=2))  # (H-1,W)
        disc_h_n = cos_h < float(tau_n_cos)
        disc_v_n = cos_v < float(tau_n_cos)
    else:
        disc_h_n = np.zeros_like(disc_h_d)
        disc_v_n = np.zeros_like(disc_v_d)

    # 쌍 경계 결합
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

    # 픽셀 1D 경계로 승격: 어느 방향으로든 경계에 접하면 True
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
      - candL_ok..: 이웃 후보 허용(True=허용)   (경계 넘지 않기)
      - pix_disc: 픽셀 경계
    """
    H, W = disc.shape
    # 두 픽셀 중 하나라도 경계면 그 '쌍'은 경계로 간주
    pair_disc_h = disc[:, :-1] | disc[:, 1:]   # (H,W-1)
    pair_disc_v = disc[:-1, :] | disc[1:, :]   # (H-1,W)

    mask_h = ~pair_disc_h
    mask_v = ~pair_disc_v
    pix_disc = disc

    # 이웃 후보: 경계 '넘지 않기' → 해당 쌍이 경계면 False
    candL_ok = np.zeros((H, W), dtype=bool); candL_ok[:, 1:]  = ~pair_disc_h
    candR_ok = np.zeros((H, W), dtype=bool); candR_ok[:, :-1] = ~pair_disc_h
    candU_ok = np.zeros((H, W), dtype=bool); candU_ok[1:, :]  = ~pair_disc_v
    candD_ok = np.zeros((H, W), dtype=bool); candD_ok[:-1, :] = ~pair_disc_v

    return dict(mask_h=mask_h, mask_v=mask_v,
                candL_ok=candL_ok, candR_ok=candR_ok, candU_ok=candU_ok, candD_ok=candD_ok,
                pix_disc=pix_disc)


# ---------- Main (vectorized, all-pixel variables) ----------

def refine_depth_normal_alignment(
    depth_in: np.ndarray,
    known_mask: np.ndarray,
    hole_mask: np.ndarray,
    n_guide: np.ndarray | None,
    K: np.ndarray | None,
    discontinuity_maps: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    lambda_normal: float = 3.0,
    lambda_smooth: float = 0.2,
    lambda_equal: float = 1.0,
    lambda_plane: float = 1.0,
    lambda_screen: float = 1e-3,
    lambda_keep: float = 30.0,
    tol: float = 1e-4,
    maxiter: int = 200,
    solver: str = "lsmr"
) -> np.ndarray:
    """
    Discontinuity(깊이 경계) 기반 게이팅:
      - (S) 스무딩: 경계 쌍 제외
      - (D) 데이터: 모든 픽셀에 대해 4방향 중 비-경계 이웃 가운데 |Δz_ref| 최소 하나로 스칼라 앵커
      - (N) 노멀: 경계 쌍 제외
    guide_gray/edge_alpha는 사용하지 않음(시그니처만 유지).

    Args:
        depth_in: 입력 깊이 맵
        known_mask: 알려진 영역 마스크
        hole_mask: 홀 영역 마스크
        n_guide: 가이드 노멀 벡터
        K: 카메라 내부 파라미터
        discontinuity_maps: discontinuity 감지 결과 (disc_h, disc_v, discL, discR, discU, discD)
                           None이면 자동으로 감지
        lambda_normal: 노멀 정합 가중치
        lambda_smooth: 스무딩 가중치
        lambda_data: 데이터 가중치
        lambda_screen: 스크린 앵커 가중치
        lambda_n: 노멀 유사도 가중치
        tau_n: 노멀 유사도 임계값
        tol: 수렴 기준
        maxiter: 최대 반복수
        solver: 솔버 선택
        tau_discon_rel: 상대 discontinuity 임계값
        tau_discon_abs: 절대 discontinuity 임계값
    """
    H, W = depth_in.shape
    depth_in = depth_in.astype(np.float32, copy=False)

    # 변수 인덱스: 전 픽셀 변수
    idx_map = np.arange(H * W, dtype=np.int32).reshape(H, W)
    N = H * W

    # intrinsics / normals
    if K is None:
        K = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=np.float32)
    if n_guide is None:
        n = np.zeros((H, W, 3), dtype=np.float32)
        n[..., 2] = 1.0
    else:
        n = n_guide.astype(np.float32, copy=False)

    rays = _make_rays(K, H, W)

    # 노멀 정규화 + 카메라를 향하도록 플립
    n_norm = np.linalg.norm(n, axis=2, keepdims=True)
    bad = (n_norm < 1e-6)
    if bad.any():
        n[bad[..., 0]] = np.array([0, 0, 1], dtype=np.float32)
        n_norm = np.linalg.norm(n, axis=2, keepdims=True)
    n = n / np.clip(n_norm, 1e-6, None)
    dot = np.sum(n * rays, axis=2)
    n[dot > 0] *= -1.0

    # Ray differences
    rx = rays[:, 1:, :] - rays[:, :-1, :]
    ry = rays[1:, :, :] - rays[:-1, :, :]

    # ---- Discontinuity detection ----
    fused = gates_from_disc_1d(discontinuity_maps)
    mask_h, mask_v = fused["mask_h"], fused["mask_v"]
    pix_disc = fused["pix_disc"]

    # ----- 빅 COO/벡터 버퍼 -----
    data_buf = []
    row_buf = []
    col_buf = []
    b_buf = []
    row_ofs = 0

    def _append_block(vals, ridx, cidx, rhs):
        nonlocal row_ofs
        if vals.size == 0:
            return
        row_buf.append(ridx + row_ofs)
        col_buf.append(cidx)
        data_buf.append(vals.astype(np.float32, copy=False))
        b_buf.append(rhs.astype(np.float32, copy=False))
        row_ofs += rhs.size

    # ===== (N) 노멀 정합: discontinuity 쌍 제외 =====
    if lambda_normal > 0:
        lamN = np.sqrt(lambda_normal)

        # Horizontal (p: left, q: right)
        n_p = n[:, :-1, :].reshape(-1, 3)
        r_p = rays[:, :-1, :].reshape(-1, 3)
        rdx = rx.reshape(-1, 3)
        p_idx = idx_map[:, :-1].reshape(-1)
        q_idx = idx_map[:,  1:].reshape(-1)

        base_w_h = lamN * np.ones(p_idx.size, np.float32)

        ww_h = base_w_h

        ok_h = mask_h.reshape(-1)

        a = np.sum(n_p * r_p, axis=1).astype(np.float32)
        b = np.sum(n_p * rdx, axis=1).astype(np.float32)
        a = np.nan_to_num(a); b = np.nan_to_num(b)
        c_p = (b - a); c_q = a
        den = np.sqrt(c_p * c_p + c_q * c_q) + 1e-6
        c_p /= den; c_q /= den

        if np.any(ok_h):
            Kk = int(np.count_nonzero(ok_h))
            rr = np.arange(Kk, dtype=np.int32)
            vals = np.concatenate([ww_h[ok_h]*c_p[ok_h], ww_h[ok_h]*c_q[ok_h]])
            ridx = np.concatenate([rr, rr])
            cidx = np.concatenate([p_idx[ok_h], q_idx[ok_h]])
            rhs  = np.zeros(Kk, np.float32)
            _append_block(vals, ridx, cidx, rhs)

        # Vertical (p: up, q: down)
        n_p = n[:-1, :, :].reshape(-1, 3)
        r_p = rays[:-1, :, :].reshape(-1, 3)
        rdy = ry.reshape(-1, 3)
        p_idx = idx_map[:-1, :].reshape(-1)
        q_idx = idx_map[ 1:, :].reshape(-1)

        base_w_v = lamN * np.ones(p_idx.size, np.float32)

        ww_v = base_w_v

        ok_v = mask_v.reshape(-1)

        a = np.sum(n_p * r_p, axis=1).astype(np.float32)
        b = np.sum(n_p * rdy, axis=1).astype(np.float32)
        a = np.nan_to_num(a); b = np.nan_to_num(b)
        c_p = (b - a); c_q = a
        den = np.sqrt(c_p * c_p + c_q * c_q) + 1e-6
        c_p /= den; c_q /= den

        if np.any(ok_v):
            Kk = int(np.count_nonzero(ok_v))
            rr = np.arange(Kk, dtype=np.int32)
            vals = np.concatenate([ww_v[ok_v]*c_p[ok_v], ww_v[ok_v]*c_q[ok_v]])
            ridx = np.concatenate([rr, rr])
            cidx = np.concatenate([p_idx[ok_v], q_idx[ok_v]])
            rhs  = np.zeros(Kk, np.float32)
            _append_block(vals, ridx, cidx, rhs)

    # ===== (S) 스무딩: 경계 쌍 제외 (균등 가중) =====
    if lambda_smooth > 0:
        lamS = np.sqrt(lambda_smooth)

        # Horizontal
        if np.any(mask_h):
            p_idx = idx_map[:, :-1][mask_h].reshape(-1)
            q_idx = idx_map[:,  1:][mask_h].reshape(-1)
            Kk = p_idx.size
            rr = np.arange(Kk, dtype=np.int32)
            ww = lamS * np.ones(Kk, np.float32)
            vals = np.concatenate([ww, -ww])
            ridx = np.concatenate([rr, rr])
            cidx = np.concatenate([p_idx, q_idx])
            rhs  = np.zeros(Kk, np.float32)
            _append_block(vals, ridx, cidx, rhs)

        # Vertical
        if np.any(mask_v):
            p_idx = idx_map[:-1, :][mask_v].reshape(-1)
            q_idx = idx_map[ 1:, :][mask_v].reshape(-1)
            Kk = p_idx.size
            rr = np.arange(Kk, dtype=np.int32)
            ww = lamS * np.ones(Kk, np.float32)
            vals = np.concatenate([ww, -ww])
            ridx = np.concatenate([rr, rr])
            cidx = np.concatenate([p_idx, q_idx])
            rhs  = np.zeros(Kk, np.float32)
            _append_block(vals, ridx, cidx, rhs)

    ## ===== (D_weighted_multi) 경계 p → 모든 이웃 q 사용, 초기 |Δz|로 가우시안 가중 =====
    sigma_equal  = 0.02      # 가우시안 폭(깊이 단위). 작을수록 가까운 것만 강하게.
    sigma_plane  = 0.02
    tau_n_plane  = None      # 예: 0.95 쓰면 법선 유사도 필터링. None이면 미사용.

    if (lambda_equal > 0) or (lambda_plane > 0):
        lamE = np.float32(np.sqrt(lambda_equal)) if lambda_equal > 0 else np.float32(0.0)
        lamP = np.float32(np.sqrt(lambda_plane)) if lambda_plane > 0 else np.float32(0.0)
        se2  = np.float32(2.0*(max(sigma_equal, 1e-8)**2))
        sp2  = np.float32(2.0*(max(sigma_plane, 1e-8)**2))
        eps  = 1e-6

        # (p: 중심, q: 이웃) 4-이웃 슬라이스
        dirs = [
            ((slice(None), slice(1,   W)), (slice(None), slice(0,   W-1))),  # left
            ((slice(None), slice(0,   W-1)), (slice(None), slice(1,   W))),  # right
            ((slice(1,   H), slice(None)), (slice(0,   H-1), slice(None))),  # up
            ((slice(0,   H-1), slice(None)), (slice(1,   H),   slice(None))) # down
        ]

        for p_sl, q_sl in dirs:
            # 경계 픽셀만 대상
            pmask = pix_disc[p_sl]
            if not np.any(pmask):
                continue

            # 인덱스/기하량/초기깊이
            p_ids = idx_map[p_sl][pmask]
            q_ids = idx_map[q_sl][pmask]

            zp = depth_in[p_sl][pmask]      # (K,)
            zq = depth_in[q_sl][pmask]      # (K,)
            finite_q = np.isfinite(zq)

            n_p = n[p_sl][pmask]
            n_q = n[q_sl][pmask]
            r_p = rays[p_sl][pmask]
            r_q = rays[q_sl][pmask]

            # 초기 |Δz|
            dz = np.abs(zp - zq).astype(np.float32)

            # (옵션) 법선 유사도 필터
            if tau_n_plane is not None:
                cos_n = np.abs(np.sum(n_p * n_q, axis=-1)).astype(np.float32)
                normal_ok = (cos_n >= float(tau_n_plane))
            else:
                normal_ok = np.ones_like(dz, dtype=bool)

            # ---------- Equal: d_p ≈ d_q (가우시안 가중) ----------
            if lambda_equal > 0:
                # known이면 RHS 앵커, 그 외 커플링
                q_known = known_mask[q_sl][pmask] & finite_q

                # 가우시안 가중치 (가까울수록 큼)
                wE = lamE * np.exp(-(dz*dz)/se2).astype(np.float32)

                # A) q known → p 앵커: sqrt(λ) * wE * (d_p - z_q) = 0
                okA = normal_ok & q_known & (wE > 0)
                if np.any(okA):
                    var_p = p_ids[okA]
                    ww    = wE[okA]
                    rr    = np.arange(var_p.size, dtype=np.int32)
                    rhs   = ww * zq[okA].astype(np.float32)
                    _append_block(ww, rr, var_p, rhs)

                # B) q hole/경계 → p-q 커플링: sqrt(λ) * wE * (d_p - d_q) = 0
                okB = normal_ok & (~q_known) & (wE > 0) & (q_ids >= 0)  # q 인덱스 유효
                if np.any(okB):
                    var_p = p_ids[okB]
                    var_q = q_ids[okB].astype(np.int32)
                    wwP   = wE[okB]
                    wwQ   = -wE[okB]
                    Kk    = var_p.size
                    rr    = np.arange(Kk, dtype=np.int32)
                    vals  = np.concatenate([wwP, wwQ])
                    ridx  = np.concatenate([rr, rr])
                    cidx  = np.concatenate([var_p, var_q])
                    rhs   = np.zeros(Kk, np.float32)
                    _append_block(vals, ridx, cidx, rhs)

            # ---------- Plane: (n_q·r_p) d_p ≈ (n_q·r_q) d_q (가우시안 가중) ----------
            if lambda_plane > 0:
                alpha = np.sum(n_q * r_p, axis=-1).astype(np.float32)
                gamma = np.sum(n_q * r_q, axis=-1).astype(np.float32)
                stable = (np.abs(alpha) > eps) & (np.abs(gamma) > eps)

                # 가우시안 가중치
                wP = lamP * np.exp(-(dz*dz)/sp2).astype(np.float32)

                # A) q known → p 앵커: sqrt(λ) * wP * (alpha d_p - gamma z_q) = 0
                okPA = normal_ok & stable & known_mask[q_sl][pmask] & finite_q & (wP > 0)
                if np.any(okPA):
                    var_p = p_ids[okPA]
                    ww    = wP[okPA] * alpha[okPA]
                    rr    = np.arange(var_p.size, dtype=np.int32)
                    rhs   = (wP[okPA] * gamma[okPA] * zq[okPA].astype(np.float32))
                    _append_block(ww, rr, var_p, rhs)

                # B) q hole/경계 → p-q 커플링: sqrt(λ) * wP * (alpha d_p - gamma d_q) = 0
                okPB = normal_ok & stable & (~known_mask[q_sl][pmask]) & (wP > 0) & (q_ids >= 0)
                if np.any(okPB):
                    var_p = p_ids[okPB]
                    var_q = q_ids[okPB].astype(np.int32)
                    wwPp  = wP[okPB] * alpha[okPB]
                    wwPq  = -wP[okPB] * gamma[okPB]
                    Kk    = var_p.size
                    rr    = np.arange(Kk, dtype=np.int32)
                    vals  = np.concatenate([wwPp, wwPq])
                    ridx  = np.concatenate([rr,   rr  ])
                    cidx  = np.concatenate([var_p, var_q])
                    rhs   = np.zeros(Kk, np.float32)
                    _append_block(vals, ridx, cidx, rhs)

    # (Screen) 비-경계 hole만
    if lambda_screen > 0:
        lamR = np.float32(np.sqrt(lambda_screen))
        ids = idx_map[hole_mask]
        if ids.size > 0:
            Kk = ids.size
            rr = np.arange(Kk, dtype=np.int32)
            ww = lamR * np.ones(Kk, np.float32)
            _append_block(ww, rr, ids, ww * depth_in[hole_mask].astype(np.float32))

    # ===== (K) Keep-known =====
    if lambda_keep > 0:
        lamK = np.float32(np.sqrt(lambda_keep))
        ids = idx_map[known_mask]
        if ids.size > 0:
            Kk = ids.size
            rr = np.arange(Kk, dtype=np.int32)
            ww = lamK * np.ones(Kk, np.float32)
        _append_block(ww, rr, ids, ww * depth_in[known_mask].astype(np.float32))

    # ===== 시스템 조립/해 =====
    if len(data_buf) == 0:
        out = depth_in.copy()
        out[:] = depth_in
        return out

    data = np.concatenate(data_buf)
    rows = np.concatenate(row_buf)
    cols = np.concatenate(col_buf)
    b    = np.concatenate(b_buf).astype(np.float32)

    A = coo_matrix((data, (rows, cols)), shape=(rows.max()+1, N)).tocsr()

    if solver == "cg":
        AtA = (A.T @ A).tocsr()
        Atb = A.T @ b
        M = diags(1.0 / np.clip(AtA.diagonal(), 1e-6, None))
        z_vec, _ = cg(AtA, Atb, tol=tol, maxiter=maxiter, M=M)
    else:
        sol = lsmr(A, b, atol=tol, btol=tol, maxiter=maxiter)
        z_vec = sol[0]

    out = depth_in.copy()
    out.reshape(-1)[:] = z_vec.astype(np.float32)
    return out
