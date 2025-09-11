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


# ---------- Main (vectorized, all-pixel variables) ----------

def refine_depth_normal_alignment(
    depth_init: np.ndarray,
    depth_in: np.ndarray,
    known_mask: np.ndarray,
    hole_mask: np.ndarray,
    guide_gray: np.ndarray | None,  # 시그니처 유지 (미사용)
    n_guide: np.ndarray | None,
    K: np.ndarray | None,
    lambda_normal: float = 3.0,
    lambda_smooth: float = 0.2,
    lambda_data: float = 1.0,
    lambda_screen: float = 1e-3,
    lambda_n: float | None = 0.5,
    tau_n: float | None = 0.95,
    edge_alpha: float = 6.0,        # 시그니처 유지 (미사용)
    tol: float = 1e-4,
    maxiter: int = 200,
    solver: str = "lsmr",
) -> np.ndarray:
    """
    Discontinuity(깊이 경계) 기반 게이팅:
      - (S) 스무딩: 경계 쌍 제외
      - (D) 데이터: 모든 픽셀에 대해 4방향 중 비-경계 이웃 가운데 |Δz_ref| 최소 하나로 스칼라 앵커
      - (N) 노멀: 경계 쌍 제외
    guide_gray/edge_alpha는 사용하지 않음(시그니처만 유지).
    """
    H, W = depth_in.shape
    depth_init = depth_init.astype(np.float32, copy=False)
    depth_in   = depth_in.astype(np.float32,   copy=False)

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

    # ---- Discontinuity detection (depth-based) ----
    # 기준 깊이: known → depth_in, else → depth_init
    z_ref = np.where(known_mask, depth_in, depth_init).astype(np.float32)

    dz_h = np.abs(z_ref[:, 1:] - z_ref[:, :-1])   # (H, W-1)
    dz_v = np.abs(z_ref[1:, :] - z_ref[:-1, :])   # (H-1, W)

    # 상대/절대 임계
    tau_discon_rel = 0.05
    tau_discon_abs = None  # 예: 0.02 등 절대 임계. None이면 미사용.

    thr_h_rel = tau_discon_rel * np.maximum(z_ref[:, 1:], z_ref[:, :-1])
    thr_v_rel = tau_discon_rel * np.maximum(z_ref[1:, :], z_ref[:-1, :])
    thr_h = thr_h_rel if tau_discon_abs is None else np.maximum(thr_h_rel, tau_discon_abs)
    thr_v = thr_v_rel if tau_discon_abs is None else np.maximum(thr_v_rel, tau_discon_abs)

    disc_h = dz_h > thr_h            # (H, W-1)  between (i,j) and (i,j+1)
    disc_v = dz_v > thr_v            # (H-1, W)  between (i,j) and (i+1,j)

    # ---- Align discontinuities to current-pixel viewpoints (L/R/U/D), all (H,W) ----
    discL = np.zeros((H, W), dtype=bool); discL[:, 1:]  = disc_h
    discR = np.zeros((H, W), dtype=bool); discR[:, :-1] = disc_h
    discU = np.zeros((H, W), dtype=bool); discU[1:, :]  = disc_v
    discD = np.zeros((H, W), dtype=bool); discD[:-1, :] = disc_v

    # ----- 빅 COO/벡터 버퍼 -----
    data_buf: list[np.ndarray] = []
    row_buf:  list[np.ndarray] = []
    col_buf:  list[np.ndarray] = []
    b_buf:    list[np.ndarray] = []
    row_ofs  = 0

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
        if (lambda_n is not None) and (tau_n is not None):
            n_q = n[:, 1:, :].reshape(-1, 3)
            sim = np.abs(np.sum(n_p * n_q, axis=1)).astype(np.float32)
            valid_h = sim >= float(tau_n)
            ww_h = base_w_h * np.sqrt(np.exp(-float(lambda_n) * (1.0 - sim)).astype(np.float32))
        else:
            valid_h = np.ones(p_idx.size, dtype=bool)
            ww_h = base_w_h

        ok_h = (~disc_h).reshape(-1) & valid_h

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
        if (lambda_n is not None) and (tau_n is not None):
            n_q = n[1:, :, :].reshape(-1, 3)
            sim = np.abs(np.sum(n_p * n_q, axis=1)).astype(np.float32)
            valid_v = sim >= float(tau_n)
            ww_v = base_w_v * np.sqrt(np.exp(-float(lambda_n) * (1.0 - sim)).astype(np.float32))
        else:
            valid_v = np.ones(p_idx.size, dtype=bool)
            ww_v = base_w_v

        ok_v = (~disc_v).reshape(-1) & valid_v

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
        mask_h = ~disc_h
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
        mask_v = ~disc_v
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

         # ===== (D) 데이터 앵커: 경계 픽셀 & hole 픽셀에만 적용 (one-hot, L→R→U→D) =====
    if lambda_data > 0:
        lamD = np.sqrt(lambda_data)

        # 픽셀 단위 "경계 여부": 4방 중 하나라도 경계면 True
        pix_disc = discL | discR | discU | discD

        # 비-경계 이웃만 후보로 허용 (경계 넘지 않기)
        candL_ok = np.zeros((H, W), dtype=bool); candL_ok[:, 1:]  = ~discL[:, 1:]
        candR_ok = np.zeros((H, W), dtype=bool); candR_ok[:, :-1] = ~discR[:, :-1]
        candU_ok = np.zeros((H, W), dtype=bool); candU_ok[1:, :]  = ~discU[1:, :]
        candD_ok = np.zeros((H, W), dtype=bool); candD_ok[:-1, :] = ~discD[:-1, :]

        # |Δz_ref| (불가능 후보는 inf)
        diffL = np.full((H, W), np.inf, np.float32); diffL[:, 1:]  = np.abs(z_ref[:, 1:] - z_ref[:, :-1]); diffL[~candL_ok] = np.inf
        diffR = np.full((H, W), np.inf, np.float32); diffR[:, :-1] = np.abs(z_ref[:, :-1] - z_ref[:, 1:]); diffR[~candR_ok] = np.inf
        diffU = np.full((H, W), np.inf, np.float32); diffU[1:, :]  = np.abs(z_ref[1:, :] - z_ref[:-1, :]); diffU[~candU_ok] = np.inf
        diffD = np.full((H, W), np.inf, np.float32); diffD[:-1, :] = np.abs(z_ref[:-1, :] - z_ref[1:, :]); diffD[~candD_ok] = np.inf

        # 최소 후보와 one-hot 선택 (L→R→U→D 우선)
        diffs = np.stack([diffL, diffR, diffU, diffD], axis=-1)  # HxWx4
        min_diff = np.min(diffs, axis=-1)                        # HxW
        allow = np.isfinite(min_diff) & hole_mask & pix_disc     # 경계&hole인 픽셀만 허용

        onehot = np.zeros((H, W, 4), dtype=bool)
        for d in (diffL, diffR, diffU, diffD):
            sel = (d == min_diff) & allow & (~onehot.any(axis=-1))
            # 해당 방향 인덱스에 True 세팅
            if d is diffL:   onehot[..., 0] = sel
            elif d is diffR: onehot[..., 1] = sel
            elif d is diffU: onehot[..., 2] = sel
            else:            onehot[..., 3] = sel

        # 선택된 이웃의 인덱스 맵
        idxL = np.full((H, W), -1, np.int32); idxL[:, 1:]  = idx_map[:, :-1]
        idxR = np.full((H, W), -1, np.int32); idxR[:, :-1] = idx_map[:, 1:]
        idxU = np.full((H, W), -1, np.int32); idxU[1:, :]  = idx_map[:-1, :]
        idxD = np.full((H, W), -1, np.int32); idxD[:-1, :] = idx_map[1:, :]

        nei_idx = (
            idxL * onehot[..., 0] +
            idxR * onehot[..., 1] +
            idxU * onehot[..., 2] +
            idxD * onehot[..., 3]
        )

        valid_anchor = onehot.any(axis=-1)  # 경계&hole에서 실제 하나가 선택된 픽셀
        if np.any(valid_anchor):
            var_ids = idx_map[valid_anchor]
            nei_ids = nei_idx[valid_anchor]
            z_a     = z_ref.reshape(-1)[nei_ids]

            Kk = var_ids.size
            rr = np.arange(Kk, dtype=np.int32)
            ww = lamD * np.ones(Kk, np.float32)
            _append_block(ww, rr, var_ids, ww * z_a)


    # ===== (R) 스크린 앵커 =====
    if lambda_screen > 0:
        lamR = np.sqrt(lambda_screen)
        ids = idx_map[hole_mask]
        if ids.size > 0:
            Kk = ids.size
            rr = np.arange(Kk, dtype=np.int32)
            ww = lamR * np.ones(Kk, np.float32)
            _append_block(ww, rr, ids, ww * depth_init[hole_mask])

    # ===== (K) Keep-known =====
    keep_scale = 30.0
    ids = idx_map[known_mask]
    if ids.size > 0:
        Kk = ids.size
        rr = np.arange(Kk, dtype=np.int32)
        ww = np.sqrt(keep_scale) * np.ones(Kk, np.float32)
        _append_block(ww, rr, ids, ww * depth_in[known_mask])

    # ===== 시스템 조립/해 =====
    if len(data_buf) == 0:
        out = depth_init.copy()
        out[:] = depth_init
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

    out = depth_init.copy()
    out.reshape(-1)[:] = z_vec.astype(np.float32)
    return out
