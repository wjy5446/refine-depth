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


def _edge_weights_from_gray(guide_gray: np.ndarray | None, edge_alpha: float, H: int, W: int):
    """엣지 가중치 w_edge = exp(-alpha * |ΔI|). guide_gray=None이면 1 반환."""
    if guide_gray is None:
        one_h = np.ones((H, W-1), np.float32)
        one_v = np.ones((H-1, W), np.float32)
        return one_h, one_v
    g = guide_gray.astype(np.float32, copy=False)
    if g.max() > 1.5:  # 0~255로 들어오는 경우 [0,1] 정규화
        g = g / 255.0
    diff_h = np.abs(g[:, 1:] - g[:, :-1]).astype(np.float32)
    diff_v = np.abs(g[1:, :] - g[:-1, :]).astype(np.float32)
    return np.exp(-edge_alpha * diff_h), np.exp(-edge_alpha * diff_v)


# ---------- Main (vectorized) ----------

def refine_depth_normal_alignment(
    depth_init: np.ndarray,
    depth_in: np.ndarray,
    known_mask: np.ndarray,
    hole_mask: np.ndarray,
    guide_gray: np.ndarray | None,
    n_guide: np.ndarray | None,
    K: np.ndarray | None,
    lambda_normal: float = 3.0,
    lambda_smooth: float = 0.2,
    lambda_data: float = 1.0,
    lambda_screen: float = 1e-3,
    lambda_n: float | None = 0.5,
    tau_n: float | None = 0.95,
    edge_alpha: float = 6.0,
    tol: float = 1e-4,
    maxiter: int = 200,
    solver: str = "lsmr",
) -> np.ndarray:
    """
    (N) 노멀 정합: 엣지 가중 미사용
    (S) 스무딩:    엣지 가중 사용
    (D) 데이터:    엣지 가중 사용
    (R) 스크린:    z ≈ z_init
    """
    H, W = depth_in.shape
    depth_init = depth_init.astype(np.float32, copy=False)
    depth_in   = depth_in.astype(np.float32,   copy=False)

    # 변수 인덱스
    idx_map = -np.ones((H, W), dtype=np.int32)
    idx_map[hole_mask] = np.arange(int(hole_mask.sum()), dtype=np.int32)
    N = int(hole_mask.sum())
    if N == 0:
        return depth_in.copy()

    # intrinsics / normals
    if K is None:
        K = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=np.float32)
    if n_guide is None:
        n = np.zeros((H, W, 3), dtype=np.float32)
        n[..., 2] = 1.0
    else:
        n = n_guide.astype(np.float32, copy=False)

    rays = _make_rays(K, H, W)
    # 노멀 방향 정규화 + 카메라를 향하도록 플립
    n_norm = np.linalg.norm(n, axis=2, keepdims=True)
    bad = (n_norm < 1e-6)
    if bad.any():
        n[bad[..., 0]] = np.array([0, 0, 1], dtype=np.float32)
        n_norm = np.linalg.norm(n, axis=2, keepdims=True)
    n = n / np.clip(n_norm, 1e-6, None)
    dot = np.sum(n * rays, axis=2)
    n[dot > 0] *= -1.0

    rx = rays[:, 1:, :] - rays[:, :-1, :]
    ry = rays[1:, :, :] - rays[:-1, :, :]
    w_e_h, w_e_v = _edge_weights_from_gray(guide_gray, edge_alpha, H, W)

    # ----- 빅 COO/벡터 버퍼 -----
    data_buf = []
    row_buf  = []
    col_buf  = []
    b_buf    = []
    row_ofs  = 0  # 누적 행 오프셋

    def _append_block(vals, ridx, cidx, rhs):
        nonlocal row_ofs
        if vals.size == 0:
            return
        # ridx는 [0..k-1] local 인덱스라고 가정 → 누적 오프셋 더해 전역화
        row_buf.append(ridx + row_ofs)
        col_buf.append(cidx)
        data_buf.append(vals)
        b_buf.append(rhs.astype(np.float32, copy=False))
        row_ofs += rhs.size

    # ===== (N) 노멀 정합: 수평/수직 (엣지 가중 X) =====
    if lambda_normal > 0:
        lamN = np.sqrt(lambda_normal)

        # --- Horizontal ---
        mask_any = (hole_mask[:, :-1] | hole_mask[:, 1:])
        if mask_any.any():
            n_p = n[:, :-1, :][mask_any]
            n_q = n[:,  1:, :][mask_any]
            r_p = rays[:, :-1, :][mask_any]
            rdx = rx[mask_any]
            p_idx = idx_map[:, :-1][mask_any]
            q_idx = idx_map[:,  1:][mask_any]
            z_known_q = depth_in[:, 1:][mask_any]
            z_known_p = depth_in[:, :-1][mask_any]

            base_w = lamN * np.ones(rdx.shape[0], np.float32)
            if (lambda_n is not None) and (tau_n is not None):
                sim = np.abs(np.sum(n_p * n_q, axis=1)).astype(np.float32)
                valid = sim >= float(tau_n)
                ww = base_w * np.sqrt(np.exp(-float(lambda_n) * (1.0 - sim)).astype(np.float32))
            else:
                valid = np.ones(rdx.shape[0], dtype=bool)
                ww = base_w

            a = np.sum(n_p * r_p, axis=1).astype(np.float32)
            b = np.sum(n_p * rdx, axis=1).astype(np.float32)
            a = np.nan_to_num(a); b = np.nan_to_num(b)
            c_p = (b - a); c_q = a
            den = np.sqrt(c_p * c_p + c_q * c_q) + 1e-6
            c_p /= den; c_q /= den

            # case 분기를 한 번에 벡터화
            # hole-hole
            ok_hh = valid & (p_idx >= 0) & (q_idx >= 0)
            K = int(np.count_nonzero(ok_hh))
            if K > 0:
                rr = np.arange(K, dtype=np.int32)
                vals = np.concatenate([ww[ok_hh]*c_p[ok_hh], ww[ok_hh]*c_q[ok_hh]])
                ridx = np.concatenate([rr, rr])
                cidx = np.concatenate([p_idx[ok_hh], q_idx[ok_hh]])
                rhs  = np.zeros(K, np.float32)
                _append_block(vals, ridx, cidx, rhs)

            # hole-known (q known)
            ok_hk = valid & (p_idx >= 0) & (q_idx < 0)
            K = int(np.count_nonzero(ok_hk))
            if K > 0:
                rr = np.arange(K, dtype=np.int32)
                vals = ww[ok_hk]*c_p[ok_hk]
                ridx = rr
                cidx = p_idx[ok_hk]
                rhs  = -ww[ok_hk]*c_q[ok_hk]*z_known_q[ok_hk]
                _append_block(vals, ridx, cidx, rhs)

            # known-hole (p known)
            ok_kh = valid & (p_idx < 0) & (q_idx >= 0)
            K = int(np.count_nonzero(ok_kh))
            if K > 0:
                rr = np.arange(K, dtype=np.int32)
                vals = ww[ok_kh]*c_q[ok_kh]
                ridx = rr
                cidx = q_idx[ok_kh]
                rhs  = -ww[ok_kh]*c_p[ok_kh]*z_known_p[ok_kh]
                _append_block(vals, ridx, cidx, rhs)

        # --- Vertical ---
        mask_any = (hole_mask[:-1, :] | hole_mask[1:, :])
        if mask_any.any():
            n_p = n[:-1, :, :][mask_any]
            n_q = n[ 1:, :, :][mask_any]
            r_p = rays[:-1, :, :][mask_any]
            rdy = ry[mask_any]
            p_idx = idx_map[:-1, :][mask_any]
            q_idx = idx_map[ 1:, :][mask_any]
            z_known_q = depth_in[1:, :][mask_any]
            z_known_p = depth_in[:-1, :][mask_any]

            base_w = lamN * np.ones(rdy.shape[0], np.float32)
            if (lambda_n is not None) and (tau_n is not None):
                sim = np.abs(np.sum(n_p * n_q, axis=1)).astype(np.float32)
                valid = sim >= float(tau_n)
                ww = base_w * np.sqrt(np.exp(-float(lambda_n) * (1.0 - sim)).astype(np.float32))
            else:
                valid = np.ones(rdy.shape[0], dtype=bool)
                ww = base_w

            a = np.sum(n_p * r_p, axis=1).astype(np.float32)
            b = np.sum(n_p * rdy, axis=1).astype(np.float32)
            a = np.nan_to_num(a); b = np.nan_to_num(b)
            c_p = (b - a); c_q = a
            den = np.sqrt(c_p * c_p + c_q * c_q) + 1e-6
            c_p /= den; c_q /= den

            ok_hh = valid & (p_idx >= 0) & (q_idx >= 0)
            K = int(np.count_nonzero(ok_hh))
            if K > 0:
                rr = np.arange(K, dtype=np.int32)
                vals = np.concatenate([ww[ok_hh]*c_p[ok_hh], ww[ok_hh]*c_q[ok_hh]])
                ridx = np.concatenate([rr, rr])
                cidx = np.concatenate([p_idx[ok_hh], q_idx[ok_hh]])
                rhs  = np.zeros(K, np.float32)
                _append_block(vals, ridx, cidx, rhs)

            ok_hk = valid & (p_idx >= 0) & (q_idx < 0)
            K = int(np.count_nonzero(ok_hk))
            if K > 0:
                rr = np.arange(K, dtype=np.int32)
                vals = ww[ok_hk]*c_p[ok_hk]
                ridx = rr
                cidx = p_idx[ok_hk]
                rhs  = -ww[ok_hk]*c_q[ok_hk]*z_known_q[ok_hk]
                _append_block(vals, ridx, cidx, rhs)

            ok_kh = valid & (p_idx < 0) & (q_idx >= 0)
            K = int(np.count_nonzero(ok_kh))
            if K > 0:
                rr = np.arange(K, dtype=np.int32)
                vals = ww[ok_kh]*c_q[ok_kh]
                ridx = rr
                cidx = q_idx[ok_kh]
                rhs  = -ww[ok_kh]*c_p[ok_kh]*z_known_p[ok_kh]
                _append_block(vals, ridx, cidx, rhs)

    # ===== (S) 스무딩: 엣지 가중 사용 =====
    if lambda_smooth > 0:
        lamS = np.sqrt(lambda_smooth)

        # Horizontal hole-hole
        mask = hole_mask[:, :-1] & hole_mask[:, 1:]
        if mask.any():
            ww = lamS * np.sqrt(w_e_h[mask].astype(np.float32))
            p_idx = idx_map[:, :-1][mask]
            q_idx = idx_map[:,  1:][mask]
            K = p_idx.size
            rr = np.arange(K, dtype=np.int32)
            vals = np.concatenate([ww, -ww])
            ridx = np.concatenate([rr, rr])
            cidx = np.concatenate([p_idx, q_idx])
            rhs  = np.zeros(K, np.float32)
            _append_block(vals, ridx, cidx, rhs)

        # Vertical hole-hole
        mask = hole_mask[:-1, :] & hole_mask[1:, :]
        if mask.any():
            ww = lamS * np.sqrt(w_e_v[mask].astype(np.float32))
            p_idx = idx_map[:-1, :][mask]
            q_idx = idx_map[ 1:, :][mask]
            K = p_idx.size
            rr = np.arange(K, dtype=np.int32)
            vals = np.concatenate([ww, -ww])
            ridx = np.concatenate([rr, rr])
            cidx = np.concatenate([p_idx, q_idx])
            rhs  = np.zeros(K, np.float32)
            _append_block(vals, ridx, cidx, rhs)

    # ===== (D) 데이터 앵커: 엣지 가중 사용 =====
    if lambda_data > 0:
        lamD = np.sqrt(lambda_data)

        # Horizontal 양방향
        mask = hole_mask[:, :-1] & known_mask[:, 1:]
        if mask.any():
            p_idx = idx_map[:, :-1][mask]
            z_a   = depth_in[:, 1:][mask]
            ww    = lamD * np.sqrt(w_e_h[mask].astype(np.float32))
            K = p_idx.size
            rr = np.arange(K, dtype=np.int32)
            vals = ww
            ridx = rr
            cidx = p_idx
            rhs  = ww * z_a
            _append_block(vals, ridx, cidx, rhs)

        mask = known_mask[:, :-1] & hole_mask[:, 1:]
        if mask.any():
            q_idx = idx_map[:, 1:][mask]
            z_a   = depth_in[:, :-1][mask]
            ww    = lamD * np.sqrt(w_e_h[mask].astype(np.float32))
            K = q_idx.size
            rr = np.arange(K, dtype=np.int32)
            vals = ww
            ridx = rr
            cidx = q_idx
            rhs  = ww * z_a
            _append_block(vals, ridx, cidx, rhs)

        # Vertical 양방향
        mask = hole_mask[:-1, :] & known_mask[1:, :]
        if mask.any():
            p_idx = idx_map[:-1, :][mask]
            z_a   = depth_in[1:, :][mask]
            ww    = lamD * np.sqrt(w_e_v[mask].astype(np.float32))
            K = p_idx.size
            rr = np.arange(K, dtype=np.int32)
            vals = ww
            ridx = rr
            cidx = p_idx
            rhs  = ww * z_a
            _append_block(vals, ridx, cidx, rhs)

        mask = known_mask[:-1, :] & hole_mask[1:, :]
        if mask.any():
            q_idx = idx_map[1:, :][mask]
            z_a   = depth_in[:-1, :][mask]
            ww    = lamD * np.sqrt(w_e_v[mask].astype(np.float32))
            K = q_idx.size
            rr = np.arange(K, dtype=np.int32)
            vals = ww
            ridx = rr
            cidx = q_idx
            rhs  = ww * z_a
            _append_block(vals, ridx, cidx, rhs)

    # ===== (R) 스크린 앵커: 모든 hole =====
    if lambda_screen > 0:
        lamR = np.sqrt(lambda_screen)
        ww = lamR * np.ones(N, np.float32)
        rr = np.arange(N, dtype=np.int32)
        vals = ww
        ridx = rr
        cidx = np.arange(N, dtype=np.int32)
        rhs  = ww * depth_init[hole_mask]
        _append_block(vals, ridx, cidx, rhs)

    # ===== 시스템 조립 =====
    if len(data_buf) == 0:
        out = depth_init.copy()
        out[hole_mask] = depth_init[hole_mask]
        return out

    data = np.concatenate(data_buf)
    rows = np.concatenate(row_buf)
    cols = np.concatenate(col_buf)
    b    = np.concatenate(b_buf).astype(np.float32)

    A = coo_matrix((data, (rows, cols)), shape=(rows.max()+1, N)).tocsr()

    # ===== 선형해 =====
    if solver == "cg":
        AtA = (A.T @ A).tocsr()
        Atb = A.T @ b
        M = diags(1.0 / np.clip(AtA.diagonal(), 1e-6, None))
        z_vec, _ = cg(AtA, Atb, tol=tol, maxiter=maxiter, M=M)
    else:
        sol = lsmr(A, b, atol=tol, btol=tol, maxiter=maxiter)
        z_vec = sol[0]

    out = depth_init.copy()
    out[hole_mask] = z_vec.astype(np.float32)
    return out
