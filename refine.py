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
    if g.max() > 1.5:  # 0~255 → [0,1]
        g = g / 255.0
    diff_h = np.abs(g[:, 1:] - g[:, :-1]).astype(np.float32)
    diff_v = np.abs(g[1:, :] - g[:-1, :]).astype(np.float32)
    return np.exp(-edge_alpha * diff_h), np.exp(-edge_alpha * diff_v)


# ---------- Main (vectorized, all-pixel variables) ----------

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
    (S) 스무딩:    엣지 가중 사용 (이웃 차분)
    (D) 데이터:    엣지 가중 사용 (known-이웃 앵커)
    (R) 스크린:    z ≈ z_init (주로 hole 안정화)
    (K) Keep-known: mask가 아닌(known) 픽셀은 z ≈ depth_in으로 강하게 고정 (per-pixel)
    """
    H, W = depth_in.shape
    depth_init = depth_init.astype(np.float32, copy=False)
    depth_in   = depth_in.astype(np.float32,   copy=False)

    # 변수 인덱스: 전 픽셀을 변수로 사용
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

    # Ray differences, edge weights
    rx = rays[:, 1:, :] - rays[:, :-1, :]
    ry = rays[1:, :, :] - rays[:-1, :, :]
    w_e_h, w_e_v = _edge_weights_from_gray(guide_gray, edge_alpha, H, W)

    # ----- 빅 COO/벡터 버퍼 -----
    data_buf = []
    row_buf  = []
    col_buf  = []
    b_buf    = []
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

    # ===== (N) 노멀 정합: 수평/수직 (엣지 가중 X) =====
    if lambda_normal > 0:
        lamN = np.sqrt(lambda_normal)

        # Horizontal pairs (p: left, q: right)
        n_p = n[:, :-1, :].reshape(-1, 3)
        n_q = n[:,  1:, :].reshape(-1, 3)
        r_p = rays[:, :-1, :].reshape(-1, 3)
        rdx = rx.reshape(-1, 3)
        p_idx = idx_map[:, :-1].reshape(-1)
        q_idx = idx_map[:,  1:].reshape(-1)

        base_w = lamN * np.ones(p_idx.size, np.float32)
        if (lambda_n is not None) and (tau_n is not None):
            sim = np.abs(np.sum(n_p * n_q, axis=1)).astype(np.float32)
            valid_h = sim >= float(tau_n)
            ww = base_w * np.sqrt(np.exp(-float(lambda_n) * (1.0 - sim)).astype(np.float32))
        else:
            valid_h = np.ones(p_idx.size, dtype=bool)
            ww = base_w

        a = np.sum(n_p * r_p, axis=1).astype(np.float32)
        b = np.sum(n_p * rdx, axis=1).astype(np.float32)
        a = np.nan_to_num(a); b = np.nan_to_num(b)
        c_p = (b - a); c_q = a
        den = np.sqrt(c_p * c_p + c_q * c_q) + 1e-6
        c_p /= den; c_q /= den

        ok = valid_h
        if ok.any():
            Kk = int(np.count_nonzero(ok))
            rr = np.arange(Kk, dtype=np.int32)
            vals = np.concatenate([ww[ok]*c_p[ok], ww[ok]*c_q[ok]])
            ridx = np.concatenate([rr, rr])
            cidx = np.concatenate([p_idx[ok], q_idx[ok]])
            rhs  = np.zeros(Kk, np.float32)
            _append_block(vals, ridx, cidx, rhs)

        # Vertical pairs (p: up, q: down)
        n_p = n[:-1, :, :].reshape(-1, 3)
        n_q = n[ 1:, :, :].reshape(-1, 3)
        r_p = rays[:-1, :, :].reshape(-1, 3)
        rdy = ry.reshape(-1, 3)
        p_idx = idx_map[:-1, :].reshape(-1)
        q_idx = idx_map[ 1:, :].reshape(-1)

        base_w = lamN * np.ones(p_idx.size, np.float32)
        if (lambda_n is not None) and (tau_n is not None):
            sim = np.abs(np.sum(n_p * n_q, axis=1)).astype(np.float32)
            valid_v = sim >= float(tau_n)
            ww = base_w * np.sqrt(np.exp(-float(lambda_n) * (1.0 - sim)).astype(np.float32))
        else:
            valid_v = np.ones(p_idx.size, dtype=bool)
            ww = base_w

        a = np.sum(n_p * r_p, axis=1).astype(np.float32)
        b = np.sum(n_p * rdy, axis=1).astype(np.float32)
        a = np.nan_to_num(a); b = np.nan_to_num(b)
        c_p = (b - a); c_q = a
        den = np.sqrt(c_p * c_p + c_q * c_q) + 1e-6
        c_p /= den; c_q /= den

        ok = valid_v
        if ok.any():
            Kk = int(np.count_nonzero(ok))
            rr = np.arange(Kk, dtype=np.int32)
            vals = np.concatenate([ww[ok]*c_p[ok], ww[ok]*c_q[ok]])
            ridx = np.concatenate([rr, rr])
            cidx = np.concatenate([p_idx[ok], q_idx[ok]])
            rhs  = np.zeros(Kk, np.float32)
            _append_block(vals, ridx, cidx, rhs)

    # ===== (S) 스무딩: 전 픽셀 이웃, 엣지 가중 =====
    if lambda_smooth > 0:
        lamS = np.sqrt(lambda_smooth)

        # Horizontal all-variable pairs
        ww = lamS * np.sqrt(w_e_h.astype(np.float32).reshape(-1))
        p_idx = idx_map[:, :-1].reshape(-1)
        q_idx = idx_map[:,  1:].reshape(-1)
        Kk = p_idx.size
        if Kk > 0:
            rr = np.arange(Kk, dtype=np.int32)
            vals = np.concatenate([ww, -ww])
            ridx = np.concatenate([rr, rr])
            cidx = np.concatenate([p_idx, q_idx])
            rhs  = np.zeros(Kk, np.float32)
            _append_block(vals, ridx, cidx, rhs)

        # Vertical all-variable pairs
        ww = lamS * np.sqrt(w_e_v.astype(np.float32).reshape(-1))
        p_idx = idx_map[:-1, :].reshape(-1)
        q_idx = idx_map[ 1:, :].reshape(-1)
        Kk = p_idx.size
        if Kk > 0:
            rr = np.arange(Kk, dtype=np.int32)
            vals = np.concatenate([ww, -ww])
            ridx = np.concatenate([rr, rr])
            cidx = np.concatenate([p_idx, q_idx])
            rhs  = np.zeros(Kk, np.float32)
            _append_block(vals, ridx, cidx, rhs)

    # ===== (D) 데이터 앵커: known-이웃 경계 (엣지 가중) =====
    if lambda_data > 0:
        lamD = np.sqrt(lambda_data)

        # Horizontal: p known, q variable → q를 z_a로 끌어줌
        mask = known_mask[:, :-1] & (~known_mask[:, 1:])
        if mask.any():
            q_idx = idx_map[:, 1:][mask]
            z_a   = depth_in[:, :-1][mask]
            ww    = lamD * np.sqrt(w_e_h[mask].astype(np.float32))
            Kk = q_idx.size
            rr = np.arange(Kk, dtype=np.int32)
            _append_block(ww, rr, q_idx, ww * z_a)

        mask = (~known_mask[:, :-1]) & known_mask[:, 1:]
        if mask.any():
            p_idx = idx_map[:, :-1][mask]
            z_a   = depth_in[:, 1:][mask]
            ww    = lamD * np.sqrt(w_e_h[mask].astype(np.float32))
            Kk = p_idx.size
            rr = np.arange(Kk, dtype=np.int32)
            _append_block(ww, rr, p_idx, ww * z_a)

        # Vertical
        mask = known_mask[:-1, :] & (~known_mask[1:, :])
        if mask.any():
            q_idx = idx_map[1:, :][mask]
            z_a   = depth_in[:-1, :][mask]
            ww    = lamD * np.sqrt(w_e_v[mask].astype(np.float32))
            Kk = q_idx.size
            rr = np.arange(Kk, dtype=np.int32)
            _append_block(ww, rr, q_idx, ww * z_a)

        mask = (~known_mask[:-1, :]) & known_mask[1:, :]
        if mask.any():
            p_idx = idx_map[:-1, :][mask]
            z_a   = depth_in[1:, :][mask]
            ww    = lamD * np.sqrt(w_e_v[mask].astype(np.float32))
            Kk = p_idx.size
            rr = np.arange(Kk, dtype=np.int32)
            _append_block(ww, rr, p_idx, ww * z_a)

    # ===== (R) 스크린 앵커: hole에만 적용(기존 Stage-1 유지 유도) =====
    if lambda_screen > 0:
        lamR = np.sqrt(lambda_screen)
        ids = idx_map[hole_mask]
        if ids.size > 0:
            Kk = ids.size
            rr = np.arange(Kk, dtype=np.int32)
            ww = lamR * np.ones(Kk, np.float32)
            _append_block(ww, rr, ids, ww * depth_init[hole_mask])

    # ===== (K) Keep-known: known 픽셀은 z ≈ depth_in으로 강하게 =====
    # 내부 상수로 강도 설정(원하면 lambda_data 배수로 튜닝)
    keep_scale = 100.0  # 필요시 5~50 범위에서 조절
    ids = idx_map[known_mask]
    if ids.size > 0:
        Kk = ids.size
        rr = np.arange(Kk, dtype=np.int32)
        ww = np.sqrt(lambda_data * keep_scale) * np.ones(Kk, np.float32)
        _append_block(ww, rr, ids, ww * depth_in[known_mask])

    # ===== 시스템 조립 =====
    if len(data_buf) == 0:
        out = depth_init.copy()
        out[:] = depth_init
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
    out.reshape(-1)[:] = z_vec.astype(np.float32)
    return out
