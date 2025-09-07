import numpy as np
from scipy.sparse import coo_matrix, vstack, diags
from scipy.sparse.linalg import lsmr, cg


# ---------- Utilities ----------

def _make_rays(K: np.ndarray, H: int, W: int) -> np.ndarray:
    """픽셀 광선 r_p = K^{-1} [x,y,1]^T (HxWx3), L2 정규화."""
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    pix = np.stack([xx, yy, np.ones_like(xx)], axis=-1)  # HxWx3
    Kinv = np.linalg.inv(K).astype(np.float32)
    rays = pix @ Kinv.T
    rays /= np.linalg.norm(rays, axis=2, keepdims=True).clip(1e-6, None)
    return rays.astype(np.float32)


def _edge_weights_from_gray(guide_gray: np.ndarray | None, edge_alpha: float, H: int, W: int):
    """
    엣지 가중치 w_edge = exp(-alpha * |ΔI|).
    guide_gray가 None이면 정확한 shape의 1을 반환.
    """
    if guide_gray is None:
        return np.ones((H, W-1), np.float32), np.ones((H-1, W), np.float32)
    diff_h = np.abs(guide_gray[:, 1:] - guide_gray[:, :-1]).astype(np.float32)
    diff_v = np.abs(guide_gray[1:, :] - guide_gray[:-1, :]).astype(np.float32)
    w_e_h = np.exp(-edge_alpha * diff_h).astype(np.float32)  # H x (W-1)
    w_e_v = np.exp(-edge_alpha * diff_v).astype(np.float32)  # (H-1) x W
    return w_e_h, w_e_v


def _push_row(rows_list, vals, row_idx, col_idx, nrow, N):
    """COO 희소 행을 rows_list에 누적."""
    if nrow <= 0:
        return
    rows_list.append(coo_matrix((vals, (row_idx, col_idx)), shape=(nrow, N)))


# ---------- Main ----------

def refine_depth_normal_alignment(
    depth_init: np.ndarray,            # 초기 깊이 (Stage-1)
    depth_in: np.ndarray,              # 원본 깊이 (경계 anchor)
    known_mask: np.ndarray,            # True = known
    hole_mask: np.ndarray,             # True = variable (미지수)
    guide_gray: np.ndarray | None,     # 엣지 가이드 (스무딩·가중)
    n_guide: np.ndarray | None,        # HxWx3 단위 법선
    K: np.ndarray | None,              # intrinsics (3x3)
    # Weights
    lambda_normal: float = 3.0,        # (N) 법선 정합 강도
    lambda_smooth: float = 0.2,        # (S) 스무딩
    lambda_data: float = 1.0,          # (D) 경계 데이터  ※ 엣지 가중 적용
    lambda_screen: float = 1e-3,       # (R) 스크린 앵커
    # Normal similarity (optional)
    lambda_n: float | None = 0.5,      # None이면 미사용
    tau_n: float | None = 0.95,        # None이면 미사용
    # Solver params
    edge_alpha: float = 6.0,
    tol: float = 1e-4,
    maxiter: int = 200,
    solver: str = "lsmr",              # "lsmr" | "cg"
) -> np.ndarray:
    """
    법선 정합(N): n^T ∂X/∂x = 0, n^T ∂X/∂y = 0 을 선형 LS로 최소화.
    X_p = z_p r_p,  ∂X/∂x ≈ r_x z_p + r_p (z_q - z_p)  (q: 우/하 이웃)
    """
    H, W = depth_in.shape
    assert depth_init.shape == (H, W)
    assert known_mask.shape == (H, W) and hole_mask.shape == (H, W)

    # 변수 인덱스
    idx_map = -np.ones((H, W), dtype=np.int32)
    idx_map[hole_mask] = np.arange(int(hole_mask.sum()), dtype=np.int32)
    N = int(hole_mask.sum())
    if N == 0:
        return depth_in.astype(np.float32)

    # 기본 intrinsics / normals
    if K is None:
        K = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=np.float32)
    if n_guide is None:
        n_guide = np.zeros((H, W, 3), dtype=np.float32)
        n_guide[:, :, 2] = 1.0

    # Rays
    rays = _make_rays(K, H, W)                      # HxWx3

    # Normals + orientation fix (카메라를 향하도록 통일: n·r <= 0)
    n = n_guide.astype(np.float32)
    n /= np.linalg.norm(n, axis=2, keepdims=True).clip(1e-6, None)
    dot = np.sum(n * rays, axis=2)
    n[dot > 0] *= -1.0
    n /= np.linalg.norm(n, axis=2, keepdims=True).clip(1e-6, None)

    # Edge weights(스무딩/데이터용)
    w_e_h, w_e_v = _edge_weights_from_gray(guide_gray, edge_alpha, H, W)

    rows = []
    rhs_all = []

    # ---- Helper: 노멀 유사 가중/마스크 ----
    def pair_weight_and_mask(nL, nR, base_w):
        if (lambda_n is None) or (tau_n is None):
            return base_w, np.ones(base_w.shape, bool)
        sim = np.abs(np.sum(nL * nR, axis=1)).astype(np.float32)  # |cos|
        valid = (sim >= tau_n)
        w_n = np.exp(-float(lambda_n) * (1.0 - sim)).astype(np.float32)
        return base_w * np.sqrt(w_n), valid

    # ===== (N) Normal-Alignment =====
    if lambda_normal > 0:
        lamN = np.sqrt(lambda_normal)

        # Rays gradient (이웃 차분으로 근사)
        rx = rays[:, 1:, :] - rays[:, :-1, :]   # H x (W-1) x 3
        ry = rays[1:, :, :] - rays[:-1, :, :]   # (H-1) x W x 3

        # ---------- Horizontal (left p -> right q) ----------
        mask_any = (hole_mask[:, :-1] | hole_mask[:, 1:])
        if mask_any.any():
            n_p = n[:, :-1, :][mask_any]
            n_q = n[:,  1:, :][mask_any]
            r_p = rays[:, :-1, :][mask_any]
            rx_p = rx[mask_any]
            p_idx = idx_map[:, :-1][mask_any]
            q_idx = idx_map[:,  1:][mask_any]

            base_w = lamN * np.sqrt(w_e_h[mask_any].astype(np.float32))  # 엣지 보존
            ww, valid = pair_weight_and_mask(n_p, n_q, base_w)

            # residual: (n_p·r_p)*(z_q - z_p) + (n_p·rx_p)*z_p = 0
            # => c_p*z_p + c_q*z_q = 0  where  c_p = (n_p·rx_p - n_p·r_p), c_q = (n_p·r_p)
            a = np.sum(n_p * r_p, axis=1).astype(np.float32)   # n_p·r_p
            b = np.sum(n_p * rx_p, axis=1).astype(np.float32)  # n_p·rx_p
            c_p = b - a
            c_q = a

            # coefficient normalization (스케일 편향 제거)
            den = np.sqrt(c_p * c_p + c_q * c_q) + 1e-6
            c_p /= den; c_q /= den

            # hole-hole
            ok = valid & (p_idx >= 0) & (q_idx >= 0)
            K = int(np.count_nonzero(ok))
            if K > 0:
                rr = np.arange(K)
                _push_row(rows,
                          np.concatenate([ww[ok]*c_p[ok], ww[ok]*c_q[ok]]),
                          np.concatenate([rr, rr]),
                          np.concatenate([p_idx[ok], q_idx[ok]]),
                          K, N)
                rhs_all.append(np.zeros(K, np.float32))

            # hole-known (q known)  —— RHS = ww * (c_q * z_a)
            ok = valid & (p_idx >= 0) & (q_idx < 0)
            K = int(np.count_nonzero(ok))
            if K > 0:
                rr = np.arange(K)
                _push_row(rows, ww[ok]*c_p[ok], rr, p_idx[ok], K, N)
                rhs = ww[ok] * (-c_q[ok] * depth_in[:, 1:][mask_any][ok].astype(np.float32))
                rhs_all.append(rhs)

            # known-hole (p known) —— RHS = ww * (-c_p * z_a)
            ok = valid & (p_idx < 0) & (q_idx >= 0)
            K = int(np.count_nonzero(ok))
            if K > 0:
                rr = np.arange(K)
                _push_row(rows, ww[ok]*c_q[ok], rr, q_idx[ok], K, N)
                rhs = ww[ok] * (-c_p[ok] * depth_in[:, :-1][mask_any][ok].astype(np.float32))
                rhs_all.append(rhs)

        # ---------- Vertical (up p -> down q) ----------
        mask_any = (hole_mask[:-1, :] | hole_mask[1:, :])
        if mask_any.any():
            n_p = n[:-1, :, :][mask_any]
            n_q = n[ 1:, :, :][mask_any]
            r_p = rays[:-1, :, :][mask_any]
            ry_p = ry[mask_any]
            p_idx = idx_map[:-1, :][mask_any]
            q_idx = idx_map[ 1:, :][mask_any]

            base_w = lamN * np.sqrt(w_e_v[mask_any].astype(np.float32))
            ww, valid = pair_weight_and_mask(n_p, n_q, base_w)

            a = np.sum(n_p * r_p, axis=1).astype(np.float32)   # n_p·r_p
            b = np.sum(n_p * ry_p, axis=1).astype(np.float32)  # n_p·ry_p
            c_p = b - a
            c_q = a

            den = np.sqrt(c_p * c_p + c_q * c_q) + 1e-6
            c_p /= den; c_q /= den

            ok = valid & (p_idx >= 0) & (q_idx >= 0)
            K = int(np.count_nonzero(ok))
            if K > 0:
                rr = np.arange(K)
                _push_row(rows,
                          np.concatenate([ww[ok]*c_p[ok], ww[ok]*c_q[ok]]),
                          np.concatenate([rr, rr]),
                          np.concatenate([p_idx[ok], q_idx[ok]]),
                          K, N)
                rhs_all.append(np.zeros(K, np.float32))

            ok = valid & (p_idx >= 0) & (q_idx < 0)
            K = int(np.count_nonzero(ok))
            if K > 0:
                rr = np.arange(K)
                _push_row(rows, ww[ok]*c_p[ok], rr, p_idx[ok], K, N)
                rhs = ww[ok] * (-c_q[ok] * depth_in[1:, :][mask_any][ok].astype(np.float32))
                rhs_all.append(rhs)

            ok = valid & (p_idx < 0) & (q_idx >= 0)
            K = int(np.count_nonzero(ok))
            if K > 0:
                rr = np.arange(K)
                _push_row(rows, ww[ok]*c_q[ok], rr, q_idx[ok], K, N)
                rhs = ww[ok] * (-c_p[ok] * depth_in[:-1, :][mask_any][ok].astype(np.float32))
                rhs_all.append(rhs)

    # ===== (S) Smoothing: z_p - z_q = 0 (hole-hole만, 엣지 보존) =====
    if lambda_smooth > 0:
        lamS = np.sqrt(lambda_smooth)

        # Horizontal
        mask = hole_mask[:, :-1] & hole_mask[:, 1:]
        if mask.any():
            ww = lamS * np.sqrt(w_e_h[mask].astype(np.float32))
            p_idx = idx_map[:, :-1][mask]
            q_idx = idx_map[:,  1:][mask]
            K = p_idx.size
            rr = np.arange(K)
            _push_row(rows,
                      np.concatenate([ww, -ww]),
                      np.concatenate([rr, rr]),
                      np.concatenate([p_idx, q_idx]),
                      K, N)
            rhs_all.append(np.zeros(K, np.float32))

        # Vertical
        mask = hole_mask[:-1, :] & hole_mask[1:, :]
        if mask.any():
            ww = lamS * np.sqrt(w_e_v[mask].astype(np.float32))
            p_idx = idx_map[:-1, :][mask]
            q_idx = idx_map[ 1:, :][mask]
            K = p_idx.size
            rr = np.arange(K)
            _push_row(rows,
                      np.concatenate([ww, -ww]),
                      np.concatenate([rr, rr]),
                      np.concatenate([p_idx, q_idx]),
                      K, N)
            rhs_all.append(np.zeros(K, np.float32))

    # ===== (D) Data: hole-경계 anchor  (엣지 가중 재적용) =====
    if lambda_data > 0:
        lamD = np.sqrt(lambda_data)

        # Horizontal (p=hole, q=known)
        mask = hole_mask[:, :-1] & known_mask[:, 1:]
        if mask.any():
            p_idx = idx_map[:, :-1][mask]
            z_a  = depth_in[:, 1:][mask].astype(np.float32)
            ww   = lamD * np.sqrt(w_e_h[mask].astype(np.float32))
            K = p_idx.size; rr = np.arange(K)
            _push_row(rows, ww, rr, p_idx, K, N); rhs_all.append(ww * z_a)

        # Horizontal (p=known, q=hole)
        mask = known_mask[:, :-1] & hole_mask[:, 1:]
        if mask.any():
            q_idx = idx_map[:, 1:][mask]
            z_a  = depth_in[:, :-1][mask].astype(np.float32)
            ww   = lamD * np.sqrt(w_e_h[mask].astype(np.float32))
            K = q_idx.size; rr = np.arange(K)
            _push_row(rows, ww, rr, q_idx, K, N); rhs_all.append(ww * z_a)

        # Vertical (p=hole, q=known)
        mask = hole_mask[:-1, :] & known_mask[1:, :]
        if mask.any():
            p_idx = idx_map[:-1, :][mask]
            z_a  = depth_in[1:, :][mask].astype(np.float32)
            ww   = lamD * np.sqrt(w_e_v[mask].astype(np.float32))
            K = p_idx.size; rr = np.arange(K)
            _push_row(rows, ww, rr, p_idx, K, N); rhs_all.append(ww * z_a)

        # Vertical (p=known, q=hole)
        mask = known_mask[:-1, :] & hole_mask[1:, :]
        if mask.any():
            q_idx = idx_map[1:, :][mask]
            z_a  = depth_in[:-1, :][mask].astype(np.float32)
            ww   = lamD * np.sqrt(w_e_v[mask].astype(np.float32))
            K = q_idx.size; rr = np.arange(K)
            _push_row(rows, ww, rr, q_idx, K, N); rhs_all.append(ww * z_a)

    # ===== (R) Screen: z ≈ z_init (모든 hole) =====
    if lambda_screen > 0:
        lamR = np.sqrt(lambda_screen)
        ww = lamR * np.ones(N, np.float32)
        rr = np.arange(N)
        _push_row(rows, ww, rr, np.arange(N), N, N)
        rhs_all.append(ww * depth_init[hole_mask].astype(np.float32))

    # ===== Assemble & Solve =====
    A = vstack(rows).tocsr()
    b = np.concatenate(rhs_all).astype(np.float32)

    if solver == "cg":
        AtA = (A.T @ A).tocsr()
        Atb = A.T @ b
        M = diags(1.0 / np.clip(AtA.diagonal(), 1e-6, None))
        z_vec, _ = cg(AtA, Atb, tol=tol, maxiter=maxiter, M=M)
    else:
        sol = lsmr(A, b, atol=tol, btol=tol, maxiter=maxiter)
        z_vec = sol[0]

    out = depth_init.astype(np.float32).copy()
    out[hole_mask] = z_vec.astype(np.float32)
    return out
