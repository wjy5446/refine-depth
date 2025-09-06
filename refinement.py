import numpy as np
from dataclasses import dataclass
from typing import Literal, Tuple
from scipy.sparse import coo_matrix, vstack, csr_matrix
from scipy.sparse.linalg import lsmr


@dataclass
class GNConfig:
    # Weights
    lambda_normal: float = 2.0     # 노멀 정합
    lambda_smooth: float = 0.3     # 엣지-가중 스무딩
    lambda_screen: float = 0.1     # z ~= z_init 앵커 (드리프트 억제)

    # IRLS / robust
    loss: Literal["charbonnier", "huber"] = "charbonnier"
    eps_charb: float = 1e-3
    huber_delta: float = 1.0
    robust_on_normal: bool = True

    # GN
    gn_iters: int = 2
    step_clip_frac: float = 0.5    # |δz| <= step_clip_frac * z (폭주 방지)
    delta_eps_frac: float = 1e-3   # 수치미분 perturb (상대)

    # Solver
    atol: float = 1e-5
    btol: float = 1e-5
    maxiter: int = 300

    # Edge weight
    edge_alpha: float = 6.0        # w_edge = exp(-alpha*|I_p - I_q|)

    # Speed-accuracy tradeoff
    normal_stride: int = 1         # 1이면 모든 픽셀, 2/3로 올리면 하위표본화
    use_central_diff: bool = True  # 노멀 계산용 중앙차분

    # Depth clamp
    clip_min: float | None = 0.0
    clip_max: float | None = None


def _make_uv_grid(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    v = np.arange(H, dtype=np.float32)
    u = np.arange(W, dtype=np.float32)
    U, V = np.meshgrid(u, v)
    return U, V


def _depth_to_points(z: np.ndarray, K_inv: np.ndarray) -> np.ndarray:
    """ z(H,W) -> X(H,W,3) with X = z * K^{-1}[u,v,1]^T """
    H, W = z.shape
    U, V = _make_uv_grid(H, W)
    ones = np.ones_like(U)
    pix = np.stack([U, V, ones], axis=-1).reshape(-1, 3)  # (HW,3)
    rays = (K_inv @ pix.T).T.reshape(H, W, 3)             # (H,W,3)
    X = z[..., None] * rays
    return X


def _normals_from_depth(z: np.ndarray, K: np.ndarray, use_central: bool = True) -> np.ndarray:
    """ 원근 고려 노멀: X=z*K^{-1}[u,v,1]^T, n ∝ dX/du × dX/dv (정규화) """
    H, W = z.shape
    K_inv = np.linalg.inv(K).astype(np.float32)
    X = _depth_to_points(z, K_inv)  # (H,W,3)

    # 미분: 중심/전진 차분
    if use_central:
        dXu = np.zeros_like(X)
        dXv = np.zeros_like(X)
        dXu[:, 1:-1, :] = 0.5 * (X[:, 2:, :] - X[:, :-2, :])
        dXu[:, 0, :] = X[:, 1, :] - X[:, 0, :]
        dXu[:, -1, :] = X[:, -1, :] - X[:, -2, :]

        dXv[1:-1, :, :] = 0.5 * (X[2:, :, :] - X[:-2, :, :])
        dXv[0, :, :] = X[1, :, :] - X[0, :, :]
        dXv[-1, :, :] = X[-1, :, :] - X[-2, :, :]
    else:
        dXu = np.zeros_like(X)
        dXv = np.zeros_like(X)
        dXu[:, :-1, :] = X[:, 1:, :] - X[:, :-1, :]
        dXu[:, -1, :] = dXu[:, -2, :]
        dXv[:-1, :, :] = X[1:, :, :] - X[:-1, :, :]
        dXv[-1, :, :] = dXv[-2, :, :]

    n = np.cross(dXu, dXv)  # (H,W,3)
    norm = np.linalg.norm(n, axis=2, keepdims=True) + 1e-12
    n = n / norm
    return n.astype(np.float32)


def _edge_weights(guide_gray: np.ndarray | None, H: int, W: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    if guide_gray is None:
        return np.ones((H, W-1), np.float32), np.ones((H-1, W), np.float32)
    diff_h = np.abs(guide_gray[:, 1:] - guide_gray[:, :-1]).astype(np.float32)
    diff_v = np.abs(guide_gray[1:, :] - guide_gray[:-1, :]).astype(np.float32)
    w_e_h = np.exp(-alpha * diff_h).astype(np.float32)
    w_e_v = np.exp(-alpha * diff_v).astype(np.float32)
    return w_e_h, w_e_v


def _robust_weight(res: np.ndarray, loss: str, eps_charb: float, huber_delta: float) -> np.ndarray:
    if loss == "charbonnier":
        return 1.0 / np.sqrt(res**2 + eps_charb**2)
    # huber
    absr = np.abs(res)
    w = np.ones_like(res)
    large = absr > huber_delta
    w[large] = huber_delta / absr[large]
    return w


def refine_depth_match_normals_gn_completion(
    depth_init: np.ndarray,       # 초기 깊이 (Poisson 결과 권장)
    known_mask: np.ndarray,       # 알려진 영역(True면 고정)
    hole_mask: np.ndarray,        # 미지수 영역(True면 변수)
    n_guide: np.ndarray,          # (H,W,3) 가이드 노멀(단위벡터)
    K: np.ndarray,                # 3x3
    guide_gray: np.ndarray | None,
    cfg: GNConfig = GNConfig(),
) -> np.ndarray:
    """
    노멀 정합을 주항으로 하는 GN/IRLS 정련 (변수=hole 영역 깊이 z).
    """
    H, W = depth_init.shape
    assert n_guide.shape == (H, W, 3)
    assert known_mask.shape == (H, W) and hole_mask.shape == (H, W)

    # 변수 인덱싱 (hole-only)
    idx_map = -np.ones((H, W), dtype=np.int32)
    idx_map[hole_mask] = np.arange(hole_mask.sum(), dtype=np.int32)
    N = int(hole_mask.sum())
    if N == 0:
        out = depth_init.astype(np.float32).copy()
        if cfg.clip_min is not None:
            out = np.maximum(out, cfg.clip_min)
        if cfg.clip_max is not None:
            out = np.minimum(out, cfg.clip_max)
        return out

    z = depth_init.astype(np.float32).copy()
    z0 = z.copy()

    # 스무딩용 엣지 가중
    w_e_h, w_e_v = _edge_weights(guide_gray, H, W, cfg.edge_alpha)

    # 스무딩 간선 mask
    Lh_Rh = hole_mask[:, :-1] & hole_mask[:, 1:]
    Uh_Dh = hole_mask[:-1, :] & hole_mask[1:, :]

    # 숫자 편미분에서 사용할 perturb 사이즈
    def _perturb_amount(val: np.ndarray) -> np.ndarray:
        return np.maximum(cfg.delta_eps_frac * np.maximum(val, 1e-3), 1e-6)

    K_inv = np.linalg.inv(K).astype(np.float32)

    for it in range(cfg.gn_iters):
        # ---------- 1) 노멀 및 잔차 ----------
        n_curr = _normals_from_depth(z, K, use_central=cfg.use_central_diff)  # (H,W,3)
        # dot residual r = 1 - n·n_g (작을수록 좋음)
        r_full = 1.0 - np.sum(n_curr * n_guide, axis=2)                       # (H,W)
        # ROI 샘플링 (stride) - 계산량 절약
        stride = max(1, int(cfg.normal_stride))
        sample_mask = np.zeros_like(hole_mask)
        sample_mask[::stride, ::stride] = True
        normal_roi = hole_mask & sample_mask

        ys, xs = np.where(normal_roi)
        M = len(ys)         # 노멀 잔차 개수
        if M == 0:
            break

        # IRLS 로버스트 가중
        if cfg.robust_on_normal:
            w_normal = _robust_weight(r_full[normal_roi], cfg.loss, cfg.eps_charb, cfg.huber_delta).astype(np.float32)
        else:
            w_normal = np.ones(M, dtype=np.float32)

        # ---------- 2) J_normal 수치미분(5-point stencil) ----------
        # 각 잔차 r(y,x)는 z[y,x], z[y±1,x], z[y,x±1]에만 의해 변한다고 가정
        rows = []
        data = []
        cols = []
        b_norm = []

        # 미리 현재 r_p 및 perturb 크기 준비
        r0 = r_full[normal_roi].astype(np.float32)
        # 각 이웃의 (dy,dx)
        nbrs = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]

        # perturb용 공용 z buffer를 카피하지 않고 in-place로 조정/복구
        # 안전을 위해 복사본 하나 유지
        z_buf = z

        for i in range(M):
            y = ys[i]
            x = xs[i]
            # J 행 인덱스
            row_id = i
            r_base = r0[i]
            w_i = np.sqrt(cfg.lambda_normal) * np.sqrt(w_normal[i])

            for (dy, dx) in nbrs:
                yy = y + dy
                xx = x + dx
                if yy < 0 or yy >= H or xx < 0 or xx >= W:
                    continue
                col = idx_map[yy, xx]
                if col < 0:  # known이면 변수 아님
                    continue

                # 상대 perturb
                dz = _perturb_amount(z_buf[yy, xx])
                z_buf[yy, xx] += dz
                # r(y,x) 재계산 (이 픽셀의 r만 필요하므로 국소 재계산을 하고 싶지만
                # 간단함을 위해 노멀 전체를 다시 계산 -> 정확하지만 느릴 수 있음)
                n_tmp = _normals_from_depth(z_buf, K, use_central=cfg.use_central_diff)
                r_pert = 1.0 - np.dot(n_tmp[y, x, :], n_guide[y, x, :])
                # 복구
                z_buf[yy, xx] -= dz

                # dr/dz ≈ (r_pert - r_base)/dz
                j = (r_pert - r_base) / (dz + 1e-12)

                rows.append(row_id)
                cols.append(col)
                data.append(w_i * j)

            # RHS: sqrt(λ_n)*sqrt(w_i)*( - r_base )  (표준 GN: J δ = -r)
            b_norm.append(w_i * (r_base * -1.0))

        # J_normal, b_normal
        if len(rows) == 0:
            # 노멀 항이 모두 known에만 걸렸다면 스킵
            J_normal = coo_matrix((0, N), dtype=np.float32)
            b_normal = np.zeros((0,), dtype=np.float32)
        else:
            J_normal = coo_matrix((np.array(data, dtype=np.float32),
                                   (np.array(rows, dtype=np.int32),
                                    np.array(cols, dtype=np.int32))),
                                  shape=(M, N)).tocsr()
            b_normal = np.array(b_norm, dtype=np.float32)

        # ---------- 3) 스무딩/스크린 선형항 (z 기준) ----------
        rows_ls = []
        b_ls = []

        lam_s = np.sqrt(cfg.lambda_smooth)
        lam_scr = np.sqrt(cfg.lambda_screen)

        def add_pair(mask_pair, p_sel, q_sel, w_edge, weight):
            if not mask_pair.any():
                return
            p_idx = p_sel[mask_pair]
            q_idx = q_sel[mask_pair]
            ww = weight * np.sqrt(w_edge[mask_pair]).astype(np.float32)

            m = p_idx.size
            r = np.arange(m, dtype=np.int32)

            rows_ls.append(coo_matrix(
                (np.concatenate([-ww, ww]),
                 (np.concatenate([r, r]), np.concatenate([p_idx, q_idx]))),
                shape=(m, N)
            ))
            b_ls.append(np.zeros(m, dtype=np.float32))

        # hole↔hole smooth (가로/세로)
        if cfg.lambda_smooth > 0:
            add_pair(Lh_Rh, idx_map[:, :-1], idx_map[:, 1:], w_e_h, lam_s)
            add_pair(Uh_Dh, idx_map[:-1, :], idx_map[1:, :], w_e_v, lam_s)

        # screen: z - z0 = 0 (hole 픽셀에만)
        if cfg.lambda_screen > 0:
            p_idx = idx_map[hole_mask]
            m = p_idx.size
            r = np.arange(m, dtype=np.int32)
            rows_ls.append(coo_matrix((np.ones(m, np.float32) * lam_scr, (r, p_idx)), shape=(m, N)))
            b_ls.append(lam_scr * z0[hole_mask].astype(np.float32))

        A_ls = vstack(rows_ls).tocsr() if rows_ls else coo_matrix((0, N), dtype=np.float32).tocsr()
        b_ls = np.concatenate(b_ls).astype(np.float32) if b_ls else np.zeros((0,), dtype=np.float32)

        # ---------- 4) 선형계 풀기: [A_ls; J_normal] δ = [b_ls; b_normal] ----------
        A = vstack([A_ls, J_normal]).tocsr()
        b = np.concatenate([b_ls, b_normal]).astype(np.float32)

        if A.shape[0] == 0:
            # 더 정제할 게 없음
            break

        sol = lsmr(A, b, atol=cfg.atol, btol=cfg.btol, maxiter=cfg.maxiter)
        dz = sol[0].astype(np.float32)

        # ---------- 5) 업데이트 & 안정화 ----------
        dz_img = np.zeros_like(z, dtype=np.float32)
        dz_img[hole_mask] = dz

        # step clamp (상대)
        max_step = cfg.step_clip_frac * np.maximum(z, 1e-6)
        dz_img = np.clip(dz_img, -max_step, max_step)

        z = z + dz_img
        if cfg.clip_min is not None:
            z = np.maximum(z, cfg.clip_min)
        if cfg.clip_max is not None:
            z = np.minimum(z, cfg.clip_max)

    return z
