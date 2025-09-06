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

    K_inv = np.linalg.inv(K).astype(np.float32)
    rays = _depth_to_points(np.ones((H, W), np.float32), K_inv)

    for it in range(cfg.gn_iters):
        # ---------- 1) 노멀 및 잔차 ----------
        X = z[..., None] * rays
        if not cfg.use_central_diff:
            raise NotImplementedError("Only central difference is supported")
        dXu = np.zeros_like(X)
        dXv = np.zeros_like(X)
        dXu[:, 1:-1, :] = 0.5 * (X[:, 2:, :] - X[:, :-2, :])
        dXu[:, 0, :] = X[:, 1, :] - X[:, 0, :]
        dXu[:, -1, :] = X[:, -1, :] - X[:, -2, :]

        dXv[1:-1, :, :] = 0.5 * (X[2:, :, :] - X[:-2, :, :])
        dXv[0, :, :] = X[1, :, :] - X[0, :, :]
        dXv[-1, :, :] = X[-1, :, :] - X[-2, :, :]

        n = np.cross(dXu, dXv)
        norm = np.linalg.norm(n, axis=2)
        n_curr = n / (norm[..., None] + 1e-12)

        # dot residual r = 1 - n·n_g (작을수록 좋음)
        r_full = 1.0 - np.sum(n_curr * n_guide, axis=2)

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

        # ---------- 2) J_normal Jacobian (vectorized) ----------
        n_dot_ng = np.sum(n_curr * n_guide, axis=2)

        def dn_to_dr(dn: np.ndarray) -> np.ndarray:
            dn_dot_ng = np.sum(dn * n_guide, axis=2)
            n_hat_dot_dn = np.sum(n_curr * dn, axis=2)
            return -(dn_dot_ng - n_hat_dot_dn * n_dot_ng) / (norm + 1e-12)

        dn_c = np.zeros_like(n)
        dn_e = np.zeros_like(n)
        dn_w = np.zeros_like(n)
        dn_s = np.zeros_like(n)
        dn_n = np.zeros_like(n)

        # center contributions (boundaries)
        dn_c[:, 0, :] += np.cross(-rays[:, 0, :], dXv[:, 0, :])
        dn_c[:, -1, :] += np.cross(rays[:, -1, :], dXv[:, -1, :])
        dn_c[0, :, :] += np.cross(dXu[0, :, :], -rays[0, :, :])
        dn_c[-1, :, :] += np.cross(dXu[-1, :, :], rays[-1, :, :])

        # east/west neighbors
        dn_e[:, :-1, :] = np.cross(0.5 * rays[:, 1:, :], dXv[:, :-1, :])
        dn_w[:, 1:, :] = np.cross(-0.5 * rays[:, :-1, :], dXv[:, 1:, :])

        # south/north neighbors
        dn_s[:-1, :, :] = np.cross(dXu[:-1, :, :], 0.5 * rays[1:, :, :])
        dn_n[1:, :, :] = np.cross(dXu[1:, :, :], -0.5 * rays[:-1, :, :])

        dr_c = dn_to_dr(dn_c)
        dr_e = dn_to_dr(dn_e)
        dr_w = dn_to_dr(dn_w)
        dr_s = dn_to_dr(dn_s)
        dr_n = dn_to_dr(dn_n)

        w_i = np.sqrt(cfg.lambda_normal) * np.sqrt(w_normal)
        rows_list = []
        cols_list = []
        data_list = []

        row_ids = np.arange(M, dtype=np.int32)

        # center
        cols_c = idx_map[ys, xs]
        data_c = w_i * dr_c[ys, xs]
        rows_list.append(row_ids)
        cols_list.append(cols_c)
        data_list.append(data_c)

        # east
        mask = xs + 1 < W
        cols_e = idx_map[ys[mask], xs[mask] + 1]
        valid = cols_e >= 0
        rows_list.append(row_ids[mask][valid])
        cols_list.append(cols_e[valid])
        data_list.append((w_i * dr_e[ys, xs])[mask][valid])

        # west
        mask = xs - 1 >= 0
        cols_w = idx_map[ys[mask], xs[mask] - 1]
        valid = cols_w >= 0
        rows_list.append(row_ids[mask][valid])
        cols_list.append(cols_w[valid])
        data_list.append((w_i * dr_w[ys, xs])[mask][valid])

        # south
        mask = ys + 1 < H
        cols_s = idx_map[ys[mask] + 1, xs[mask]]
        valid = cols_s >= 0
        rows_list.append(row_ids[mask][valid])
        cols_list.append(cols_s[valid])
        data_list.append((w_i * dr_s[ys, xs])[mask][valid])

        # north
        mask = ys - 1 >= 0
        cols_n = idx_map[ys[mask] - 1, xs[mask]]
        valid = cols_n >= 0
        rows_list.append(row_ids[mask][valid])
        cols_list.append(cols_n[valid])
        data_list.append((w_i * dr_n[ys, xs])[mask][valid])

        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        data = np.concatenate(data_list)
        b_norm = w_i * (-r_full[normal_roi])

        # J_normal, b_normal
        if len(rows) == 0:
            J_normal = coo_matrix((0, N), dtype=np.float32)
            b_normal = np.zeros((0,), dtype=np.float32)
        else:
            J_normal = coo_matrix((data.astype(np.float32),
                                   (rows.astype(np.int32),
                                    cols.astype(np.int32))),
                                  shape=(M, N)).tocsr()
            b_normal = b_norm.astype(np.float32)

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
