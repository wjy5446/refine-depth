import numpy as np
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsmr


def initial_guess_logpoisson_completion(
    depth_in: np.ndarray,
    known_mask: np.ndarray,
    hole_mask: np.ndarray,
    guide_gray: np.ndarray | None,
    lambda_grad: float = 3.0,
    lambda_smooth: float = 0.2,
    edge_alpha: float = 6.0,
    tol: float = 1e-4,
    maxiter: int = 300,
    clip_min: float | None = 0.0,
    clip_max: float | None = None,
) -> np.ndarray:
    """
    Log-Poisson completion을 사용해 hole 영역만 변수를 두고 초기 깊이를 추정합니다.
    - (B) log-기울기 전파 (u_q - u_p = 0)
    - (C) 엣지-가중 스무딩
    """
    H, W = depth_in.shape

    # log-depth
    depth0 = np.maximum(depth_in.astype(np.float32), 1e-6)
    u0 = np.log(depth0)

    # edge weights
    if guide_gray is not None:
        diff_h = np.abs(guide_gray[:, 1:] - guide_gray[:, :-1]).astype(np.float32)
        diff_v = np.abs(guide_gray[1:, :] - guide_gray[:-1, :]).astype(np.float32)
        w_e_h = np.exp(-edge_alpha * diff_h).astype(np.float32)
        w_e_v = np.exp(-edge_alpha * diff_v).astype(np.float32)
    else:
        w_e_h = np.ones((H, W - 1), dtype=np.float32)
        w_e_v = np.ones((H - 1, W), dtype=np.float32)

    # 변수 인덱스: hole만
    idx_map = -np.ones((H, W), dtype=np.int32)
    idx_map[hole_mask] = np.arange(hole_mask.sum(), dtype=np.int32)
    N = int(hole_mask.sum())
    if N == 0:
        out = depth_in.astype(np.float32).copy()
        if clip_min is not None:
            out = np.maximum(out, clip_min)
        if clip_max is not None:
            out = np.minimum(out, clip_max)
        return out

    rows = []
    b_all = []

    lam_g = np.sqrt(lambda_grad)
    lam_s = np.sqrt(lambda_smooth)

    # 간선 마스크 (좌우/상하)
    Lh_Rh = hole_mask[:, :-1] & hole_mask[:, 1:]
    Lh_Rk = hole_mask[:, :-1] & known_mask[:, 1:]
    Lk_Rh = known_mask[:, :-1] & hole_mask[:, 1:]
    Uh_Dh = hole_mask[:-1, :] & hole_mask[1:, :]
    Uh_Dk = hole_mask[:-1, :] & known_mask[1:, :]
    Uk_Dh = known_mask[:-1, :] & hole_mask[1:, :]

    def add_pair_terms(mask_pair, p_sel, q_sel, w_edge, weight, rhs_known=None, rhs_arr=None):
        """
        mask_pair: 간선 존재 마스크
        p_sel, q_sel: idx_map 부분뷰 (p, q의 변수 인덱스), rhs_known인 경우 q_sel은 사용하지 않음
        w_edge: 해당 방향 엣지 가중
        weight: 항 가중 (sqrt 적용된 값)
        rhs_known: None이면 hole↔hole, 배열이면 hole↔known에서 사용할 u_known(원본 u0의 부분뷰)
        rhs_arr: rhs_known과 동일(가독성용), 외부에서 u0의 부분뷰를 넘김
        """
        if not mask_pair.any():
            return
        p_idx = p_sel[mask_pair]
        ww = weight * np.sqrt(w_edge[mask_pair])
        m = p_idx.size
        r = np.arange(m)

        if rhs_known is None and rhs_arr is None:
            # hole↔hole: u_q - u_p = 0
            q_idx = q_sel[mask_pair]  # <<-- rhs 있을 때는 계산하지 않음
            rows.append(coo_matrix(
                (np.concatenate([-ww, ww]),
                 (np.concatenate([r, r]), np.concatenate([p_idx, q_idx]))),
                shape=(m, N)
            ))
            b_all.append(np.zeros(m, dtype=np.float32))
        else:
            # hole↔known: u_var - u_known = 0
            rhs = (rhs_known if rhs_known is not None else rhs_arr)[mask_pair].astype(np.float32)
            rows.append(coo_matrix((ww, (r, p_idx)), shape=(m, N)))
            b_all.append(ww * rhs)

    # (B) 기울기 전파
    add_pair_terms(Lh_Rh, idx_map[:, :-1], idx_map[:, 1:], w_e_h, lam_g)                  # hole↔hole (가로)
    add_pair_terms(Uh_Dh, idx_map[:-1, :], idx_map[1:, :], w_e_v, lam_g)                  # hole↔hole (세로)
    add_pair_terms(Lh_Rk, idx_map[:, :-1], None,         w_e_h, lam_g, rhs_arr=u0[:, 1:]) # hole↔known (가로 오른쪽)
    add_pair_terms(Lk_Rh, idx_map[:, 1:],  None,         w_e_h, lam_g, rhs_arr=u0[:, :-1])# hole↔known (가로 왼쪽)
    add_pair_terms(Uh_Dk, idx_map[:-1, :], None,         w_e_v, lam_g, rhs_arr=u0[1:, :]) # hole↔known (세로 아래)
    add_pair_terms(Uk_Dh, idx_map[1:, :],  None,         w_e_v, lam_g, rhs_arr=u0[:-1, :])# hole↔known (세로 위)

    # (C) 스무딩
    if lambda_smooth > 0:
        add_pair_terms(Lh_Rh, idx_map[:, :-1], idx_map[:, 1:], w_e_h, lam_s)
        add_pair_terms(Uh_Dh, idx_map[:-1, :], idx_map[1:, :], w_e_v, lam_s)
        add_pair_terms(Lh_Rk, idx_map[:, :-1], None,         w_e_h, lam_s, rhs_arr=u0[:, 1:])
        add_pair_terms(Lk_Rh, idx_map[:, 1:],  None,         w_e_h, lam_s, rhs_arr=u0[:, :-1])
        add_pair_terms(Uh_Dk, idx_map[:-1, :], None,         w_e_v, lam_s, rhs_arr=u0[1:, :])
        add_pair_terms(Uk_Dh, idx_map[1:, :],  None,         w_e_v, lam_s, rhs_arr=u0[:-1, :])

    # 안전장치: 간선이 전혀 없을 때(희귀) - 첫 변수 픽셀을 u0에 약하게 앵커
    if not rows:
        yx = np.argwhere(hole_mask)[0]
        anchor_idx = idx_map[tuple(yx)]
        rows.append(coo_matrix(([1.0], ([0], [anchor_idx])), shape=(1, N)))
        b_all.append(np.array([u0[tuple(yx)]], dtype=np.float32))

    # 선형 최소제곱 (LSMR)
    A = vstack(rows).tocsr()
    b = np.concatenate(b_all).astype(np.float32)
    sol = lsmr(A, b, atol=tol, btol=tol, maxiter=maxiter)
    u = sol[0].astype(np.float32)

    # 복원 & 클리핑
    out = depth_in.astype(np.float32).copy()
    out[hole_mask] = np.exp(u)
    if clip_min is not None:
        out = np.maximum(out, clip_min)
    if clip_max is not None:
        out = np.minimum(out, clip_max)
    return out
