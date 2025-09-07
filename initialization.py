import hashlib
import time

import numpy as np
from scipy.sparse import coo_matrix, diags, vstack
from scipy.sparse.linalg import cg, lsmr, splu
from scipy.ndimage import zoom


# Cache for Poisson matrix decompositions (keyed by geometry/weights)
_POISSON_CACHE: dict[str, dict] = {}


def initial_guess_logpoisson_completion(
    depth_in: np.ndarray,
    known_mask: np.ndarray,
    hole_mask: np.ndarray,
    guide_gray: np.ndarray | None,
    n_guide: np.ndarray | None = None,
    lambda_grad: float = 3.0,
    lambda_smooth: float = 0.2,
    edge_alpha: float = 6.0,
    lambda_normal_edge: float = 0.0,
    tol: float = 1e-4,
    maxiter: int = 300,
    clip_min: float | None = 0.0,
    clip_max: float | None = None,
    use_cache: bool = False,
    approx_iters: int | None = None,
    coarse_factor: int = 1,
    precondition: bool = True,
) -> np.ndarray:
    """
    Log-Poisson completion을 사용해 hole 영역만 변수를 두고 초기 깊이를 추정합니다.
    - (B) log-기울기 전파 (u_q - u_p = 0)
    - (C) 엣지-가중 스무딩
    n_guide가 주어지면 법선 유사도를 이용한 추가 엣지 가중치를 적용합니다.
    """
    H, W = depth_in.shape

    if coarse_factor > 1:
        # Solve on a coarse grid then upsample.
        sf = 1.0 / float(coarse_factor)
        depth_small = zoom(depth_in, sf, order=1)
        known_small = zoom(known_mask.astype(np.float32), sf, order=0) >= 0.5
        hole_small = zoom(hole_mask.astype(np.float32), sf, order=0) >= 0.5
        guide_small = None if guide_gray is None else zoom(guide_gray, sf, order=1)
        n_small = None if n_guide is None else zoom(n_guide, (sf, sf, 1), order=1)
        low_res = initial_guess_logpoisson_completion(
            depth_small,
            known_small,
            hole_small,
            guide_small,
            n_small,
            lambda_grad=lambda_grad,
            lambda_smooth=lambda_smooth,
            edge_alpha=edge_alpha,
            lambda_normal_edge=lambda_normal_edge,
            tol=tol,
            maxiter=maxiter,
            clip_min=clip_min,
            clip_max=clip_max,
            use_cache=False,
            approx_iters=approx_iters,
            coarse_factor=1,
            precondition=precondition,
        )
        up = zoom(low_res, coarse_factor, order=1)
        up = up[:H, :W]
        out = depth_in.astype(np.float32).copy()
        out[hole_mask] = up[hole_mask]
        if clip_min is not None:
            out = np.maximum(out, clip_min)
        if clip_max is not None:
            out = np.minimum(out, clip_max)
        return out

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

    if n_guide is not None:
        n = n_guide.astype(np.float32)
        n /= np.linalg.norm(n, axis=2, keepdims=True).clip(1e-6, None)
        sim_h = np.abs(np.sum(n[:, :-1, :] * n[:, 1:, :], axis=2))
        sim_v = np.abs(np.sum(n[:-1, :, :] * n[1:, :, :], axis=2))
        w_e_h *= np.exp(-lambda_normal_edge * (1.0 - sim_h)).astype(np.float32)
        w_e_v *= np.exp(-lambda_normal_edge * (1.0 - sim_v)).astype(np.float32)

    # Cache lookup
    cache_entry = None
    use_cached_A = False
    if use_cache:
        h = hashlib.md5()
        h.update(hole_mask.tobytes())
        h.update(known_mask.tobytes())
        h.update(w_e_h.tobytes())
        h.update(w_e_v.tobytes())
        h.update(np.array([lambda_grad, lambda_smooth], dtype=np.float32).tobytes())
        key = h.hexdigest()
        cache_entry = _POISSON_CACHE.get(key)
        use_cached_A = cache_entry is not None
    else:
        key = ""

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

    rows = [] if not use_cached_A else None
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
        ww = weight * np.sqrt(w_edge[mask_pair])
        m = ww.size
        if not use_cached_A:
            p_idx = p_sel[mask_pair]
            r = np.arange(m)
        if rhs_known is None and rhs_arr is None:
            # hole↔hole: u_q - u_p = 0
            if not use_cached_A:
                q_idx = q_sel[mask_pair]  # rhs 있을 때는 계산하지 않음
                rows.append(coo_matrix(
                    (np.concatenate([-ww, ww]),
                     (np.concatenate([r, r]), np.concatenate([p_idx, q_idx]))),
                    shape=(m, N)
                ))
            b_all.append(np.zeros(m, dtype=np.float32))
        else:
            # hole↔known: u_var - u_known = 0
            rhs = (rhs_known if rhs_known is not None else rhs_arr)[mask_pair].astype(np.float32)
            if not use_cached_A:
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
    if not use_cached_A and not rows:
        yx = np.argwhere(hole_mask)[0]
        anchor_idx = idx_map[tuple(yx)]
        rows.append(coo_matrix(([1.0], ([0], [anchor_idx])), shape=(1, N)))
        b_all.append(np.array([u0[tuple(yx)]], dtype=np.float32))

    if use_cached_A:
        A = cache_entry["A"]
    else:
        A = vstack(rows).tocsr()

    b = np.concatenate(b_all).astype(np.float32)

    if approx_iters is not None:
        if use_cached_A:
            AtA = cache_entry.get("AtA")
            M = cache_entry.get("M")
            if AtA is None:
                AtA = A.T @ A
                M = diags(1.0 / np.clip(AtA.diagonal(), 1e-6, None)) if precondition else None
                cache_entry["AtA"] = AtA
                cache_entry["M"] = M
        else:
            AtA = A.T @ A
            M = diags(1.0 / np.clip(AtA.diagonal(), 1e-6, None)) if precondition else None
            if use_cache:
                _POISSON_CACHE[key] = {"A": A, "AtA": AtA, "M": M}
        Atb = A.T @ b
        u, _ = cg(AtA, Atb, maxiter=approx_iters, rtol=tol, atol=0.0, M=M)
        u = u.astype(np.float32)
    else:
        if use_cached_A:
            solver = cache_entry.get("solver")
            if solver is None:
                AtA = (A.T @ A).tocsc()
                solver = splu(AtA)
                cache_entry["solver"] = solver
        elif use_cache:
            AtA = (A.T @ A).tocsc()
            solver = splu(AtA)
            _POISSON_CACHE[key] = {"A": A, "solver": solver}
        if use_cached_A or use_cache:
            Atb = A.T @ b
            u = solver.solve(Atb).astype(np.float32)
        else:
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


def benchmark_initialization(
    depth_in: np.ndarray,
    known_mask: np.ndarray,
    hole_mask: np.ndarray,
    guide_gray: np.ndarray | None = None,
    n_guide: np.ndarray | None = None,
    lambda_grad: float = 3.0,
    lambda_smooth: float = 0.2,
    edge_alpha: float = 6.0,
    lambda_normal_edge: float = 0.0,
    tol: float = 1e-4,
    maxiter: int = 300,
    clip_min: float | None = 0.0,
    clip_max: float | None = None,
    coarse_factors: tuple[int, ...] = (1, 2, 4),
    approx_iters_list: tuple[int | None, ...] = (None, 20, 50),
    repeats: int = 1,
) -> tuple[np.ndarray, list[dict]]:
    """Benchmark accuracy-speed tradeoff for initialization.

    Returns baseline result and list of records containing
    coarse factor, iteration count, runtime, and MSE vs baseline.
    """
    baseline = initial_guess_logpoisson_completion(
        depth_in,
        known_mask,
        hole_mask,
        guide_gray,
        n_guide,
        lambda_grad=lambda_grad,
        lambda_smooth=lambda_smooth,
        edge_alpha=edge_alpha,
        lambda_normal_edge=lambda_normal_edge,
        tol=tol,
        maxiter=maxiter,
        clip_min=clip_min,
        clip_max=clip_max,
        use_cache=False,
    )

    records: list[dict] = []
    for cf in coarse_factors:
        for it in approx_iters_list:
            if cf == 1 and it is None:
                continue
            t_sum = 0.0
            pred = None
            for _ in range(repeats):
                start = time.time()
                pred = initial_guess_logpoisson_completion(
                    depth_in,
                    known_mask,
                    hole_mask,
                    guide_gray,
                    n_guide,
                    lambda_grad=lambda_grad,
                    lambda_smooth=lambda_smooth,
                    edge_alpha=edge_alpha,
                    lambda_normal_edge=lambda_normal_edge,
                    tol=tol,
                    maxiter=maxiter,
                    clip_min=clip_min,
                    clip_max=clip_max,
                    use_cache=False,
                    approx_iters=it,
                    coarse_factor=cf,
                )
                t_sum += time.time() - start
            mse = np.mean((pred[hole_mask] - baseline[hole_mask]) ** 2)
            records.append(
                {
                    "coarse_factor": cf,
                    "approx_iters": it,
                    "time": t_sum / repeats,
                    "mse": float(mse),
                }
            )
    return baseline, records
