import numpy as np


def _ax_ay(K, H, W):
    """카메라 내부 파라미터로부터 ax, ay 계산"""
    if K is None:
        yy, xx = np.mgrid[0:H, 0:W]
        ax = np.zeros((H, W), dtype=np.float32)
        ay = np.zeros((H, W), dtype=np.float32)
        fx = fy = 1.0
        return ax, ay, fx, fy
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    yy, xx = np.mgrid[0:H, 0:W]
    ax = (xx - cx) / fx
    ay = (yy - cy) / fy
    return ax.astype(np.float32), ay.astype(np.float32), fx, fy


def _depth_to_X(z, ax, ay):
    """깊이를 3D 좌표로 변환"""
    return np.stack([ax*z, ay*z, z], axis=-1)


def _normals_from_depth(z, ax, ay):
    """깊이로부터 노멀 벡터 계산"""
    X = _depth_to_X(z, ax, ay)
    Xu = (X[1:-1, 2:, :] - X[1:-1, 0:-2, :]) * 0.5
    Xv = (X[2:, 1:-1, :] - X[0:-2, 1:-1, :]) * 0.5
    n = np.cross(Xu, Xv)
    n_norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    n = n / n_norm
    H, W = z.shape
    n_full = np.zeros((H, W, 3), dtype=z.dtype)
    n_full[1:-1, 1:-1, :] = n
    n_full[0, :, :] = n_full[1, :, :]
    n_full[-1, :, :] = n_full[-2, :, :]
    n_full[:, 0, :] = n_full[:, 1, :]
    n_full[:, -1, :] = n_full[:, -2, :]
    return n_full


def _rho_weight(res, loss="charbonnier", eps_charb=1e-3, huber_delta=1.0):
    """로버스트 손실 함수의 가중치 계산"""
    if loss == "charbonnier":
        return (1.0 / np.sqrt(res*res + eps_charb*eps_charb)).astype(np.float32)
    ab = np.abs(res)
    return np.where(ab <= huber_delta, 1.0, huber_delta/ab).astype(np.float32)


def make_intrinsics(H, W, fx=600.0, fy=600.0):
    """카메라 내부 파라미터 행렬 생성"""
    cx = W/2.0
    cy = H/2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    return K


def compute_normals_from_depth(depth, K):
    """깊이로부터 노멀 벡터 계산 (그래디언트 기반)"""
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    z = depth.astype(np.float32, copy=False)

    dz_dx = np.gradient(z, axis=1).astype(np.float32)
    dz_dy = np.gradient(z, axis=0).astype(np.float32)

    nx = -dz_dx * np.float32(fx) / np.maximum(z, 1e-6)
    ny = -dz_dy * np.float32(fy) / np.maximum(z, 1e-6)
    nz = np.ones_like(z, dtype=np.float32)

    norm = np.sqrt(nx**2 + ny**2 + nz**2).astype(np.float32)
    nx /= norm
    ny /= norm
    nz /= norm
    return np.stack([nx, ny, nz], axis=-1).astype(np.float32)


def inpaint_completion(depth_in, known_mask, hole_mask, method='fast_marching'):
    """
    Fast Marching Method를 사용한 깊이 완성

    Args:
        depth_in: 입력 깊이 맵 (NaN/0 포함 가능)
        known_mask: 알려진 깊이 영역 마스크
        hole_mask: 채워야 할 홀 영역 마스크
        method: inpaint 방법 (현재는 fast_marching만 지원)

    Returns:
        Fast Marching으로 완성된 깊이 맵
    """
    H, W = depth_in.shape
    depth_out = depth_in.copy().astype(np.float32)

    if not hole_mask.any():
        return depth_out

    # NaN 값을 0으로 설정
    depth_clean = np.nan_to_num(depth_in, nan=0.0, posinf=0.0, neginf=0.0)

    # Fast Marching Method 사용
    return _fast_marching_inpaint(depth_clean, known_mask, hole_mask)


def _fast_marching_inpaint(depth_clean, known_mask, hole_mask):
    """
    Fast Marching Method를 사용한 inpainting
    """
    from scipy.ndimage import distance_transform_edt
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    H, W = depth_clean.shape

    # 1단계: Fast Marching으로 우선순위 계산
    known_inv = ~known_mask
    dist = distance_transform_edt(known_inv)

    # 2단계: 홀 영역을 거리 순으로 정렬
    hole_coords = np.column_stack(np.where(hole_mask))
    if len(hole_coords) == 0:
        return depth_clean

    # 거리 기반 우선순위 (가장자리부터 채우기)
    hole_distances = dist[hole_mask]
    sorted_indices = np.argsort(hole_distances)

    # 3단계: 순차적으로 채우기
    depth_result = depth_clean.copy()

    for idx in sorted_indices:
        y, x = hole_coords[idx]

        # 주변 알려진 값들의 가중 평균 계산
        weights = []
        values = []

        # 3x3 또는 5x5 주변 검색
        search_radius = 2
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                ny, nx = y + dy, x + dx
                if (0 <= ny < H and 0 <= nx < W and
                    known_mask[ny, nx] and not hole_mask[ny, nx]):

                    # 거리 기반 가중치 (가까울수록 높은 가중치)
                    d = np.sqrt(dy*dy + dx*dx)
                    if d > 0:
                        weight = 1.0 / (d * d)  # 거리의 제곱에 반비례
                        weights.append(weight)
                        values.append(depth_result[ny, nx])

        if values:
            # 가중 평균으로 값 계산
            weights = np.array(weights)
            values = np.array(values)
            weighted_avg = np.average(values, weights=weights)
            depth_result[y, x] = weighted_avg
        else:
            # 주변에 알려진 값이 없으면 평균값 사용
            mean_val = np.mean(depth_clean[known_mask])
            depth_result[y, x] = mean_val

    return depth_result




def linear_interpolation_completion(depth_in, known_mask, hole_mask):
    """
    선형 보간을 사용한 깊이 완성

    Args:
        depth_in: 입력 깊이 맵 (NaN/0 포함 가능)
        known_mask: 알려진 깊이 영역 마스크
        hole_mask: 채워야 할 홀 영역 마스크

    Returns:
        선형 보간으로 완성된 깊이 맵
    """
    from scipy.interpolate import griddata

    H, W = depth_in.shape
    depth_out = depth_in.copy().astype(np.float32)

    if not hole_mask.any():
        return depth_out

    # 알려진 점들의 좌표와 값
    known_coords = np.column_stack(np.where(known_mask))
    known_values = depth_in[known_mask]

    # 홀 영역의 좌표
    hole_coords = np.column_stack(np.where(hole_mask))

    if len(known_coords) == 0:
        # 알려진 점이 없으면 평균값으로 채움
        mean_depth = np.nanmean(depth_in)
        if np.isnan(mean_depth):
            mean_depth = 1.0
        depth_out[hole_mask] = mean_depth
        return depth_out

    # 선형 보간 수행
    interpolated_values = griddata(
        known_coords,
        known_values,
        hole_coords,
        method='linear',
        fill_value=np.nan
    )

    # NaN 값이 있으면 nearest neighbor로 채움
    nan_mask = np.isnan(interpolated_values)
    if nan_mask.any():
        nearest_values = griddata(
            known_coords,
            known_values,
            hole_coords[nan_mask],
            method='nearest'
        )
        interpolated_values[nan_mask] = nearest_values

    # 결과 적용
    depth_out[hole_mask] = interpolated_values

    return depth_out
