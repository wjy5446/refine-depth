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

    # 더 안전한 0으로 나누기 방지
    z_safe = np.maximum(np.abs(z), 1e-8)
    nx = -dz_dx * np.float32(fx) / z_safe
    ny = -dz_dy * np.float32(fy) / z_safe
    nz = np.ones_like(z, dtype=np.float32)

    norm = np.sqrt(nx**2 + ny**2 + nz**2).astype(np.float32)
    # 0으로 나누기 방지
    norm = np.maximum(norm, 1e-8)
    nx /= norm
    ny /= norm
    nz /= norm
    return np.stack([nx, ny, nz], axis=-1).astype(np.float32)
