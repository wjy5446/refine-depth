import numpy as np
import time
from initialization import initial_guess_logpoisson_completion


def depth_completion(
    depth_in: np.ndarray,  # 입력 심도 (NaN/0 포함 가능)
    refine_roi: np.ndarray,  # ROI (bool/0-1) - 빈 구간 채울 영역
    valid_mask: np.ndarray,  # 유효 심도(True=관측 존재)
    n_guide: np.ndarray | None,  # (H,W,3) 단위 노멀 (선택)
    guide_gray: np.ndarray | None,  # 엣지 가이드 (선택)
    K: np.ndarray | None,  # 카메라 내파라미터
    lambda_screen_init: float = 1.0,  # 초기화 시 known에 대한 스크린 강도
) -> tuple[np.ndarray, dict]:
    """
    깊이 완성 메인 파이프라인: initialize

    Args:
        depth_in: 입력 깊이 맵 (NaN/0 포함 가능)
        refine_roi: 채울 영역 마스크 (bool/0-1)
        valid_mask: 유효한 관측 위치 마스크
        n_guide: 가이드 노멀 벡터 (H,W,3) - 선택사항
        guide_gray: 엣지 가이드 그레이스케일 이미지 - 선택사항
        K: 카메라 내부 파라미터 - 선택사항
        lambda_screen_init: 초기화 시 known 영역에 대한 스크린 강도

    Returns:
        (initialize_result, timing_info) 튜플
        timing_info: 각 단계별 처리 시간 정보 딕셔너리
    """
    # 시간 측정을 위한 딕셔너리 초기화
    timing_info = {
        'init_time': 0.0,
        'total_time': 0.0
    }

    total_start_time = time.time()
    depth_in = depth_in.astype(np.float32)
    refine_roi = refine_roi.astype(bool)
    known_mask = (valid_mask.astype(bool) & np.isfinite(depth_in))
    hole_mask = refine_roi & (~known_mask)

    # Log-Poisson completion으로 초기화
    init_start_time = time.time()
    depth_initialize = initial_guess_logpoisson_completion(
        depth_in=depth_in,
        known_mask=known_mask,
        hole_mask=hole_mask,
        guide_gray=guide_gray,
        lambda_grad=3.0,
        lambda_smooth=0.3,
        edge_alpha=6.0,
        tol=1e-4,
        maxiter=300,
        clip_min=0.0,
        clip_max=None
    )
    init_end_time = time.time()
    timing_info['init_time'] = init_end_time - init_start_time

    total_end_time = time.time()
    timing_info['total_time'] = total_end_time - total_start_time

    return depth_initialize, timing_info
