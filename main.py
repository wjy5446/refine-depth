import numpy as np
import time
from initialization import initial_guess_logpoisson_completion
from refine_2stage import refine_depth_normal_alignment, refine_mask_equal_plane_from_depth
from refine import detect_discontinuities


def depth_completion(
    depth_in: np.ndarray,  # 입력 심도 (NaN/0 포함 가능)
    refine_roi: np.ndarray,  # ROI (bool/0-1) - 빈 구간 채울 영역
    valid_mask: np.ndarray,  # 유효 심도(True=관측 존재)
    n_guide: np.ndarray | None = None,  # (H,W,3) 단위 노멀 (선택)
    guide_gray: np.ndarray | None = None,  # 엣지 가이드 (선택)
    K: np.ndarray | None = None,  # 카메라 내파라미터
    # 초기화 단계 파라미터
    lambda_init_grad: float = 3.0,  # 초기화 그래디언트 가중치
    lambda_init_smooth: float = 0.2,  # 초기화 스무딩 가중치
    lambda_init_normal_edge: float = 0.0,  # 초기화 노멀 `엣지 가중치
    # 정련 단계 파라미터
    lambda_refine_normal: float = 3.0,  # 정련 공면 쌍항 가중치
    lambda_refine_smooth: float = 0.2,  # 정련 스무딩 가중치
    lambda_refine_equal: float = 1.0,  # 정련 equal 가중치
    lambda_refine_plane: float = 1.0,  # 정련 plane 가중치
    lambda_refine_screen: float = 1e-3,  # 정련 스크린 앵커 가중치
    lambda_refine_keep: float | None = None,  # 정련 초기화 anchor 가중치
    # 공통 파라미터
    edge_alpha: float = 6.0,  # 엣지 강도
    tol: float = 1e-4,  # 수렴 기준
    maxiter: int = 300,  # 최대 반복수
    clip_min: float = 0.0,  # 최소값 클리핑
    clip_max: float | None = None,  # 최대값 클리핑
    solver: str = "lsmr",  # 솔버 ("lsmr" 또는 "cg")
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    깊이 완성 메인 파이프라인: 초기화 + 정련

    Args:
        depth_in: 입력 깊이 맵 (NaN/0 포함 가능)
        refine_roi: 채울 영역 마스크 (bool/0-1)
        valid_mask: 유효한 관측 위치 마스크
        n_guide: 가이드 노멀 벡터 (H,W,3) - 선택사항
        guide_gray: 엣지 가이드 그레이스케일 이미지 - 선택사항
        K: 카메라 내부 파라미터 - 선택사항
        lambda_init_grad: 초기화 단계 그래디언트 가중치
        lambda_init_smooth: 초기화 단계 스무딩 가중치
        lambda_init_normal_edge: 초기화 단계 노멀 엣지 가중치
        lambda_refine_normal: 정련 단계 공면 쌍항 가중치
        lambda_refine_smooth: 정련 단계 스무딩 가중치
        lambda_refine_data: 정련 단계 데이터 정합 가중치
        lambda_refine_equal: 정련 단계 equal 가중치
        lambda_refine_plane: 정련 단계 plane 가중치
        lambda_refine_screen: 정련 단계 스크린 앵커 가중치
        lambda_refine_keep: 정련 단계 초기화 anchor 가중치
        edge_alpha: 엣지 보존 강도
        tol: 수렴 판정 기준
        maxiter: 최대 반복 횟수
        clip_min: 깊이 값의 최소값 제한
        clip_max: 깊이 값의 최대값 제한
        solver: 선형 솔버 ("lsmr" 또는 "cg")

    Returns:
        (depth_initialize, depth_refined, discontinue_maps, timing_info) 튜플
        depth_initialize: 초기화된 깊이 맵
        depth_refined: 정련된 깊이 맵
        discontinue_maps: 불연속성 맵 (경계 감지 결과)
        timing_info: 각 단계별 처리 시간 정보 딕셔너리
    """
    # 시간 측정을 위한 딕셔너리 초기화
    timing_info = {
        'init_time': 0.0,
        'refine_time': 0.0,
        'total_time': 0.0
    }

    total_start_time = time.time()
    depth_in = depth_in.astype(np.float32)
    known_mask = (valid_mask.astype(bool) & np.isfinite(depth_in.astype(np.float64)))
    hole_mask = ~known_mask

    print(lambda_init_grad, lambda_init_smooth, lambda_init_normal_edge)
    print(lambda_refine_normal, lambda_refine_smooth, lambda_refine_equal, lambda_refine_plane, lambda_refine_screen)
    print(edge_alpha, tol, maxiter, clip_min, clip_max, solver)

    # Log-Poisson completion으로 초기화
    init_start_time = time.time()
    depth_initialize = initial_guess_logpoisson_completion(
        depth_in=depth_in,
        known_mask=known_mask,
        hole_mask=hole_mask,
        guide_gray=guide_gray,
        n_guide=n_guide,
        lambda_grad=lambda_init_grad,
        lambda_smooth=lambda_init_smooth,
        edge_alpha=edge_alpha,
        lambda_normal_edge=lambda_init_normal_edge,
        tol=tol,
        maxiter=maxiter,
        clip_min=clip_min,
        clip_max=clip_max
    )
    init_end_time = time.time()
    timing_info['init_time'] = init_end_time - init_start_time

    discontinue_maps = detect_discontinuities(
        depth_in=depth_initialize,
        n=n_guide,
        use_normals=True,
        tau_n_cos=0.99,
        normal_logic="or"
    )

    # 정련 단계
    refine_start_time = time.time()
    depth_refined, refine_next = refine_depth_normal_alignment(
        depth_in=depth_initialize,
        known_mask=known_mask,
        hole_mask=hole_mask,
        n_guide=n_guide,            # 없으면 (P)(PB)는 자동 생략
        K=K,
        discontinuity_maps=discontinue_maps,
        lambda_normal=lambda_refine_normal,
        lambda_screen=lambda_refine_screen,
        lambda_keep=lambda_refine_keep if lambda_refine_keep is not None else 30.0,
    )

    depth_refined = refine_mask_equal_plane_from_depth(
        depth_in=depth_refined,
        target_mask=refine_next,
        K=K,
        lambda_equal=lambda_refine_equal,
        lambda_plane=lambda_refine_plane,
    )
    refine_end_time = time.time()
    timing_info['refine_time'] = refine_end_time - refine_start_time

    total_end_time = time.time()
    timing_info['total_time'] = total_end_time - total_start_time
    return depth_initialize, depth_refined, refine_next, timing_info
