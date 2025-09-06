import numpy as np
from initialization import initial_guess_logpoisson_completion
from refinement import refine_depth_match_normals_gn_completion, GNConfig


def depth_completion(
    depth_in: np.ndarray,  # 입력 심도 (NaN/0 포함 가능)
    refine_roi: np.ndarray,  # ROI (bool/0-1) - 빈 구간 채울 영역
    valid_mask: np.ndarray,  # 유효 심도(True=관측 존재)
    n_guide: np.ndarray | None,  # (H,W,3) 단위 노멀 (선택)
    guide_gray: np.ndarray | None,  # 엣지 가이드 (선택)
    K: np.ndarray | None,  # 카메라 내파라미터
    cfg_gn: GNConfig | None = None,  # GN 설정(선택)
    lambda_screen_init: float = 1.0,  # 초기화 시 known에 대한 스크린 강도
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    깊이 완성 메인 파이프라인: inpaint → initialize → refine

    Args:
        depth_in: 입력 깊이 맵 (NaN/0 포함 가능)
        refine_roi: 채울 영역 마스크 (bool/0-1)
        valid_mask: 유효한 관측 위치 마스크
        n_guide: 가이드 노멀 벡터 (H,W,3) - 선택사항
        guide_gray: 엣지 가이드 그레이스케일 이미지 - 선택사항
        K: 카메라 내부 파라미터 - 선택사항
        cfg_gn: 가우스-뉴턴 정제 설정 - 선택사항
        lambda_screen_init: 초기화 시 known 영역에 대한 스크린 강도
        inpaint_method: inpaint 방법 ('telea' 또는 'ns')

    Returns:
        (inpaint_result, initialize_result, refine_result) 튜플
    """
    depth_in = depth_in.astype(np.float32)
    refine_roi = refine_roi.astype(bool)
    known_mask = (valid_mask.astype(bool) & np.isfinite(depth_in))
    hole_mask = refine_roi & (~known_mask)

    # 2단계: Log-Poisson completion으로 초기화
    depth_initialize = initial_guess_logpoisson_completion(
        depth_in=depth_in,  # inpaint 결과를 입력으로 사용
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

    n_g = n_guide / (np.linalg.norm(n_guide, axis=2, keepdims=True) + 1e-12)

    # 3단계: GN 정제 (가이드 노멀 있는 경우 강력 추천)
    if cfg_gn is not None and n_guide is not None:
        depth_refine = refine_depth_match_normals_gn_completion(
            depth_initialize, known_mask, hole_mask, n_g, K, guide_gray, cfg_gn)
    else:
        depth_refine = depth_initialize

    return depth_initialize, depth_refine
