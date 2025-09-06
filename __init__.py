"""
Normal Depth Refinement Package

깊이 완성을 위한 패키지로, 다음과 같은 모듈들을 포함합니다:

- utils: 공통 유틸리티 함수들
- config: 설정 클래스
- initialization: 초기화 함수들 (Log-Poisson completion)
- refinement: 정제 함수들 (Gauss-Newton refinement)
- main: 메인 파이프라인
- test_visualization: 테스트 및 시각화 함수들
"""

from .config import Config
from .main import depth_completion
from .utils import make_intrinsics, compute_normals_from_depth
from .test_visualization import create_synthetic_scene_with_holes, visualize_depth_completion, run_example

__version__ = "1.0.0"
__author__ = "Depth Completion Team"

__all__ = [
    "Config",
    "depth_completion",
    "make_intrinsics",
    "compute_normals_from_depth",
    "create_synthetic_scene_with_holes",
    "visualize_depth_completion",
    "run_example"
]
