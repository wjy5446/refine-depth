"""
깊이 완성 파이프라인 Streamlit 앱
Log-Poisson completion을 사용한 깊이 완성 시스템
"""

import streamlit as st
import numpy as np
import time
from PIL import Image

from main import depth_completion
from utils import compute_normals_from_depth, make_intrinsics
from test_visualization import create_synthetic_scene_with_holes
from ui_components import (
    setup_korean_font, render_parameter_sidebar,
    render_matplotlib_visualization, render_plotly_3d_visualization,
    render_performance_metrics, render_download_buttons,
    render_initialize_parameter_sidebar  # render_interactive_initialize_parameters 제거
)

# 앱 시작 시 한글 폰트 설정
setup_korean_font()

# 페이지 설정
st.set_page_config(
    page_title="Depth Completion Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 메인 타이틀과 설명
st.title("🔍 깊이 완성 파이프라인")
st.markdown("""
**Log-Poisson completion을 사용한 고품질 깊이 완성 시스템**

이 앱은 깊이 맵의 홀(구멍) 영역을 자동으로 복원하는 AI 기반 파이프라인입니다.
합성 데이터를 생성하거나 실제 깊이 맵을 업로드하여 테스트할 수 있습니다.
""")


# 파라미터 설정
params = render_parameter_sidebar()
initialize_params = render_initialize_parameter_sidebar()
algorithm_params = {'lambda_normal_edge': 0.0}  # 기본값으로 설정

# 데이터 타입에 따른 처리
if params['data_type'] == "합성 데이터":
    # 합성 데이터 생성
    with st.spinner("합성 데이터를 생성하는 중..."):
        (depth_clean, depth_in, depth_noisy, n_guide, guide_gray, K,
         refine_roi, valid_mask, hole_mask) = create_synthetic_scene_with_holes(
            H=params['height'],
            W=params['width'],
            seed=params['seed'],
            noise_sigma=params['noise_sigma'],
            hole_mode=params['hole_mode']
        )

    # 세션 상태에 저장
    st.session_state.depth_gt = depth_clean  # 깨끗한 깊이 맵 (Ground Truth)
    st.session_state.depth_in = depth_in     # 홀이 있는 입력 깊이 맵
    st.session_state.depth_noisy = depth_noisy  # 노이즈가 있는 깊이 맵
    st.session_state.hole_mask = hole_mask
    st.session_state.normals_gt = n_guide    # 가이드 노멀
    st.session_state.guide_gray = guide_gray  # 가이드 그레이
    st.session_state.K = K                   # 카메라 내부 파라미터
    st.session_state.refine_roi = refine_roi  # 정제 영역
    st.session_state.valid_mask = valid_mask  # 유효 마스크
    st.session_state.data_loaded = True

    st.success(f"✅ 합성 데이터가 생성되었습니다! (크기: {params['height']}×{params['width']})")

else:
    # 실제 데이터 처리
    if params['uploaded_file'] is not None:
        with st.spinner("데이터를 로드하는 중..."):
            try:
                if params['uploaded_file'].name.endswith('.npy'):
                    depth_data = np.load(params['uploaded_file'])
                else:
                    # 이미지 파일을 깊이 맵으로 변환
                    image = Image.open(params['uploaded_file'])
                    depth_data = np.array(image.convert('L')).astype(np.float32) / 255.0 * 3.0

                # 홀 마스크 생성 (임시로 랜덤하게)
                hole_mask = np.random.random(depth_data.shape) < 0.3

                # 카메라 내부 파라미터 생성
                H, W = depth_data.shape
                K = make_intrinsics(H, W)

                # 필요한 파라미터들 생성
                valid_mask = np.isfinite(depth_data) & (depth_data > 0)
                valid_mask[hole_mask] = False

                depth_in = depth_data.copy()
                depth_in[hole_mask] = 0.0

                # 가이드 노멀과 그레이스케일 생성
                n_guide = compute_normals_from_depth(depth_data, K)
                guide_gray = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min() + 1e-12)

                st.session_state.depth_gt = depth_data
                st.session_state.depth_in = depth_in
                st.session_state.depth_noisy = depth_data  # 실제 데이터는 노이즈가 있는 것으로 간주
                st.session_state.hole_mask = hole_mask
                st.session_state.normals_gt = n_guide
                st.session_state.guide_gray = guide_gray
                st.session_state.K = K
                st.session_state.refine_roi = hole_mask
                st.session_state.valid_mask = valid_mask
                st.session_state.data_loaded = True

                st.success("✅ 데이터가 성공적으로 로드되었습니다!")

            except Exception as e:
                st.error(f"❌ 데이터 로드 중 오류가 발생했습니다: {str(e)}")
    else:
        st.info("📤 데이터를 업로드하거나 합성 데이터를 선택하세요.")

# 파이프라인 실행
if 'data_loaded' in st.session_state and st.session_state.data_loaded:

    # 실행 버튼
    if st.button("🔧 파이프라인 실행", type="primary", use_container_width=True):
        # 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 전체 시작 시간
        total_start_time = time.time()

        # 1단계: Initialize
        status_text.text("🔄 1단계: Initialize 실행 중...")
        progress_bar.progress(50)

        start_time = time.time()
        depth_initialize, timing_info = depth_completion(
            st.session_state.depth_in,
            st.session_state.refine_roi,
            st.session_state.valid_mask,
            st.session_state.normals_gt,
            st.session_state.guide_gray,
            st.session_state.K,
            lambda_normal_edge=initialize_params['lambda_normal_edge'],
            lambda_screen_init=1.0,  # 기본값
            lambda_grad=initialize_params['lambda_grad'],
            lambda_smooth=initialize_params['lambda_smooth'],
            edge_alpha=initialize_params['edge_alpha'],
            tol=initialize_params['tol'],
            maxiter=initialize_params['maxiter'],
            clip_min=initialize_params['clip_min'],
            clip_max=initialize_params['clip_max']
        )
        end_time = time.time()

        # 전체 완료
        total_end_time = time.time()
        total_time = total_end_time - total_start_time

        progress_bar.progress(100)
        status_text.text("✅ 모든 단계 완료!")

        # 세션 상태에 결과 저장
        st.session_state.depth_initialize = depth_initialize
        st.session_state.results_ready = True
        st.session_state.timing_info = timing_info

        st.success(f"🎉 파이프라인 실행이 완료되었습니다! (소요시간: {total_time:.2f}초)")

    # 결과 표시
    if 'results_ready' in st.session_state and st.session_state.results_ready:

        # 탭으로 구분
        tab1, tab2 = st.tabs(["🔍 단계별 비교", "📈 성능 메트릭"])

        with tab1:
            st.subheader("단계별 결과 비교")

            # 시각화 방식 선택
            viz_type = st.radio(
                "시각화 방식 선택:",
                ["Matplotlib (정적)", "Plotly 3D (인터랙티브)"],
                horizontal=True
            )

            if viz_type == "Matplotlib (정적)":
                render_matplotlib_visualization(
                    st.session_state.depth_gt,
                    st.session_state.depth_in,
                    st.session_state.depth_initialize
                )
            else:  # Plotly 3D 인터랙티브
                render_plotly_3d_visualization(
                    st.session_state.depth_gt,
                    st.session_state.depth_in,
                    st.session_state.depth_initialize
                )

        with tab2:
            render_performance_metrics(
                st.session_state.depth_gt,
                st.session_state.depth_in,
                st.session_state.depth_initialize,
                st.session_state.hole_mask
            )

        # 다운로드 버튼
        render_download_buttons(
            st.session_state.depth_gt,
            st.session_state.depth_in,
            st.session_state.depth_initialize
        )

# 실시간 파라미터 조절 UI 제거
# if st.session_state.data_loaded:
#     render_interactive_initialize_parameters(...)

else:
    st.info("📁 데이터를 먼저 로드해주세요.")

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    <p>🔍 깊이 완성 파이프라인 | Log-Poisson completion 기반</p>
    </div>
    """,
    unsafe_allow_html=True
)
