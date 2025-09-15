"""
깊이 완성 파이프라인 Streamlit 앱
Log-Poisson completion + Normal Alignment refinement을 사용한 깊이 완성 시스템
"""

import streamlit as st
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

from main import depth_completion
from refine import refine_depth_normal_alignment
from initialization import initial_guess_logpoisson_completion
from utils import compute_normals_from_depth, make_intrinsics
from test_visualization import create_synthetic_scene_with_holes
from ui_components import (
    setup_korean_font, render_parameter_sidebar,
    render_matplotlib_visualization, render_plotly_3d_visualization,
    render_performance_metrics,
    render_depth_completion_parameter_sidebar,
    show_3d_point_cloud, show_3d_surface_mesh, show_3d_height_map
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
st.title("🔍 깊이 완성 & 정련 실험")
st.markdown("""
**Log-Poisson completion + Normal Alignment refinement을 사용한 고품질 깊이 완성 시스템**

이 앱은 깊이 맵의 홀(구멍) 영역을 자동으로 복원하는 AI 기반 파이프라인입니다.
- **1단계**: Log-Poisson completion으로 초기화
- **2단계**: Normal Alignment refinement로 정련

합성 데이터를 생성하거나 실제 깊이 맵을 업로드하여 실시간으로 파라미터를 조정하며 실험할 수 있습니다.
""")

# 파라미터 설정
params = render_parameter_sidebar()
depth_completion_params = render_depth_completion_parameter_sidebar()

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
                valid_mask = np.isfinite(depth_data.astype(np.float64)) & (depth_data > 0)
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

# 실험 모드 선택
if 'data_loaded' in st.session_state and st.session_state.data_loaded:

    # 실험 모드 선택
    st.subheader("🧪 실험 모드 선택")
    experiment_mode = st.radio(
        "실험 모드를 선택하세요:",
        ["전체 파이프라인", "개별 단계 실험", "파라미터 비교"],
        horizontal=True
    )

    if experiment_mode == "전체 파이프라인":
        # 전체 파이프라인 실행
        if st.button("🔧 Depth Completion 파이프라인 실행", type="primary", use_container_width=True):
            # 진행 상황 표시
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 전체 시작 시간
            total_start_time = time.time()

            # Depth Completion 파이프라인 실행
            status_text.text("🔧 Depth Completion 파이프라인 실행 중...")
            progress_bar.progress(50)

            start_time = time.time()
            depth_initialize, depth_refined, discontinue_maps, timing_info = depth_completion(
                depth_in=st.session_state.depth_in,
                refine_roi=st.session_state.refine_roi,
                valid_mask=st.session_state.valid_mask,
                n_guide=st.session_state.normals_gt,
                guide_gray=st.session_state.guide_gray,
                K=st.session_state.K,
                # 초기화 파라미터
                lambda_init_grad=depth_completion_params['lambda_init_grad'],
                lambda_init_smooth=depth_completion_params['lambda_init_smooth'],
                lambda_init_normal_edge=depth_completion_params['lambda_init_normal_edge'],
                # 정련 파라미터 (새로운 refine.py 기반)
                lambda_refine_normal=depth_completion_params['lambda_refine_normal'],
                lambda_refine_smooth=depth_completion_params['lambda_refine_smooth'],
                lambda_refine_equal=depth_completion_params['lambda_refine_equal'],
                lambda_refine_plane=depth_completion_params['lambda_refine_plane'],
                lambda_refine_screen=depth_completion_params['lambda_refine_screen'],
                # 공통 파라미터
                edge_alpha=depth_completion_params['edge_alpha'],
                solver=depth_completion_params['solver'],
                tol=depth_completion_params['tol'],
                maxiter=depth_completion_params['maxiter'],
                clip_min=depth_completion_params['clip_min'],
                clip_max=depth_completion_params['clip_max']
            )
            end_time = time.time()

            # 전체 완료
            total_end_time = time.time()
            total_time = total_end_time - total_start_time

            progress_bar.progress(100)
            status_text.text("✅ 모든 단계 완료!")

            # 세션 상태에 결과 저장
            st.session_state.depth_initialize = depth_initialize
            st.session_state.depth_refined = depth_refined
            st.session_state.discontinue_maps = discontinue_maps
            st.session_state.results_ready = True
            st.session_state.timing_info = timing_info

            st.success(f"🎉 Depth Completion 파이프라인 실행이 완료되었습니다! (소요시간: {total_time:.2f}초)")

            # 타이밍 정보 표시
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("초기화 시간", f"{timing_info['init_time']:.2f}초")
            with col2:
                st.metric("정련 시간", f"{timing_info['refine_time']:.2f}초")
            with col3:
                st.metric("총 시간", f"{timing_info['total_time']:.2f}초")

    elif experiment_mode == "개별 단계 실험":
        st.subheader("🔬 개별 단계 실험")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**1단계: Log-Poisson 초기화**")
            if st.button("초기화 실행", type="secondary"):
                with st.spinner("초기화 실행 중..."):
                    start_time = time.time()
                    depth_initialize = initial_guess_logpoisson_completion(
                        depth_in=st.session_state.depth_in,
                        known_mask=st.session_state.valid_mask,
                        hole_mask=st.session_state.hole_mask,
                        guide_gray=st.session_state.guide_gray,
                        n_guide=st.session_state.normals_gt,
                        lambda_grad=depth_completion_params['lambda_init_grad'],
                        lambda_smooth=depth_completion_params['lambda_init_smooth'],
                        edge_alpha=depth_completion_params['edge_alpha'],
                        lambda_normal_edge=depth_completion_params['lambda_init_normal_edge'],
                        tol=depth_completion_params['tol'],
                        maxiter=depth_completion_params['maxiter'],
                        clip_min=depth_completion_params['clip_min'],
                        clip_max=depth_completion_params['clip_max']
                    )
                    init_time = time.time() - start_time

                    st.session_state.depth_initialize = depth_initialize
                    st.session_state.init_time = init_time
                    st.success(f"✅ 초기화 완료! (소요시간: {init_time:.2f}초)")

        with col2:
            st.write("**2단계: Normal Alignment 정련**")
            if st.button("정련 실행", type="secondary"):
                if 'depth_initialize' not in st.session_state:
                    st.warning("⚠️ 먼저 초기화를 실행해주세요!")
                else:
                    with st.spinner("정련 실행 중..."):
                        start_time = time.time()
                        depth_refined = refine_depth_normal_alignment(
                            depth_in=st.session_state.depth_initialize,
                            known_mask=st.session_state.valid_mask,
                            hole_mask=st.session_state.hole_mask,
                            n_guide=st.session_state.normals_gt,
                            K=st.session_state.K,
                            discontinuity_maps=None,  # 자동 감지
                            lambda_normal=depth_completion_params['lambda_refine_normal'],
                            lambda_smooth=depth_completion_params['lambda_refine_smooth'],
                            lambda_equal=depth_completion_params.get('lambda_refine_equal', 1.0),
                            lambda_plane=depth_completion_params.get('lambda_refine_plane', 1.0),
                            lambda_screen=depth_completion_params['lambda_refine_screen'],
                            lambda_keep=depth_completion_params.get('lambda_refine_keep', 30.0),
                            tol=depth_completion_params['tol'],
                            maxiter=depth_completion_params['maxiter'],
                            solver=depth_completion_params['solver']
                        )
                        refine_time = time.time() - start_time

                        st.session_state.depth_refined = depth_refined
                        st.session_state.refine_time = refine_time
                        st.success(f"✅ 정련 완료! (소요시간: {refine_time:.2f}초)")

        # 개별 단계 결과 표시
        if 'depth_initialize' in st.session_state:
            st.subheader("초기화 결과")
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            im = ax.imshow(st.session_state.depth_initialize, cmap='viridis')
            ax.set_title("Log-Poisson 초기화 결과")
            ax.axis('off')
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)

        if 'depth_refined' in st.session_state:
            st.subheader("정련 결과")
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            im = ax.imshow(st.session_state.depth_refined, cmap='viridis')
            ax.set_title("Normal Alignment 정련 결과")
            ax.axis('off')
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)

    elif experiment_mode == "파라미터 비교":
        st.subheader("📊 파라미터 비교 실험")

        # 비교할 파라미터 선택
        param_to_compare = st.selectbox(
            "비교할 파라미터 선택:",
            ["λ_init_grad", "λ_init_smooth", "λ_refine_normal", "λ_refine_smooth", "edge_alpha"]
        )

        # 파라미터 값 범위 설정
        if param_to_compare == "λ_init_grad":
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
        elif param_to_compare == "λ_init_smooth":
            values = [0.1, 0.2, 0.3, 0.4, 0.5]
        elif param_to_compare == "λ_refine_normal":
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
        elif param_to_compare == "λ_refine_smooth":
            values = [0.1, 0.2, 0.3, 0.4, 0.5]
        else:  # edge_alpha
            values = [3.0, 6.0, 9.0, 12.0, 15.0]

        if st.button("파라미터 비교 실행", type="primary"):
            with st.spinner("파라미터 비교 실행 중..."):
                results = []

                for i, value in enumerate(values):
                    progress = st.progress((i + 1) / len(values))
                    st.write(f"실행 중: {param_to_compare} = {value}")

                    # 파라미터 설정
                    test_params = depth_completion_params.copy()
                    test_params[param_to_compare] = value

                    # 깊이 완성 실행
                    start_time = time.time()
                    depth_initialize, depth_refined, discontinue_maps, timing_info = depth_completion(
                        depth_in=st.session_state.depth_in,
                        refine_roi=st.session_state.refine_roi,
                        valid_mask=st.session_state.valid_mask,
                        n_guide=st.session_state.normals_gt,
                        guide_gray=st.session_state.guide_gray,
                        K=st.session_state.K,
                        lambda_init_grad=test_params['lambda_init_grad'],
                        lambda_init_smooth=test_params['lambda_init_smooth'],
                        lambda_init_normal_edge=test_params['lambda_init_normal_edge'],
                        lambda_refine_normal=test_params['lambda_refine_normal'],
                        lambda_refine_smooth=test_params['lambda_refine_smooth'],
                        lambda_refine_equal=test_params['lambda_refine_equal'],
                        lambda_refine_plane=test_params['lambda_refine_plane'],
                        lambda_refine_screen=test_params['lambda_refine_screen'],
                        edge_alpha=test_params['edge_alpha'],
                        solver=test_params['solver'],
                        tol=test_params['tol'],
                        maxiter=test_params['maxiter'],
                        clip_min=test_params['clip_min'],
                        clip_max=test_params['clip_max']
                    )

                    # 성능 메트릭 계산
                    hole_mask = st.session_state.hole_mask
                    if 'depth_gt' in st.session_state:
                        gt = st.session_state.depth_gt
                        mae = np.mean(np.abs(depth_refined[hole_mask] - gt[hole_mask]))
                        # 수정된 코드 - 더 안전한 유효성 검사
                        valid_mask = (np.isfinite(depth_refined) &
                                    np.isfinite(gt) &
                                    hole_mask &
                                    (depth_refined > 0) &
                                    (gt > 0))
                        if np.sum(valid_mask) == 0:
                            rmse = 0
                        else:
                            pred_valid = depth_refined[valid_mask]
                            gt_valid = gt[valid_mask]
                            rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
                    else:
                        mae = rmse = 0

                    results.append({
                        'param_value': value,
                        'mae': mae,
                        'rmse': rmse,
                        'time': timing_info['total_time']
                    })

                st.session_state.param_comparison_results = results
                st.success("✅ 파라미터 비교 완료!")

        # 파라미터 비교 결과 표시
        if 'param_comparison_results' in st.session_state:
            results = st.session_state.param_comparison_results

            # 결과 테이블
            st.subheader("비교 결과")
            import pandas as pd
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            # 시각화
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # MAE 그래프
            axes[0].plot([r['param_value'] for r in results], [r['mae'] for r in results], 'o-')
            axes[0].set_xlabel(param_to_compare)
            axes[0].set_ylabel('MAE')
            axes[0].set_title(f'{param_to_compare} vs MAE')
            axes[0].grid(True)

            # RMSE 그래프
            axes[1].plot([r['param_value'] for r in results], [r['rmse'] for r in results], 'o-', color='orange')
            axes[1].set_xlabel(param_to_compare)
            axes[1].set_ylabel('RMSE')
            axes[1].set_title(f'{param_to_compare} vs RMSE')
            axes[1].grid(True)

            plt.tight_layout()
            st.pyplot(fig)

    # 결과 표시 (전체 파이프라인 결과)
    if 'results_ready' in st.session_state and st.session_state.results_ready:

        # 탭으로 구분
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 단계별 비교", "📈 성능 메트릭", "🌐 3D 시각화", "🔍 불연속성 분석"])

        with tab1:
            st.subheader("단계별 결과 비교")

            # 시각화 방식 선택
            viz_type = st.radio(
                "시각화 방식 선택:",
                ["Plotly 3D (인터랙티브)", "Matplotlib (정적)"],
                horizontal=True
            )

            if viz_type == "Matplotlib (정적)":
                render_matplotlib_visualization(
                    st.session_state.depth_gt,
                    st.session_state.depth_in,
                    st.session_state.depth_initialize,
                    st.session_state.depth_refined
                )
            else:  # Plotly 3D 인터랙티브
                render_plotly_3d_visualization(
                    st.session_state.depth_gt,
                    st.session_state.depth_in,
                    st.session_state.depth_initialize,
                    st.session_state.depth_refined,
                    st.session_state.discontinue_maps
                )

        with tab2:
            render_performance_metrics(
                st.session_state.depth_gt,
                st.session_state.depth_in,
                st.session_state.depth_initialize,
                st.session_state.depth_refined,
                st.session_state.hole_mask
            )

        with tab3:
            st.subheader("3D 시각화")

            # 3D 시각화 옵션
            viz_option = st.selectbox(
                "3D 시각화 옵션:",
                ["Point Cloud", "Surface Mesh", "Height Map"]
            )

            if viz_option == "Point Cloud":
                show_3d_point_cloud(st.session_state.depth_refined)
            elif viz_option == "Surface Mesh":
                show_3d_surface_mesh(st.session_state.depth_refined)
            else:  # Height Map
                show_3d_height_map(st.session_state.depth_refined)

        with tab4:
            st.subheader("불연속성 분석")

            if 'discontinue_maps' in st.session_state:
                discontinue_maps = st.session_state.discontinue_maps

                # 불연속성 맵 시각화
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**불연속성 맵 (Discontinuity Map)**")
                    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                    im = ax.imshow(discontinue_maps, cmap='hot', interpolation='nearest')
                    ax.set_title("감지된 경계/불연속성")
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, label='불연속성 강도')
                    st.pyplot(fig)

                with col2:
                    st.write("**통계 정보**")
                    total_pixels = discontinue_maps.size
                    discontinuity_pixels = np.sum(discontinue_maps)
                    discontinuity_ratio = discontinuity_pixels / total_pixels * 100

                    st.metric("총 픽셀 수", f"{total_pixels:,}")
                    st.metric("불연속성 픽셀 수", f"{discontinuity_pixels:,}")
                    st.metric("불연속성 비율", f"{discontinuity_ratio:.2f}%")

                # 불연속성 맵과 깊이 맵 오버레이
                st.write("**깊이 맵과 불연속성 오버레이**")
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                # 정련된 깊이 맵
                im1 = axes[0].imshow(st.session_state.depth_refined, cmap='viridis')
                axes[0].set_title("정련된 깊이 맵")
                axes[0].axis('off')
                plt.colorbar(im1, ax=axes[0], label='깊이')

                # 오버레이
                axes[1].imshow(st.session_state.depth_refined, cmap='viridis', alpha=0.7)
                axes[1].imshow(discontinue_maps, cmap='Reds', alpha=0.5, interpolation='nearest')
                axes[1].set_title("정련된 깊이 맵 + 불연속성 경계")
                axes[1].axis('off')

                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("불연속성 맵이 아직 생성되지 않았습니다. 먼저 파이프라인을 실행해주세요.")

    # 개별 단계 결과도 표시
    elif 'depth_initialize' in st.session_state or 'depth_refined' in st.session_state:
        st.subheader("📊 현재 결과")

        col1, col2 = st.columns(2)

        with col1:
            if 'depth_initialize' in st.session_state:
                st.write("**초기화 결과**")
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                im = ax.imshow(st.session_state.depth_initialize, cmap='viridis')
                ax.set_title("Log-Poisson 초기화")
                ax.axis('off')
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)

        with col2:
            if 'depth_refined' in st.session_state:
                st.write("**정련 결과**")
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                im = ax.imshow(st.session_state.depth_refined, cmap='viridis')
                ax.set_title("Normal Alignment 정련")
                ax.axis('off')
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)

        # 3D 시각화
        if 'depth_refined' in st.session_state:
            st.subheader("3D 시각화")

            # 3D 시각화 옵션
            viz_option = st.selectbox(
                "3D 시각화 옵션:",
                ["Point Cloud", "Surface Mesh", "Height Map"]
            )

            if viz_option == "Point Cloud":
                show_3d_point_cloud(st.session_state.depth_refined)
            elif viz_option == "Surface Mesh":
                show_3d_surface_mesh(st.session_state.depth_refined)
            else:  # Height Map
                show_3d_height_map(st.session_state.depth_refined)

else:
    st.info("📁 데이터를 먼저 로드해주세요.")

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    <p>🔍 깊이 완성 & 정련 실험 | Log-Poisson completion + Normal Alignment refinement</p>
    </div>
    """,
    unsafe_allow_html=True
)
