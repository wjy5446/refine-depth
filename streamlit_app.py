import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import time
from PIL import Image

from main import depth_completion
from utils import make_intrinsics, compute_normals_from_depth
from test_visualization import create_synthetic_scene_with_holes

# 페이지 설정
st.set_page_config(
    page_title="Depth Completion Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 제목
st.title("🔍 Depth Completion Pipeline")
st.markdown("**Initialize** 1단계 깊이 완성 파이프라인")

# 사이드바 설정
st.sidebar.header("⚙️ 설정")

# 데이터 소스 선택
data_source = st.sidebar.selectbox(
    "데이터 소스 선택",
    ["합성 데이터", "이미지 업로드"],
    help="테스트용 합성 데이터 또는 직접 업로드한 이미지 사용"
)

if data_source == "합성 데이터":
    # 합성 데이터 설정
    st.sidebar.subheader("합성 데이터 설정")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        H = st.number_input("높이", min_value=100, max_value=500, value=180, step=10)
    with col2:
        W = st.number_input("너비", min_value=100, max_value=500, value=240, step=10)

    noise_sigma = st.sidebar.slider("노이즈 강도", 0.0, 1.0, 0.25, 0.05)
    hole_mode = st.sidebar.selectbox("홀 패턴", ["random", "blob", "stripe", "mixed"])
    seed = st.sidebar.number_input("랜덤 시드", 0, 1000, 0)

    # 합성 데이터 생성
    if st.sidebar.button("합성 데이터 생성", type="primary"):
        with st.spinner("합성 데이터 생성 중..."):
            (depth_gt, depth_in, depth_noisy, n_guide, guide_gray, K,
             refine_roi, valid_mask, hole_mask) = create_synthetic_scene_with_holes(
                H, W, seed=seed, noise_sigma=noise_sigma, hole_mode=hole_mode)

            # 세션 상태에 저장
            st.session_state.depth_gt = depth_gt
            st.session_state.depth_in = depth_in
            st.session_state.n_guide = n_guide
            st.session_state.guide_gray = guide_gray
            st.session_state.K = K
            st.session_state.refine_roi = refine_roi
            st.session_state.valid_mask = valid_mask
            st.session_state.hole_mask = hole_mask
            st.session_state.data_loaded = True

        st.success("합성 데이터가 생성되었습니다!")

else:
    # 이미지 업로드
    st.sidebar.subheader("이미지 업로드")
    uploaded_file = st.sidebar.file_uploader(
        "깊이 이미지 업로드",
        type=['png', 'jpg', 'jpeg', 'tiff'],
        help="깊이 맵 이미지를 업로드하세요"
    )

    if uploaded_file is not None:
        # 이미지 로드
        image = Image.open(uploaded_file)
        depth_array = np.array(image.convert('L')).astype(np.float32)

        # 정규화 (0-1 범위로)
        depth_array = depth_array / 255.0

        # 홀 마스크 생성 (임의로)
        hole_mask = np.random.random(depth_array.shape) < 0.3
        depth_in = depth_array.copy()
        depth_in[hole_mask] = 0.0

        # 세션 상태에 저장
        st.session_state.depth_gt = depth_array
        st.session_state.depth_in = depth_in
        st.session_state.hole_mask = hole_mask
        st.session_state.valid_mask = ~hole_mask
        st.session_state.refine_roi = hole_mask

        # 카메라 내부 파라미터 생성
        H, W = depth_array.shape
        st.session_state.K = make_intrinsics(H, W)
        st.session_state.n_guide = compute_normals_from_depth(depth_array, st.session_state.K)
        st.session_state.guide_gray = depth_array
        st.session_state.data_loaded = True

        st.success("이미지가 업로드되었습니다!")

# 메인 컨텐츠
if 'data_loaded' in st.session_state and st.session_state.data_loaded:

    # 파라미터 설정
    st.header("⚙️ 파라미터 설정")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Initialize 설정")
        lambda_grad = st.slider("Gradient 가중치", 0.1, 10.0, 3.0, 0.1)
        lambda_smooth = st.slider("Smooth 가중치", 0.1, 2.0, 0.3, 0.1)
        lambda_screen = st.slider("Screen 가중치", 0.0, 5.0, 1.0, 0.1)
        lambda_normal_edge = st.slider("Normal edge 가중치", 0.0, 5.0, 0.0, 0.1)

    with col2:
        st.subheader("추가 설정")
        st.info("Log-Poisson completion을 사용합니다.")

    # 실행 버튼
    if st.button("🚀 파이프라인 실행", type="primary"):
        # 시간 측정을 위한 컨테이너들
        progress_container = st.container()
        timing_container = st.container()

        with progress_container:
            st.subheader("⏱️ 처리 진행 상황")
            progress_bar = st.progress(0)
            status_text = st.empty()

        with timing_container:
            st.subheader("📊 단계별 처리 시간")
            timing_cols = st.columns(3)

        # 전체 시작 시간
        total_start_time = time.time()

        # Initialize 실행
        status_text.text("Initialize 실행 중...")
        progress_bar.progress(50)

        # 깊이 완성 실행
        depth_initialize, timing_info = depth_completion(
            depth_in=st.session_state.depth_in,
            refine_roi=st.session_state.refine_roi,
            valid_mask=st.session_state.valid_mask,
            n_guide=st.session_state.n_guide,
            guide_gray=st.session_state.guide_gray,
            K=st.session_state.K,
            lambda_normal_edge=lambda_normal_edge,
            lambda_screen_init=lambda_screen
        )

        # 전체 완료
        total_end_time = time.time()
        total_time = total_end_time - total_start_time

        progress_bar.progress(100)
        status_text.text("✅ 모든 단계 완료!")

        # 시간 표시
        with timing_cols[0]:
            init_ratio = timing_info['init_time']/total_time*100
            st.metric("Initialize", f"{timing_info['init_time']:.3f}s", f"{init_ratio:.1f}%")
        with timing_cols[1]:
            st.metric("총 시간", f"{total_time:.3f}s", "완료")
        with timing_cols[2]:
            st.metric("", "", "")

        # 전체 시간 표시
        st.metric("총 처리 시간", f"{total_time:.3f}s", "완료")

        # 시간 분포 차트
        fig, ax = plt.subplots(figsize=(10, 6))
        stages = ['Initialize']
        times = [timing_info['init_time']]
        colors = ['lightgreen']

        bars = ax.bar(stages, times, color=colors, alpha=0.7)
        ax.set_title('단계별 처리 시간 분포', fontsize=14, fontweight='bold')
        ax.set_ylabel('처리 시간 (초)', fontsize=12)
        ax.set_xlabel('처리 단계', fontsize=12)

        # 각 막대 위에 시간 표시
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold')

        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # 세션 상태에 결과 저장
        st.session_state.depth_initialize = depth_initialize
        st.session_state.results_ready = True
        st.session_state.timing_info = timing_info

        st.success("파이프라인 실행이 완료되었습니다!")

    # 결과 표시
    if 'results_ready' in st.session_state and st.session_state.results_ready:
        st.header("📊 결과 시각화")

        # 탭으로 구분
        tab1, tab2, tab3, tab4 = st.tabs(["1단계 파이프라인", "단계별 비교", "성능 메트릭", "처리 시간 분석"])

        with tab1:
            st.subheader("Initialize 파이프라인")

            # 1단계 파이프라인 시각화
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Ground Truth
            im1 = axes[0].imshow(st.session_state.depth_gt, cmap='viridis', vmin=0, vmax=3)
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

            # Input
            im2 = axes[1].imshow(st.session_state.depth_in, cmap='viridis', vmin=0, vmax=3)
            axes[1].set_title('Input (with holes)')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

            # Initialize Result
            im3 = axes[2].imshow(st.session_state.depth_initialize, cmap='viridis', vmin=0, vmax=3)
            axes[2].set_title('Initialize Result')
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

            plt.suptitle('Depth Completion: 1-Stage Pipeline', fontsize=16, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

        with tab2:
            st.subheader("단계별 결과 비교")

            # 2D 뷰 비교
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            stages = [
                (st.session_state.depth_gt, "Ground Truth"),
                (st.session_state.depth_in, "Input"),
                (st.session_state.depth_initialize, "Initialize")
            ]

            for i, (depth, name) in enumerate(stages):
                row = i // 3
                col = i % 3
                ax = axes[row, col]
                im = ax.imshow(depth, cmap='viridis', vmin=0, vmax=3)
                ax.set_title(name)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # 빈 공간들
            axes[1, 1].axis('off')
            axes[1, 2].axis('off')

            plt.tight_layout()
            st.pyplot(fig)

        with tab3:
            st.subheader("성능 메트릭")

            # 홀 영역에서의 메트릭 계산
            gt_hole = st.session_state.depth_gt[st.session_state.hole_mask]
            in_hole = st.session_state.depth_in[st.session_state.hole_mask]
            init_hole = st.session_state.depth_initialize[st.session_state.hole_mask]

            # MAE 계산
            mae_input = float(np.mean(np.abs(gt_hole - in_hole)))
            mae_init = float(np.mean(np.abs(gt_hole - init_hole)))

            # RMSE 계산
            rmse_input = float(np.sqrt(np.mean((gt_hole - in_hole)**2)))
            rmse_init = float(np.sqrt(np.mean((gt_hole - init_hole)**2)))

            # 메트릭 표시
            col1, col2 = st.columns(2)

            with col1:
                st.metric("MAE (Input)", f"{mae_input:.4f}")
                st.metric("MAE (Initialize)", f"{mae_init:.4f}")

            with col2:
                st.metric("RMSE (Input)", f"{rmse_input:.4f}")
                st.metric("RMSE (Initialize)", f"{rmse_init:.4f}")

            # 개선도 그래프
            fig, ax = plt.subplots(figsize=(10, 6))
            stages = ['Input', 'Initialize']
            mae_values = [mae_input, mae_init]
            rmse_values = [rmse_input, rmse_init]

            ax.plot(stages, mae_values, 'o-', label='MAE', linewidth=2, markersize=8)
            ax.plot(stages, rmse_values, 's-', label='RMSE', linewidth=2, markersize=8)
            ax.set_title('Error Progression')
            ax.set_ylabel('Error')
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        with tab4:
            st.subheader("⏱️ 처리 시간 분석")

            if 'timing_info' in st.session_state:
                timing = st.session_state.timing_info

                # 시간 통계
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Initialize", f"{timing['init_time']:.3f}s")
                with col2:
                    st.metric("총 시간", f"{timing['total_time']:.3f}s")
                with col3:
                    st.metric("", "")
                with col4:
                    st.metric("", "")

                # 시간 분포 파이 차트
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # 막대 차트
                stages = ['Initialize']
                times = [timing['init_time']]
                colors = ['lightgreen']

                bars = ax1.bar(stages, times, color=colors, alpha=0.7)
                ax1.set_title('단계별 처리 시간', fontsize=14, fontweight='bold')
                ax1.set_ylabel('처리 시간 (초)', fontsize=12)

                for bar, time_val in zip(bars, times):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold')

                # 파이 차트 (0이 아닌 값들만)
                non_zero_times = [(stage, time_val) for stage, time_val in zip(stages, times) if time_val > 0]
                if non_zero_times:
                    labels, values = zip(*non_zero_times)
                    ax2.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors[:len(values)])
                    ax2.set_title('처리 시간 비율', fontsize=14, fontweight='bold')

                plt.tight_layout()
                st.pyplot(fig)

                # 성능 분석
                st.subheader("📈 성능 분석")

                if timing['total_time'] > 0:
                    init_ratio = timing['init_time'] / timing['total_time'] * 100

                    st.write(f"**Initialize 단계**가 전체 처리 시간의 {init_ratio:.1f}%를 차지합니다.")
                    st.info("💡 Initialize 단계가 주요 연산입니다. Log-Poisson completion이 핵심 처리입니다.")

                # 처리 속도 분석
                st.subheader("🚀 처리 속도 분석")

                # 픽셀당 처리 시간 계산
                total_pixels = st.session_state.depth_in.size
                pixels_per_second = total_pixels / timing['total_time'] if timing['total_time'] > 0 else 0

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("총 픽셀 수", f"{total_pixels:,}")
                    st.metric("픽셀/초", f"{pixels_per_second:,.0f}")
                with col2:
                    st.metric("이미지 크기", f"{st.session_state.depth_in.shape[0]}×{st.session_state.depth_in.shape[1]}")
                    st.metric("초당 이미지", f"{1/timing['total_time']:.2f}" if timing['total_time'] > 0 else "0.00")

        # 결과 다운로드
        st.header("💾 결과 다운로드")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Initialize 결과 다운로드"):
                img_init = (st.session_state.depth_initialize * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_init)

                buf = io.BytesIO()
                pil_img.save(buf, format='PNG')
                buf.seek(0)

                st.download_button(
                    label="Initialize 결과 다운로드",
                    data=buf.getvalue(),
                    file_name="initialize_result.png",
                    mime="image/png"
                )

else:
    # 데이터가 로드되지 않은 경우
    st.info("👈 사이드바에서 데이터를 선택하고 생성하거나 업로드하세요.")

    # 사용법 안내
    st.header("💡 사용법")

    st.markdown("""
    ### 1. 데이터 준비
    - **합성 데이터**: 테스트용으로 자동 생성되는 깊이 맵 사용
    - **이미지 업로드**: 직접 깊이 맵 이미지를 업로드하여 사용

    ### 2. 파라미터 조정
    - **Initialize 설정**: Log-Poisson completion의 가중치 조정

    ### 3. 결과 확인
    - **1단계 파이프라인**: 전체 과정의 시각화
    - **단계별 비교**: 각 단계별 결과 비교
    - **성능 메트릭**: 정량적 성능 평가

    ### 4. 결과 다운로드
    - Initialize 결과를 PNG 이미지로 다운로드 가능
    """)

    # 알고리즘 설명
    st.header("🔬 알고리즘 설명")

    st.markdown("""
    ### 1단계 파이프라인

    1. **Initialize 단계**
       - Log-Poisson completion 사용
       - 그래디언트 정합과 스무딩을 통한 초기화
       - 홀 영역을 정교하게 채우는 깊이 분포 생성
    """)

# 푸터
st.markdown("---")
st.markdown("**Depth Completion Pipeline** - 1단계 깊이 완성 시스템")
