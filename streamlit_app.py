import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

from main import depth_completion
from refinement import GNConfig
from utils import make_intrinsics, compute_normals_from_depth
from test_visualization import create_synthetic_scene_with_holes, visualize_depth_completion_3stage

# 페이지 설정
st.set_page_config(
    page_title="Depth Completion Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 제목
st.title("🔍 Depth Completion Pipeline")
st.markdown("**Inpaint → Initialize → Refine** 3단계 깊이 완성 파이프라인")

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

    with col2:
        st.subheader("Refine 설정")
        lambda_data = st.slider("Data 가중치", 0.0, 2.0, 0.5, 0.1)
        lambda_normal = st.slider("Normal 가중치", 0.0, 10.0, 5.0, 0.5)
        lambda_smooth_refine = st.slider("Refine Smooth 가중치", 0.0, 2.0, 0.2, 0.1)
        gn_iters = st.slider("GN 반복 수", 1, 10, 2)

    # 실행 버튼
    if st.button("🚀 파이프라인 실행", type="primary"):
        with st.spinner("깊이 완성 파이프라인 실행 중..."):

            # Config 객체 생성
            cfg_gn = GNConfig(
                lambda_normal=2.0,
                lambda_smooth=0.3,
                lambda_screen=0.1,
                loss="charbonnier",
                eps_charb=1e-3,
                gn_iters=2,
                normal_stride=1,     # 느리면 2~3
                edge_alpha=6.0,
                step_clip_frac=0.5
            )

            # 깊이 완성 실행
            depth_initialize, depth_refine = depth_completion(
                depth_in=st.session_state.depth_in,
                refine_roi=st.session_state.refine_roi,
                valid_mask=st.session_state.valid_mask,
                n_guide=st.session_state.n_guide,
                guide_gray=st.session_state.guide_gray,
                K=st.session_state.K,
                cfg_gn=cfg_gn,
                lambda_screen_init=lambda_screen
            )

            # 세션 상태에 결과 저장
            st.session_state.depth_initialize = depth_initialize
            st.session_state.depth_refine = depth_refine
            st.session_state.results_ready = True

        st.success("파이프라인 실행이 완료되었습니다!")

    # 결과 표시
    if 'results_ready' in st.session_state and st.session_state.results_ready:
        st.header("📊 결과 시각화")

        # 탭으로 구분
        tab1, tab2, tab3 = st.tabs(["3단계 파이프라인", "단계별 비교", "성능 메트릭"])

        with tab1:
            st.subheader("Inpaint → Initialize → Refine 파이프라인")

            # 3단계 파이프라인 시각화
            fig = visualize_depth_completion_3stage(
                depth_gt=st.session_state.depth_gt,
                depth_in=st.session_state.depth_in,
                depth_initialize=st.session_state.depth_initialize,
                depth_refine=st.session_state.depth_refine,
                hole_mask=st.session_state.hole_mask,
                title="Depth Completion: 3-Stage Pipeline"
            )
            st.pyplot(fig)

        with tab2:
            st.subheader("단계별 결과 비교")

            # 2D 뷰 비교
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            stages = [
                (st.session_state.depth_gt, "Ground Truth"),
                (st.session_state.depth_in, "Input"),
                (st.session_state.depth_initialize, "Initialize"),
                (st.session_state.depth_refine, "Refine")
            ]

            for i, (depth, name) in enumerate(stages):
                row = i // 3
                col = i % 3
                ax = axes[row, col]
                im = ax.imshow(depth, cmap='viridis', vmin=0, vmax=3)
                ax.set_title(name)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # 마지막 빈 공간
            axes[1, 2].axis('off')

            plt.tight_layout()
            st.pyplot(fig)

        with tab3:
            st.subheader("성능 메트릭")

            # 홀 영역에서의 메트릭 계산
            gt_hole = st.session_state.depth_gt[st.session_state.hole_mask]
            in_hole = st.session_state.depth_in[st.session_state.hole_mask]
            init_hole = st.session_state.depth_initialize[st.session_state.hole_mask]
            refine_hole = st.session_state.depth_refine[st.session_state.hole_mask]

            # MAE 계산
            mae_input = float(np.mean(np.abs(gt_hole - in_hole)))
            mae_init = float(np.mean(np.abs(gt_hole - init_hole)))
            mae_refine = float(np.mean(np.abs(gt_hole - refine_hole)))

            # RMSE 계산
            rmse_input = float(np.sqrt(np.mean((gt_hole - in_hole)**2)))
            rmse_init = float(np.sqrt(np.mean((gt_hole - init_hole)**2)))
            rmse_refine = float(np.sqrt(np.mean((gt_hole - refine_hole)**2)))

            # 메트릭 표시
            col1, col2 = st.columns(2)

            with col1:
                st.metric("MAE (Input)", f"{mae_input:.4f}")
                st.metric("MAE (Initialize)", f"{mae_init:.4f}")
                st.metric("MAE (Refine)", f"{mae_refine:.4f}")

            with col2:
                st.metric("RMSE (Input)", f"{rmse_input:.4f}")
                st.metric("RMSE (Initialize)", f"{rmse_init:.4f}")
                st.metric("RMSE (Refine)", f"{rmse_refine:.4f}")

            # 개선도 그래프
            fig, ax = plt.subplots(figsize=(10, 6))
            stages = ['Input', 'Initialize', 'Refine']
            mae_values = [mae_input, mae_init, mae_refine]
            rmse_values = [rmse_input, rmse_init, rmse_refine]

            ax.plot(stages, mae_values, 'o-', label='MAE', linewidth=2, markersize=8)
            ax.plot(stages, rmse_values, 's-', label='RMSE', linewidth=2, markersize=8)
            ax.set_title('Error Progression')
            ax.set_ylabel('Error')
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        # 결과 다운로드
        st.header("💾 결과 다운로드")

        col1, col2, col3 = st.columns(3)

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

        with col2:
            if st.button("Refine 결과 다운로드"):
                img_refine = (st.session_state.depth_refine * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_refine)

                buf = io.BytesIO()
                pil_img.save(buf, format='PNG')
                buf.seek(0)

                st.download_button(
                    label="Refine 결과 다운로드",
                    data=buf.getvalue(),
                    file_name="refine_result.png",
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
    - **Inpaint 설정**: 홀 영역을 채우는 방법 선택
    - **Initialize 설정**: 초기화 단계의 가중치 조정
    - **Refine 설정**: 정제 단계의 가중치 및 반복 수 조정

    ### 3. 결과 확인
    - **3단계 파이프라인**: 전체 과정의 시각화
    - **단계별 비교**: 각 단계별 결과 비교
    - **성능 메트릭**: 정량적 성능 평가

    ### 4. 결과 다운로드
    - 각 단계별 결과를 PNG 이미지로 다운로드 가능
    """)

    # 알고리즘 설명
    st.header("🔬 알고리즘 설명")

    st.markdown("""
    ### 3단계 파이프라인

    1. **Inpaint 단계**
       - Fast Marching Method: 거리 기반 가중 평균
       - Harmonic Inpainting: 반복적 라플라시안 방정식 해결
       - 홀 영역에 초기 값을 채워넣음

    2. **Initialize 단계**
       - Log-Poisson completion 사용
       - 그래디언트 정합과 스무딩을 통한 초기화
       - 더 정교한 깊이 분포 생성

    3. **Refine 단계**
       - 가우스-뉴턴 방법 사용
       - 노멀 벡터 정합을 통한 최종 정제
       - 가장 높은 품질의 결과 생성
    """)

# 푸터
st.markdown("---")
st.markdown("**Depth Completion Pipeline** - 3단계 깊이 완성 시스템")
