"""
Streamlit UI 공통 컴포넌트 모듈
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io


def setup_korean_font():
    """한글 폰트를 설정합니다."""
    try:
        # Windows에서 사용 가능한 한글 폰트들
        korean_fonts = [
            'Malgun Gothic',  # 맑은 고딕
            'NanumGothic',    # 나눔고딕
            'Batang',         # 바탕
            'Gulim',          # 굴림
            'Dotum',          # 돋움
            'Arial Unicode MS'
        ]

        for font_name in korean_fonts:
            try:
                plt.rcParams['font.family'] = font_name
                plt.rcParams['axes.unicode_minus'] = False
                # 테스트용 텍스트로 폰트 확인
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, '한글', fontsize=12)
                plt.close(fig)
                print(f"한글 폰트 설정 완료: {font_name}")
                return True
            except Exception:
                continue

        # 폰트를 찾지 못한 경우 기본 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        print("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
        return False

    except Exception as e:
        print(f"폰트 설정 중 오류 발생: {e}")
        return False


def render_parameter_sidebar():
    """파라미터 설정 사이드바를 렌더링합니다."""
    st.sidebar.header("⚙️ 파라미터 설정")

    # 데이터 설정
    st.sidebar.subheader("📊 데이터 설정")
    data_type = st.sidebar.selectbox(
        "데이터 타입 선택:",
        ["합성 데이터", "실제 데이터"],
        help="합성 데이터는 테스트용으로 자동 생성됩니다."
    )

    if data_type == "합성 데이터":
        # 합성 데이터 파라미터
        col1, col2 = st.sidebar.columns(2)
        with col1:
            height = st.slider("높이", 100, 300, 180)
        with col2:
            width = st.slider("너비", 100, 400, 240)

        col1, col2 = st.sidebar.columns(2)
        with col1:
            noise_sigma = st.slider("노이즈 강도", 0.0, 1.0, 0.25, 0.05)
        with col2:
            hole_mode = st.selectbox(
                "홀 모드",
                ["random", "blob", "stripe", "mixed"],
                index=3
            )

        seed = st.sidebar.slider("랜덤 시드", 0, 100, 0)

        return {
            'data_type': data_type,
            'height': height,
            'width': width,
            'noise_sigma': noise_sigma,
            'hole_mode': hole_mode,
            'seed': seed
        }
    else:
        # 실제 데이터 업로드
        uploaded_file = st.sidebar.file_uploader(
            "깊이 맵 파일 업로드",
            type=['npy', 'png', 'jpg', 'jpeg'],
            help="NumPy 배열(.npy) 또는 이미지 파일을 업로드하세요."
        )

        return {
            'data_type': data_type,
            'uploaded_file': uploaded_file
        }


def render_initialize_parameter_sidebar():
    """Initialize 알고리즘 파라미터 설정 사이드바를 렌더링합니다."""
    st.sidebar.header("⚙️ Initialize 파라미터")

    # 기본 파라미터들
    st.sidebar.subheader("📊 기본 파라미터")
    lambda_grad = st.sidebar.slider(
        "λ_grad (그래디언트 가중치)",
        0.1, 10.0, 3.0, 0.1,
        help="그래디언트 일관성에 대한 가중치. 높을수록 더 부드러운 결과"
    )

    lambda_smooth = st.sidebar.slider(
        "λ_smooth (스무딩 가중치)",
        0.01, 2.0, 0.3, 0.01,
        help="스무딩에 대한 가중치. 높을수록 더 부드러운 결과"
    )

    edge_alpha = st.sidebar.slider(
        "Edge Alpha (엣지 강도)",
        1.0, 20.0, 6.0, 0.5,
        help="엣지 보존 강도. 높을수록 엣지를 더 잘 보존"
    )

    # 고급 파라미터들
    st.sidebar.subheader("⚙️ 고급 파라미터")
    lambda_normal_edge = st.sidebar.slider(
        "λ_normal_edge (노멀 엣지 가중치)",
        0.0, 5.0, 0.0, 0.1,
        help="노멀 벡터 기반 엣지 가중치. 0이면 비활성화"
    )

    tol = st.sidebar.number_input(
        "Tolerance (수렴 기준)",
        1e-6, 1e-2, 1e-4, 1e-6,
        format="%.0e",
        help="수렴 판정 기준. 작을수록 더 정확하지만 느림"
    )

    maxiter = st.sidebar.slider(
        "Max Iterations (최대 반복수)",
        50, 1000, 300, 10,
        help="최대 반복 횟수. 높을수록 더 정확하지만 느림"
    )

    # 클리핑 파라미터
    st.sidebar.subheader("📏 클리핑 파라미터")
    clip_min = st.sidebar.number_input(
        "최소값 클리핑",
        0.0, 10.0, 0.0, 0.1,
        help="깊이 값의 최소값 제한"
    )

    clip_max_enabled = st.sidebar.checkbox("최대값 클리핑 활성화", value=False)
    clip_max = None
    if clip_max_enabled:
        clip_max = st.sidebar.number_input(
            "최대값 클리핑",
            1.0, 50.0, 10.0, 0.1,
            help="깊이 값의 최대값 제한"
        )

    return {
        'lambda_grad': lambda_grad,
        'lambda_smooth': lambda_smooth,
        'edge_alpha': edge_alpha,
        'lambda_normal_edge': lambda_normal_edge,
        'tol': tol,
        'maxiter': maxiter,
        'clip_min': clip_min,
        'clip_max': clip_max
    }


def render_matplotlib_visualization(depth_gt, depth_in, depth_initialize):
    """Matplotlib을 사용한 정적 시각화를 렌더링합니다."""
    # 2D 뷰 비교
    fig, axes = plt.subplots(1, 4, figsize=(15, 10))

    stages = [
        (depth_gt, "Ground Truth"),
        (depth_in, "Input"),
        (depth_initialize, "Initialize")
    ]

    for i, (depth, name) in enumerate(stages):
        ax = axes[i]
        im = ax.imshow(depth, cmap='viridis', vmin=0, vmax=3)
        ax.set_title(name)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Initialize와 Ground Truth의 차이
    diff_map = np.abs(depth_initialize - depth_gt)
    ax = axes[3]
    im = ax.imshow(diff_map, cmap='hot', vmin=0, vmax=np.percentile(diff_map, 95))
    ax.set_title("|Initialize - GT|")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    st.pyplot(fig)


def render_plotly_3d_visualization(depth_gt, depth_in, depth_initialize):
    """Plotly를 사용한 3D 인터랙티브 시각화를 렌더링합니다."""
    st.subheader("3D 깊이 맵 시각화")

    # 3D 시각화 옵션
    col1, col2 = st.columns(2)
    with col1:
        point_size = st.slider("포인트 크기", 1, 10, 3)
    with col2:
        sample_rate = st.slider("샘플링 비율", 1, 10, 3)

    # 데이터 준비
    stages_data = [
        (depth_gt, "Ground Truth", "red"),
        (depth_in, "Input", "blue"),
        (depth_initialize, "Initialize", "green")
    ]

    # 차이 맵 계산
    diff_map = np.abs(depth_initialize - depth_gt)

    # 3D 서브플롯 생성
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=("Ground Truth", "Input", "Initialize", "|Init - GT|"),
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]]
    )

    for i, (depth, name, color) in enumerate(stages_data):
        # 샘플링으로 포인트 수 줄이기
        H, W = depth.shape
        step = sample_rate
        y_indices, x_indices = np.meshgrid(
            np.arange(0, H, step),
            np.arange(0, W, step),
            indexing='ij'
        )

        # 샘플링된 데이터
        sampled_depth = depth[::step, ::step]
        sampled_x = x_indices.flatten()
        sampled_y = y_indices.flatten()
        sampled_z = sampled_depth.flatten()

        # 유효한 깊이 값만 선택 (0이 아닌 값)
        valid_mask = sampled_z > 0
        x_valid = sampled_x[valid_mask]
        y_valid = sampled_y[valid_mask]
        z_valid = sampled_z[valid_mask]

        fig.add_trace(
            go.Scatter3d(
                x=x_valid,
                y=y_valid,
                z=z_valid,
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=z_valid,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title="Depth") if i == 0 else None
                ),
                name=name,
                text=[f"Depth: {z:.3f}" for z in z_valid],
                hovertemplate=f"{name}<br>" +
                             "X: %{x}<br>" +
                             "Y: %{y}<br>" +
                             "Depth: %{z:.3f}<br>" +
                             "<extra></extra>"
            ),
            row=1, col=i+1
        )

    # 차이 맵 3D 시각화 (4번째 서브플롯)
    H, W = diff_map.shape
    step = sample_rate
    y_indices, x_indices = np.meshgrid(
        np.arange(0, H, step),
        np.arange(0, W, step),
        indexing='ij'
    )

    # 샘플링된 차이 데이터
    sampled_diff = diff_map[::step, ::step]
    sampled_x = x_indices.flatten()
    sampled_y = y_indices.flatten()
    sampled_z = sampled_diff.flatten()

    fig.add_trace(
        go.Scatter3d(
            x=sampled_x,
            y=sampled_y,
            z=sampled_z,
            mode='markers',
            marker=dict(
                size=point_size,
                color=z_valid,
                colorscale='Hot',
                opacity=0.8,
                colorbar=dict(title="Error")
            ),
            name="|Init - GT|",
            text=[f"Error: {z:.3f}" for z in z_valid],
            hovertemplate="|Init - GT|<br>" +
                         "X: %{x}<br>" +
                         "Y: %{y}<br>" +
                         "Error: %{z:.3f}<br>" +
                         "<extra></extra>"
        ),
        row=1, col=4
    )

    # 3D 레이아웃 설정
    fig.update_layout(
        title="Depth Completion Results - Interactive 3D View",
        height=600,
        showlegend=False
    )

    # 각 서브플롯의 축 설정
    for i in range(1, 5):
        if i == 4:  # 차이 맵의 경우
            fig.update_scenes(
                xaxis_title="Width",
                yaxis_title="Height",
                zaxis_title="Error",
                row=1, col=i
            )
        else:  # 깊이 맵의 경우
            fig.update_scenes(
                xaxis_title="Width",
                yaxis_title="Height",
                zaxis_title="Depth",
                row=1, col=i
            )

    st.plotly_chart(fig, use_container_width=True)


def render_performance_metrics(depth_gt, depth_in, depth_initialize, hole_mask):
    """성능 메트릭을 렌더링합니다."""
    st.subheader("성능 메트릭")

    # 홀 영역에서의 메트릭 계산
    gt_hole = depth_gt[hole_mask]
    in_hole = depth_in[hole_mask]
    init_hole = depth_initialize[hole_mask]

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


def render_download_buttons(depth_gt, depth_in, depth_initialize):
    """다운로드 버튼들을 렌더링합니다."""
    st.subheader("📥 결과 다운로드")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Ground Truth 다운로드
        buffer = io.BytesIO()
        np.save(buffer, depth_gt)
        buffer.seek(0)
        st.download_button(
            label="Ground Truth 다운로드",
            data=buffer.getvalue(),
            file_name="depth_gt.npy",
            mime="application/octet-stream"
        )

    with col2:
        # Input 다운로드
        buffer = io.BytesIO()
        np.save(buffer, depth_in)
        buffer.seek(0)
        st.download_button(
            label="Input 다운로드",
            data=buffer.getvalue(),
            file_name="depth_input.npy",
            mime="application/octet-stream"
        )

    with col3:
        # Initialize 다운로드
        buffer = io.BytesIO()
        np.save(buffer, depth_initialize)
        buffer.seek(0)
        st.download_button(
            label="Initialize 다운로드",
            data=buffer.getvalue(),
            file_name="depth_initialize.npy",
            mime="application/octet-stream"
        )
