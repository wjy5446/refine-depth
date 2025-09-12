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
            noise_sigma = st.slider("노이즈 강도", 0.0, .1, 0.01, 0.01)
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


def render_depth_completion_parameter_sidebar():
    """Depth Completion 파이프라인 파라미터 설정 사이드바를 렌더링합니다."""
    st.sidebar.header("⚙️ Depth Completion 파라미터")

    # 초기화 단계 파라미터
    st.sidebar.subheader("🔄 초기화 단계 파라미터")
    lambda_init_grad = st.sidebar.slider(
        "λ_init_grad (초기화 그래디언트 가중치)",
        0.1, 10.0, 3.0, 0.1,
        help="초기화 단계에서 그래디언트 일관성에 대한 가중치"
    )

    lambda_init_smooth = st.sidebar.slider(
        "λ_init_smooth (초기화 스무딩 가중치)",
        0.01, 2.0, 0.2, 0.01,
        help="초기화 단계에서 스무딩에 대한 가중치"
    )

    lambda_init_normal_edge = st.sidebar.slider(
        "λ_init_normal_edge (초기화 노멀 엣지 가중치)",
        0.0, 5.0, 0.0, 0.1,
        help="초기화 단계에서 노멀 벡터 기반 엣지 가중치"
    )

    # 정련 단계 파라미터 (새로운 refine.py 기반)
    st.sidebar.subheader("🔧 정련 단계 파라미터")
    lambda_refine_normal = st.sidebar.slider(
        "λ_refine_normal (법선 정합 가중치)",
        0.1, 10.0, 3.0, 0.1,
        help="정련 단계에서 법선 정합에 대한 가중치"
    )

    lambda_refine_smooth = st.sidebar.slider(
        "λ_refine_smooth (스무딩 가중치)",
        0.0, 2.0, 0.2, 0.01,
        help="정련 단계에서 스무딩에 대한 가중치"
    )

    lambda_refine_equal = st.sidebar.slider(
        "λ_refine_equal (equal 가중치)",
        0.0, 5.0, 1.0, 0.1,
        help="정련 단계에서 equal constraint에 대한 가중치"
    )

    lambda_refine_plane = st.sidebar.slider(
        "λ_refine_plane (plane 가중치)",
        0.0, 5.0, 1.0, 0.1,
        help="정련 단계에서 plane constraint에 대한 가중치"
    )

    lambda_refine_screen = st.sidebar.slider(
        "λ_refine_screen (스크린 앵커 가중치)",
        0., 1e-2, 1e-3, 1e-5,
        format="%.0e",
        help="정련 단계에서 스크린 앵커에 대한 가중치"
    )

    lambda_refine_keep = st.sidebar.slider(
        "λ_refine_keep (알려진 값 유지 가중치)",
        0.0, 100.0, 30.0, 1.0,
        help="정련 단계에서 알려진 값 유지에 대한 가중치"
    )

    # 공통 파라미터
    st.sidebar.subheader("⚙️ 공통 파라미터")
    edge_alpha = st.sidebar.slider(
        "Edge Alpha (엣지 강도)",
        1.0, 20.0, 8.0, 0.5,
        help="엣지 보존 강도. 높을수록 엣지를 더 잘 보존"
    )

    solver = st.sidebar.selectbox(
        "솔버 선택",
        ["lsmr", "cg"],
        help="선형 시스템 솔버 선택"
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
        # 초기화 파라미터
        'lambda_init_grad': lambda_init_grad,
        'lambda_init_smooth': lambda_init_smooth,
        'lambda_init_normal_edge': lambda_init_normal_edge,
        # 정련 파라미터 (새로운 refine.py 기반)
        'lambda_refine_normal': lambda_refine_normal,
        'lambda_refine_smooth': lambda_refine_smooth,
        'lambda_refine_equal': lambda_refine_equal,
        'lambda_refine_plane': lambda_refine_plane,
        'lambda_refine_screen': lambda_refine_screen,
        'lambda_refine_keep': lambda_refine_keep,
        # 공통 파라미터
        'edge_alpha': edge_alpha,
        'solver': solver,
        'tol': tol,
        'maxiter': maxiter,
        'clip_min': clip_min,
        'clip_max': clip_max
    }


def render_matplotlib_visualization(depth_gt, depth_in, depth_initialize, depth_refined):
    """Matplotlib을 사용한 정적 시각화를 렌더링합니다."""
    # 2D 뷰 비교
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    stages = [
        (depth_gt, "Ground Truth"),
        (depth_in, "Input"),
        (depth_initialize, "Initialize"),
        (depth_refined, "Refined"),
        (np.abs(depth_initialize - depth_gt), "|Initialize - GT|"),
        (np.abs(depth_refined - depth_gt), "|Refined - GT|")
    ]

    for i, (depth, name) in enumerate(stages):
        row, col = i // 3, i % 3
        ax = axes[row, col]

        if "|" in name:  # 차이 맵
            im = ax.imshow(depth, cmap='hot', vmin=0, vmax=np.percentile(depth, 95))
        else:  # 일반 깊이 맵
            im = ax.imshow(depth, cmap='viridis', vmin=0, vmax=3)

        ax.set_title(name)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_plotly_3d_visualization(depth_gt, depth_in, depth_initialize, depth_refined):
    """Plotly를 사용한 3D 인터랙티브 시각화를 렌더링합니다."""
    from refine import detect_discontinuities

    H, W = depth_gt.shape

    # 서브플롯 생성 (1행 4열)
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=("Ground Truth", "Input", "Initialize", "Refined"),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'},
               {'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )

    # depth_initialize에서 불연속 맵 계산 (한 번만)
    disc_map_initialize = detect_discontinuities(depth_initialize, tau_rel=0.05)

    # 각 단계별 데이터 준비
    surfaces = [
        (depth_gt, "Ground Truth"),
        (depth_in, "Input"),
        (depth_initialize, "Initialize"),
        (depth_refined, "Refined")
    ]

    for i, (depth, name) in enumerate(surfaces):
        try:
            # 샘플링된 좌표 생성
            step = max(1, min(H, W) // 50)  # 최대 50x50 포인트로 샘플링
            y_indices = np.arange(0, H, step)
            x_indices = np.arange(0, W, step)

            # 메시그리드 생성
            X, Y = np.meshgrid(x_indices, y_indices)
            Z = depth[::step, ::step]

            # 유효한 포인트만 선택 (NaN이나 0이 아닌 값들)
            valid_mask = np.isfinite(Z) & (Z > 0)

            # 유효한 포인트가 있는지 확인
            if np.sum(valid_mask) == 0:
                # 유효한 포인트가 없으면 빈 trace 추가
                fig.add_trace(
                    go.Scatter3d(
                        x=[], y=[], z=[],
                        mode='markers',
                        name=name,
                        showlegend=False
                    ),
                    row=1, col=i+1
                )
                continue

            # 1D 배열로 변환
            x_flat = X[valid_mask].flatten()
            y_flat = Y[valid_mask].flatten()
            z_flat = Z[valid_mask].flatten()

            # depth_refined에만 불연속 영역 정보 적용
            if i == 3:  # depth_refined (4번째, 인덱스 3)
                # depth_initialize에서 계산된 불연속 맵 사용
                disc_sampled = disc_map_initialize[::step, ::step]
                disc_flat = disc_sampled[valid_mask].flatten()

                # 불연속 영역과 연속 영역을 분리
                disc_mask = disc_flat.astype(bool)
                cont_mask = ~disc_mask

                # 연속 영역 (깊이값에 따른 색상)
                if np.any(cont_mask):
                    # 연속 영역용 호버 텍스트 생성
                    cont_hover_texts = []
                    for j in range(np.sum(cont_mask)):
                        idx = np.where(cont_mask)[0][j]
                        x_coord = int(x_flat[idx])
                        y_coord = int(y_flat[idx])
                        z_val = z_flat[idx]
                        is_discontinuous = disc_flat[idx]

                        hover_text = (
                            f"<b>{name}</b><br>"
                            f"좌표: ({x_coord}, {y_coord})<br>"
                            f"깊이값: {z_val:.3f}<br>"
                        )
                        cont_hover_texts.append(hover_text)

                    fig.add_trace(
                        go.Scatter3d(
                            x=x_flat[cont_mask],
                            y=y_flat[cont_mask],
                            z=z_flat[cont_mask],
                            mode='markers',
                            marker=dict(
                                size=2,
                                color=z_flat[cont_mask],
                                colorscale='viridis',
                                opacity=0.8,
                                showscale=True,
                                colorbar=dict(title="Depth")
                            ),
                            name=f"{name} (연속)",
                            showlegend=False,
                            customdata=cont_hover_texts,
                            hovertemplate="%{customdata}<extra></extra>"
                        ),
                        row=1, col=i+1
                    )

                # 불연속 영역 (빨간색)
                if np.any(disc_mask):
                    # 불연속 영역용 호버 텍스트 생성
                    disc_hover_texts = []
                    for j in range(np.sum(disc_mask)):
                        idx = np.where(disc_mask)[0][j]
                        x_coord = int(x_flat[idx])
                        y_coord = int(y_flat[idx])
                        z_val = z_flat[idx]

                        hover_text = (
                            f"<b>{name}</b><br>"
                            f"좌표: ({x_coord}, {y_coord})<br>"
                            f"깊이값: {z_val:.3f}<br>"
                        )
                        disc_hover_texts.append(hover_text)

                    fig.add_trace(
                        go.Scatter3d(
                            x=x_flat[disc_mask],
                            y=y_flat[disc_mask],
                            z=z_flat[disc_mask],
                            mode='markers',
                            marker=dict(
                                size=2,  # 불연속 영역은 조금 더 크게
                                color='red',
                                opacity=0.9
                            ),
                            name=f"{name} (불연속)",
                            showlegend=False,
                            customdata=disc_hover_texts,
                            hovertemplate="%{customdata}<extra></extra>"
                        ),
                        row=1, col=i+1
                    )
            else:
                # 나머지 단계들은 기본 시각화
                fig.add_trace(
                    go.Scatter3d(
                        x=x_flat,
                        y=y_flat,
                        z=z_flat,
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=z_flat,
                            colorscale='viridis',
                            opacity=0.8,
                            showscale=(i == 0),  # 첫 번째만 컬러바 표시
                            colorbar=dict(title="Depth") if i == 0 else None
                        ),
                        name=name,
                        showlegend=False
                    ),
                    row=1, col=i+1
                )

        except Exception as e:
            # 오류 발생 시 빈 trace 추가
            print(f"Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
            fig.add_trace(
                go.Scatter3d(
                    x=[], y=[], z=[],
                    mode='markers',
                    name=name,
                    showlegend=False
                ),
                row=1, col=i+1
            )

    # 레이아웃 업데이트
    fig.update_layout(
        title="3D Point Cloud 깊이 맵 비교 (Refined에서 불연속 영역 강조, 호버로 상세 정보 확인)",
        height=600,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def render_performance_metrics(depth_gt, depth_in, depth_initialize, depth_refined, hole_mask):
    """성능 메트릭을 렌더링합니다."""
    st.subheader("📈 성능 메트릭")

    # 메트릭 계산
    def calculate_metrics(pred, gt, mask):
        valid_mask = np.isfinite(pred) & np.isfinite(gt) & mask
        if np.sum(valid_mask) == 0:
            return 0, 0, 0

        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]

        mae = np.mean(np.abs(pred_valid - gt_valid))
        rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))

        # 상관계수
        corr = np.corrcoef(pred_valid, gt_valid)[0, 1] if len(pred_valid) > 1 else 0

        return mae, rmse, corr

    # 각 단계별 메트릭 계산
    init_mae, init_rmse, init_corr = calculate_metrics(depth_initialize, depth_gt, hole_mask)
    refine_mae, refine_rmse, refine_corr = calculate_metrics(depth_refined, depth_gt, hole_mask)

    # 메트릭 표시
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Initialize MAE", f"{init_mae:.4f}")
        st.metric("Refined MAE", f"{refine_mae:.4f}")
        st.metric("개선도", f"{((init_mae - refine_mae) / init_mae * 100):.1f}%" if init_mae > 0 else "N/A")

    with col2:
        st.metric("Initialize RMSE", f"{init_rmse:.4f}")
        st.metric("Refined RMSE", f"{refine_rmse:.4f}")
        st.metric("개선도", f"{((init_rmse - refine_rmse) / init_rmse * 100):.1f}%" if init_rmse > 0 else "N/A")

    with col3:
        st.metric("Initialize 상관계수", f"{init_corr:.4f}")
        st.metric("Refined 상관계수", f"{refine_corr:.4f}")
        st.metric("개선도", f"{((refine_corr - init_corr) / abs(init_corr) * 100):.1f}%" if abs(init_corr) > 0 else "N/A")


def show_3d_point_cloud(depth_map):
    """3D 포인트 클라우드 시각화"""
    H, W = depth_map.shape
    y, x = np.mgrid[0:H, 0:W]

    # NaN 값 처리
    valid_mask = ~np.isnan(depth_map)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    z_valid = depth_map[valid_mask]

    # 샘플링 (너무 많은 점이면)
    if len(x_valid) > 5000:
        indices = np.random.choice(len(x_valid), 5000, replace=False)
        x_valid = x_valid[indices]
        y_valid = y_valid[indices]
        z_valid = z_valid[indices]

    fig = go.Figure(data=[go.Scatter3d(
        x=x_valid,
        y=y_valid,
        z=z_valid,
        mode='markers',
        marker=dict(
            size=2,
            color=z_valid,
            colorscale='viridis',
            opacity=0.8
        )
    )])

    fig.update_layout(
        title="3D Point Cloud",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Depth"
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def show_3d_surface_mesh(depth_map):
    """3D 서피스 메시 시각화"""
    H, W = depth_map.shape
    y, x = np.mgrid[0:H, 0:W]

    # NaN 값을 0으로 대체
    z = np.nan_to_num(depth_map, nan=0)

    fig = go.Figure(data=[go.Surface(
        x=x,
        y=y,
        z=z,
        colorscale='viridis',
        opacity=0.8
    )])

    fig.update_layout(
        title="3D Surface Mesh",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Depth"
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def show_3d_height_map(depth_map):
    """3D 높이 맵 시각화"""
    H, W = depth_map.shape

    # NaN 값을 0으로 대체
    z = np.nan_to_num(depth_map, nan=0)

    fig = go.Figure(data=[go.Heatmap(
        z=z,
        colorscale='viridis',
        showscale=True
    )])

    fig.update_layout(
        title="3D Height Map",
        xaxis_title="X",
        yaxis_title="Y"
    )

    st.plotly_chart(fig, use_container_width=True)
