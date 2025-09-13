import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from utils import make_intrinsics, compute_normals_from_depth
from main import depth_completion


# 한글 폰트 설정
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
                return True
            except Exception:
                continue

        # 폰트를 찾지 못한 경우 기본 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        return False

    except Exception as e:
        print(f"폰트 설정 중 오류 발생: {e}")
        return False

# 한글 폰트 설정 적용
setup_korean_font()


def create_synthetic_scene_with_holes(H=180, W=240, seed=0,
                                      noise_sigma=0.25,
                                      hole_mode="mixed"):
    """
    홀이 있는 합성 장면 생성 (평면 + 구체)

    Args:
        H: 이미지 높이
        W: 이미지 너비
        seed: 랜덤 시드
        noise_sigma: 노이즈 표준편차
        hole_mode: 홀 생성 모드 ("random", "blob", "stripe", "mixed")

    Returns:
        depth_clean, depth_in, depth_noisy, n_guide, guide_gray, K, \
        refine_roi, valid_mask, hole_mask
    """
    rng = np.random.default_rng(seed)
    K = make_intrinsics(H, W)

    # 배경 평면 (z=3) + 중앙 구체 (반경 ~ min(H,W)//4, z~1.5~2.0)
    depth_bg = np.full((H, W), 3.0, dtype=np.float32)
    yy, xx = np.ogrid[:H, :W]
    cx, cy = W//2, H//2
    radius = int(min(H, W) * 0.25)
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(np.float32)
    sphere = dist < radius
    sphere_depth = 1.5 + 0.5 * np.sqrt(
        np.maximum(0.0, 1.0 - (dist[sphere] / float(radius))**2)
    ).astype(np.float32)

    depth_clean = depth_bg.copy()
    depth_clean[sphere] = sphere_depth

    # 노이즈 추가 (센서 노이즈/양자화 흉내)
    noise = rng.normal(0, noise_sigma, size=(H, W)).astype(np.float32)
    depth_noisy = (depth_clean + noise).astype(np.float32)

    # 가이드 노멀 (GT로부터 계산)
    n_guide = compute_normals_from_depth(depth_clean, K)

    # 가이드 그레이(에지 웨이트용): 심도 대비 정규화
    g = (depth_noisy - depth_noisy.min()) / (
        depth_noisy.max() - depth_noisy.min() + 1e-12
    )
    guide_gray = g.astype(np.float32)

    # ---------------------------
    # 홀(mask) 생성
    # ---------------------------
    hole_mask = np.zeros((H, W), dtype=bool)

    def add_random_holes(area_ratio=0.20, min_patch=5, max_patch=25):
        nonlocal hole_mask
        target = int(H*W*area_ratio)
        covered = 0
        while covered < target:
            h = rng.integers(min_patch, max_patch+1)
            w = rng.integers(min_patch, max_patch+1)
            y = rng.integers(0, max(1, H-h))
            x = rng.integers(0, max(1, W-w))
            hole_mask[y:y+h, x:x+w] = True
            covered = hole_mask.sum()

    def add_blob(center=None, rad=35):
        nonlocal hole_mask
        if center is None:
            center = (rng.integers(H//4, 3*H//4),
                      rng.integers(W//4, 3*W//4))
        y0, x0 = center
        Y, X = np.ogrid[:H, :W]
        hole_mask |= ((Y-y0)**2 + (X-x0)**2) < (rad*rad)

    def add_vertical_stripes(step=16, width=6):
        nonlocal hole_mask
        for x0 in range(0, W, step):
            hole_mask[:, x0:x0+width] = True

    if hole_mode == "random":
        add_random_holes(0.22, 6, 28)
    elif hole_mode == "blob":
        add_blob((H//2, W//2), int(min(H, W)*0.22))
    elif hole_mode == "stripe":
        add_vertical_stripes(step=18, width=8)
    else:  # mixed
        add_random_holes(0.12, 6, 20)
        add_blob((int(H*0.65), int(W*0.4)),
                 int(min(H, W)*0.18))
        add_vertical_stripes(step=28, width=5)

    refine_roi = hole_mask.copy()  # 메꿀 영역
    valid_mask = np.isfinite(depth_noisy.astype(np.float64)) & (depth_noisy > 0)  # 관측 유효
    depth_in = depth_noisy.copy()
    depth_in[hole_mask] = 0.0  # 홀은 0/NaN으로 표시(유효 아님)
    valid_mask[hole_mask] = False

    return (depth_clean, depth_in, depth_noisy, n_guide, guide_gray, K,
            refine_roi, valid_mask, hole_mask)


def visualize_depth_completion(depth_gt, depth_in, depth_out, hole_mask,
                               title="Depth Completion Visualization"):
    """
    깊이 완성 결과 시각화 (3D/2D + 히스토그램/마스크)

    Args:
        depth_gt: Ground truth 깊이 맵
        depth_in: 입력 깊이 맵 (홀 포함)
        depth_out: 완성된 깊이 맵
        hole_mask: 홀 마스크
        title: 그래프 제목
    """
    H, W = depth_gt.shape
    step = 3
    y3d, x3d = np.meshgrid(np.arange(0, H, step), np.arange(0, W, step), indexing='ij')
    gt3d = depth_gt[::step, ::step]
    in3d = depth_in[::step, ::step]
    out3d = depth_out[::step, ::step]
    err3d = out3d - gt3d

    fig = plt.figure(figsize=(24, 16))

    ax1 = fig.add_subplot(3, 4, 1, projection='3d')
    ax1.plot_surface(x3d, y3d, gt3d, cmap='viridis', linewidth=0, antialiased=True)
    ax1.set_title('3D Ground Truth')
    ax1.view_init(elev=20, azim=45)

    ax2 = fig.add_subplot(3, 4, 2, projection='3d')
    ax2.plot_surface(x3d, y3d, in3d, cmap='viridis', linewidth=0, antialiased=True)
    ax2.set_title('3D Input (with holes)')
    ax2.view_init(elev=20, azim=45)

    ax3 = fig.add_subplot(3, 4, 3, projection='3d')
    ax3.plot_surface(x3d, y3d, out3d, cmap='viridis', linewidth=0, antialiased=True)
    ax3.set_title('3D Completed')
    ax3.view_init(elev=20, azim=45)

    ax4 = fig.add_subplot(3, 4, 4, projection='3d')
    ax4.plot_surface(x3d, y3d, err3d, cmap='RdBu_r', linewidth=0, antialiased=True)
    ax4.set_title('3D Error (Completed - GT)')
    ax4.view_init(elev=20, azim=45)

    ax5 = fig.add_subplot(3, 4, 5)
    im5 = ax5.imshow(depth_gt, cmap='viridis')
    ax5.set_title('2D Ground Truth')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(3, 4, 6)
    im6 = ax6.imshow(depth_in, cmap='viridis')
    ax6.set_title('2D Input (holes)')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    ax7 = fig.add_subplot(3, 4, 7)
    im7 = ax7.imshow(depth_out, cmap='viridis')
    ax7.set_title('2D Completed')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    ax8 = fig.add_subplot(3, 4, 8)
    im8 = ax8.imshow(depth_out - depth_gt, cmap='RdBu_r')
    ax8.set_title('2D Error')
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

    # 히스토그램(전체/홀 내부)
    ax9 = fig.add_subplot(3, 4, 9)
    ax10 = fig.add_subplot(3, 4, 10)
    gt_flat = depth_gt.flatten()
    in_flat = depth_in.flatten()
    out_flat = depth_out.flatten()
    ax9.hist(gt_flat, bins=50, alpha=0.7, label='GT', density=True)
    ax9.hist(in_flat, bins=50, alpha=0.7, label='Input', density=True)
    ax9.hist(out_flat, bins=50, alpha=0.7, label='Completed', density=True)
    ax9.set_title('Histogram (Full)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    gt_hole = depth_gt[hole_mask]
    in_hole = depth_in[hole_mask]
    out_hole = depth_out[hole_mask]
    ax10.hist(gt_hole, bins=50, alpha=0.7, label='GT (holes)', density=True)
    ax10.hist(out_hole, bins=50, alpha=0.7, label='Completed (holes)', density=True)
    ax10.set_title('Histogram (Hole region)')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    # 마스크/품질 지표
    ax11 = fig.add_subplot(3, 4, 10)
    ax11.imshow(hole_mask, cmap='gray')
    ax11.set_title('Hole Mask')
    ax11.axis('off')

    ax12 = fig.add_subplot(3, 4, 11)
    ax12.axis('off')
    mae_hole_in = float(np.mean(np.abs(gt_hole - in_hole)))
    mae_hole_out = float(np.mean(np.abs(gt_hole - out_hole)))
    rmse_hole_in = float(np.sqrt(np.mean((gt_hole - in_hole)**2)))
    rmse_hole_out = float(np.sqrt(np.mean((gt_hole - out_hole)**2)))

    stats = (
        f"Hole-region metrics\n"
        f"MAE  Input→GT : {mae_hole_in:.4f}\n"
        f"MAE  Out  →GT : {mae_hole_out:.4f}\n"
        f"RMSE Input→GT : {rmse_hole_in:.4f}\n"
        f"RMSE Out  →GT : {rmse_hole_out:.4f}\n"
        f"MAE gain   : {(mae_hole_in - mae_hole_out):+.4f}\n"
        f"RMSE gain  : {(rmse_hole_in - rmse_hole_out):+.4f}"
    )
    ax12.text(0.05, 0.95, stats, transform=ax12.transAxes,
              fontsize=11, va='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.85))

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def visualize_depth_completion_1stage(depth_gt, depth_in, depth_initialize,
                                      hole_mask,
                                      title="Depth Completion: 1-Stage Pipeline"):
    """
    1단계 깊이 완성 결과 시각화 (initialize)

    Args:
        depth_gt: Ground truth 깊이 맵
        depth_in: 입력 깊이 맵 (holes 포함)
        depth_initialize: initialize 결과
        hole_mask: 홀 영역 마스크
        title: 그래프 제목

    Returns:
        matplotlib Figure 객체
    """
    fig = plt.figure(figsize=(20, 15))

    # 3D 뷰 (GT, Input, Initialize)
    stages = [
        (depth_gt, "GT"),
        (depth_in, "Input"),
        (depth_initialize, "Initialize")
    ]

    for i, (depth, name) in enumerate(stages):
        ax = fig.add_subplot(4, 4, i + 1, projection='3d')
        H, W = depth.shape
        yy, xx = np.mgrid[0:H, 0:W]
        valid = np.isfinite(depth.astype(np.float64)) & (depth > 0)
        if valid.any():
            ax.scatter(xx[valid], yy[valid], depth[valid], c=depth[valid],
                       cmap='viridis', s=1, alpha=0.6)
        ax.set_title(f'{name} (3D)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Depth')

    # 2D 뷰 (GT, Input, Initialize)
    for i, (depth, name) in enumerate(stages):
        ax = fig.add_subplot(4, 4, i + 5)
        im = ax.imshow(depth, cmap='viridis', vmin=0, vmax=3)
        ax.set_title(f'{name} (2D)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Error maps (Input, Initialize vs GT)
    error_maps = [
        (np.abs(depth_gt - depth_in), "Input vs GT"),
        (np.abs(depth_gt - depth_initialize), "Initialize vs GT")
    ]

    for i, (error, name) in enumerate(error_maps):
        ax = fig.add_subplot(4, 4, i + 10)
        im = ax.imshow(error, cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'Error: {name}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hole mask
    ax16 = fig.add_subplot(4, 4, 9)
    ax16.imshow(hole_mask, cmap='gray')
    ax16.set_title('Hole Mask')
    ax16.axis('off')

    # 통계 정보
    ax17 = fig.add_subplot(4, 4, 14)
    ax17.axis('off')

    # 홀 영역에서의 메트릭 계산
    gt_hole = depth_gt[hole_mask]
    in_hole = depth_in[hole_mask]
    init_hole = depth_initialize[hole_mask]

    mae_values = [
        float(np.mean(np.abs(gt_hole - in_hole))),
        float(np.mean(np.abs(gt_hole - init_hole)))
    ]

    rmse_values = [
        float(np.sqrt(np.mean((gt_hole - in_hole)**2))),
        float(np.sqrt(np.mean((gt_hole - init_hole)**2)))
    ]

    stats = (
        f"Hole-region metrics\n"
        f"MAE:\n"
        f"  Input     : {mae_values[0]:.4f}\n"
        f"  Initialize: {mae_values[1]:.4f}\n"
        f"\nRMSE:\n"
        f"  Input     : {rmse_values[0]:.4f}\n"
        f"  Initialize: {rmse_values[1]:.4f}\n"
        f"\nImprovement:\n"
        f"  MAE gain  : {mae_values[0] - mae_values[1]:+.4f}\n"
        f"  RMSE gain : {rmse_values[0] - rmse_values[1]:+.4f}"
    )
    ax17.text(0.05, 0.95, stats, transform=ax17.transAxes,
              fontsize=10, va='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.85))

    # 단계별 개선도
    ax18 = fig.add_subplot(4, 4, 15)
    stages_names = ['Input', 'Initialize']
    ax18.plot(stages_names, mae_values, 'o-', label='MAE', linewidth=2, markersize=8)
    ax18.plot(stages_names, rmse_values, 's-', label='RMSE', linewidth=2, markersize=8)
    ax18.set_title('Error Progression')
    ax18.set_ylabel('Error')
    ax18.legend()
    ax18.grid(True, alpha=0.3)
    ax18.tick_params(axis='x', rotation=45)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def run_example():
    """깊이 완성 예제 실행"""
    H, W = 180, 240
    (depth_gt, depth_in, depth_noisy, n_guide, guide_gray, K,
     refine_roi, valid_mask, hole_mask) = create_synthetic_scene_with_holes(
        H, W, seed=0, noise_sigma=0.25, hole_mode="mixed")

    # 1단계 파이프라인 실행
    print("Running 1-stage pipeline...")
    depth_initialize, timing_info = depth_completion(
        depth_in=depth_in,
        refine_roi=refine_roi,
        valid_mask=valid_mask,
        n_guide=n_guide,
        guide_gray=guide_gray,
        K=K,
        lambda_normal_edge=0.0,
        lambda_screen_init=1.0
    )

    print("Done. Visualizing 1-stage pipeline...")
    fig = visualize_depth_completion_1stage(
        depth_gt=depth_gt,
        depth_in=depth_in,
        depth_initialize=depth_initialize,
        hole_mask=hole_mask,
        title="Depth Completion: 1-Stage Pipeline (Initialize)"
    )
    plt.show()
    return fig


if __name__ == "__main__":
    run_example()
