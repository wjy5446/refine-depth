import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from utils import make_intrinsics, compute_normals_from_depth
from main import depth_completion
from config import Config


def create_synthetic_scene_with_holes(H=180, W=240, seed=0,
                                      noise_sigma=0.25,
                                      hole_mode="mixed"):
    """홀이 있는 합성 장면 생성"""
    rng = np.random.default_rng(seed)
    K = make_intrinsics(H, W)

    # 배경 평면 (z=3) + 중앙 구체
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

    # 노이즈 추가
    noise = rng.normal(0, noise_sigma, size=(H, W)).astype(np.float32)
    depth_noisy = (depth_clean + noise).astype(np.float32)

    # 가이드 노멀
    n_guide = compute_normals_from_depth(depth_clean, K)

    # 가이드 그레이
    g = (depth_noisy - depth_noisy.min()) / (
        depth_noisy.max() - depth_noisy.min() + 1e-12
    )
    guide_gray = g.astype(np.float32)

    # 홀 생성
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

    refine_roi = hole_mask.copy()
    valid_mask = np.isfinite(depth_noisy.astype(np.float64)) & (depth_noisy > 0)
    depth_in = depth_noisy.copy()
    depth_in[hole_mask] = 0.0
    valid_mask[hole_mask] = False

    return (depth_clean, depth_in, depth_noisy, n_guide, guide_gray, K,
            refine_roi, valid_mask, hole_mask)


class InteractiveDepthCompletion:
    def __init__(self):
        # 데이터 생성
        H, W = 120, 160  # 더 작은 크기로 빠른 처리
        (self.depth_gt, self.depth_in, self.depth_noisy, self.n_guide,
         self.guide_gray, self.K, self.refine_roi, self.valid_mask,
         self.hole_mask) = create_synthetic_scene_with_holes(
            H, W, seed=0, noise_sigma=0.25, hole_mode="mixed"
        )

        # 초기 설정
        self.cfg_gn = Config(
            lambda_data=0.5,
            lambda_normal=0.2,
            lambda_smooth=0.6,
            edge_alpha=6.0,
            iters=1,  # 빠른 처리를 위해 1회만
            boundary_width=1,
            boundary_boost=2.0,
            boundary_grad_scale=0.6,
            clip_min=0.0,
            clip_max=None
        )

        # 초기 결과 계산
        self.depth_out = self.compute_depth_completion()

        # GUI 설정
        self.setup_gui()

    def compute_depth_completion(self):
        """깊이 완성 계산"""
        return depth_completion(
            depth_in=self.depth_in,
            refine_roi=self.refine_roi,
            valid_mask=self.valid_mask,
            n_guide=self.n_guide,
            guide_gray=self.guide_gray,
            K=self.K,
            lambda_normal_edge=self.cfg_gn.lambda_normal,
            lambda_screen_init=1.0
        )

    def setup_gui(self):
        """GUI 설정"""
        self.fig = plt.figure(figsize=(16, 10))

        # 메인 플롯들
        self.ax_gt = plt.subplot(2, 3, 1)
        self.ax_in = plt.subplot(2, 3, 2)
        self.ax_out = plt.subplot(2, 3, 3)
        self.ax_error = plt.subplot(2, 3, 4)
        self.ax_3d = plt.subplot(2, 3, 5, projection='3d')
        self.ax_stats = plt.subplot(2, 3, 6)

        # 컬러바 초기화
        self.cbar_gt = None
        self.cbar_in = None
        self.cbar_out = None
        self.cbar_error = None

        # 슬라이더 위치 설정
        ax_lambda_data = plt.axes([0.1, 0.02, 0.2, 0.03])
        ax_lambda_normal = plt.axes([0.35, 0.02, 0.2, 0.03])
        ax_lambda_smooth = plt.axes([0.6, 0.02, 0.2, 0.03])
        ax_edge_alpha = plt.axes([0.1, 0.07, 0.2, 0.03])
        ax_iters = plt.axes([0.35, 0.07, 0.2, 0.03])

        # 슬라이더 생성
        self.slider_lambda_data = Slider(
            ax_lambda_data, 'λ_data', 0.0, 2.0, valinit=0.5, valstep=0.1
        )
        self.slider_lambda_normal = Slider(
            ax_lambda_normal, 'λ_normal', 0.0, 10.0, valinit=0.2, valstep=0.1
        )
        self.slider_lambda_smooth = Slider(
            ax_lambda_smooth, 'λ_smooth', 0.0, 2.0, valinit=0.6, valstep=0.1
        )
        self.slider_edge_alpha = Slider(
            ax_edge_alpha, 'edge_α', 1.0, 20.0, valinit=6.0, valstep=0.5
        )
        self.slider_iters = Slider(
            ax_iters, 'iters', 1, 5, valinit=1, valstep=1
        )

        # 슬라이더 이벤트 연결
        self.slider_lambda_data.on_changed(self.update_lambda_data)
        self.slider_lambda_normal.on_changed(self.update_lambda_normal)
        self.slider_lambda_smooth.on_changed(self.update_lambda_smooth)
        self.slider_edge_alpha.on_changed(self.update_edge_alpha)
        self.slider_iters.on_changed(self.update_iters)

        # 업데이트 버튼
        ax_update = plt.axes([0.6, 0.07, 0.15, 0.03])
        self.btn_update = Button(ax_update, 'Update')
        self.btn_update.on_clicked(self.update_all)

        # 초기 플롯
        self.update_plots()

    def update_lambda_data(self, val):
        self.cfg_gn.lambda_data = val
        self.update_all(None)

    def update_lambda_normal(self, val):
        self.cfg_gn.lambda_normal = val
        self.update_all(None)

    def update_lambda_smooth(self, val):
        self.cfg_gn.lambda_smooth = val
        self.update_all(None)

    def update_edge_alpha(self, val):
        self.cfg_gn.edge_alpha = val
        self.update_all(None)

    def update_iters(self, val):
        self.cfg_gn.iters = int(val)
        self.update_all(None)

    def update_all(self, event):
        """모든 파라미터 업데이트"""
        print(f"Updating with λ_data={self.cfg_gn.lambda_data:.1f}, "
              f"λ_normal={self.cfg_gn.lambda_normal:.1f}, "
              f"λ_smooth={self.cfg_gn.lambda_smooth:.1f}, "
              f"edge_α={self.cfg_gn.edge_alpha:.1f}, "
              f"iters={self.cfg_gn.iters}")

        # 깊이 완성 재계산
        self.depth_out = self.compute_depth_completion()

        # 플롯 업데이트
        self.update_plots()

    def update_plots(self):
        """플롯 업데이트"""
        # 모든 축 클리어
        for ax in [self.ax_gt, self.ax_in, self.ax_out, self.ax_error,
                   self.ax_3d, self.ax_stats]:
            ax.clear()

        # 2D 플롯들
        im1 = self.ax_gt.imshow(self.depth_gt, cmap='viridis')
        self.ax_gt.set_title('Ground Truth')
        self.ax_gt.axis('off')
        if self.cbar_gt is not None:
            self.cbar_gt.remove()
        self.cbar_gt = plt.colorbar(im1, ax=self.ax_gt, fraction=0.046, pad=0.04)

        im2 = self.ax_in.imshow(self.depth_in, cmap='viridis')
        self.ax_in.set_title('Input (with holes)')
        self.ax_in.axis('off')
        if self.cbar_in is not None:
            self.cbar_in.remove()
        self.cbar_in = plt.colorbar(im2, ax=self.ax_in, fraction=0.046, pad=0.04)

        im3 = self.ax_out.imshow(self.depth_out, cmap='viridis')
        self.ax_out.set_title('Completed')
        self.ax_out.axis('off')
        if self.cbar_out is not None:
            self.cbar_out.remove()
        self.cbar_out = plt.colorbar(im3, ax=self.ax_out, fraction=0.046, pad=0.04)

        error = self.depth_out - self.depth_gt
        im4 = self.ax_error.imshow(error, cmap='RdBu_r')
        self.ax_error.set_title('Error (Completed - GT)')
        self.ax_error.axis('off')
        if self.cbar_error is not None:
            self.cbar_error.remove()
        self.cbar_error = plt.colorbar(im4, ax=self.ax_error, fraction=0.046, pad=0.04)

        # 3D 플롯 (서브샘플링)
        step = 2
        y3d, x3d = np.meshgrid(
            np.arange(0, self.depth_gt.shape[0], step),
            np.arange(0, self.depth_gt.shape[1], step),
            indexing='ij'
        )
        gt3d = self.depth_gt[::step, ::step]
        out3d = self.depth_out[::step, ::step]

        self.ax_3d.plot_surface(x3d, y3d, gt3d, alpha=0.7, cmap='viridis')
        self.ax_3d.plot_surface(x3d, y3d, out3d, alpha=0.7, cmap='plasma')
        self.ax_3d.set_title('3D Comparison')
        self.ax_3d.view_init(elev=20, azim=45)

        # 통계
        gt_hole = self.depth_gt[self.hole_mask]
        out_hole = self.depth_out[self.hole_mask]
        mae = np.mean(np.abs(gt_hole - out_hole))
        rmse = np.sqrt(np.mean((gt_hole - out_hole)**2))

        self.ax_stats.text(0.1, 0.8, f'MAE: {mae:.4f}',
                          transform=self.ax_stats.transAxes, fontsize=12)
        self.ax_stats.text(0.1, 0.6, f'RMSE: {rmse:.4f}',
                          transform=self.ax_stats.transAxes, fontsize=12)
        self.ax_stats.text(0.1, 0.4, f'λ_data: {self.cfg_gn.lambda_data:.1f}',
                          transform=self.ax_stats.transAxes, fontsize=10)
        self.ax_stats.text(0.1, 0.3, f'λ_normal: {self.cfg_gn.lambda_normal:.1f}',
                          transform=self.ax_stats.transAxes, fontsize=10)
        self.ax_stats.text(0.1, 0.2, f'λ_smooth: {self.cfg_gn.lambda_smooth:.1f}',
                          transform=self.ax_stats.transAxes, fontsize=10)
        self.ax_stats.text(0.1, 0.1, f'edge_α: {self.cfg_gn.edge_alpha:.1f}',
                          transform=self.ax_stats.transAxes, fontsize=10)
        self.ax_stats.set_title('Statistics')
        self.ax_stats.axis('off')

        plt.tight_layout()
        self.fig.canvas.draw()

    def show(self):
        plt.show()


if __name__ == "__main__":
    app = InteractiveDepthCompletion()
    app.show()
