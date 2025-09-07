from dataclasses import dataclass


@dataclass
class Config:
    lambda_data: float = 0.5
    lambda_normal: float = 0.2
    lambda_smooth: float = 0.6
    edge_alpha: float = 6.0
    iters: int = 1
    boundary_width: int = 1
    boundary_boost: float = 2.0
    boundary_grad_scale: float = 0.6
    clip_min: float | None = 0.0
    clip_max: float | None = None
