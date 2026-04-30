import torch
import numpy as np
import time

H, W = 1000, 1000
c = 1.0
dt = 0.001
dx = 0.01
coefficient = c**2 * dt**2 / dx**2
num_steps = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

center_y, center_x = (H - 1) / 2.0, (W - 1) / 2.0
sigma = 20.0

y_coords = torch.arange(H, device=device).float()
x_coords = torch.arange(W, device=device).float()
yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

init = torch.exp(-((xx - center_x)**2 + (yy - center_y)**2) / (2 * sigma**2))

p_old = init.clone()
p     = init.clone()
p_new = torch.zeros_like(init)

_ = p + p_old

torch.cuda.synchronize()
start = time.perf_counter()

for t in range(num_steps):
    p_new[1:-1, 1:-1] = (
        2.0 * p[1:-1, 1:-1]
        - p_old[1:-1, 1:-1]
        + coefficient * (
            p[:-2, 1:-1]   # top
          + p[2:, 1:-1]    # bottom
          + p[1:-1, :-2]   # left
          + p[1:-1, 2:]    # right
          - 4.0 * p[1:-1, 1:-1]
        )
    )

    p_old, p, p_new = p, p_new, p_old

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"PyTorch elapsed: {elapsed * 1000:.2f} ms")
print(f"Device: {device}")