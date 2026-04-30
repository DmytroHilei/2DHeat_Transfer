import torch
import time

H, W = 1000, 1000
alpha = 0.005
dt = 0.0032
dx = 0.01
r = alpha * dt / dx**2
num_steps = 62500

assert r <= 0.25, f"Unstable: r = {r:.4f} > 0.25"
print(f"Physics OK: r = {r:.4f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T_max, T_min = 100.0, 0.0
R = 100.0
center_y, center_x = (H - 1) / 2.0, (W - 1) / 2.0

y_coords = torch.arange(H, device=device).float()
x_coords = torch.arange(W, device=device).float()
yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

T = torch.where(
    (xx - center_x)**2 + (yy - center_y)**2 <= R**2,
    torch.tensor(T_max, device=device),
    torch.tensor(T_min, device=device)
)

T_new = torch.zeros_like(T)

_ = T + T_new
torch.cuda.synchronize()

start = time.perf_counter()

for t in range(num_steps):
    T_new[1:-1, 1:-1] = (
        T[1:-1, 1:-1] + r * (
            T[:-2, 1:-1]   # top
          + T[2:, 1:-1]    # bottom
          + T[1:-1, :-2]   # left
          + T[1:-1, 2:]    # right
          - 4.0 * T[1:-1, 1:-1]
        )
    )
    T_new[0, :]  = T[0, :]
    T_new[-1, :] = T[-1, :]
    T_new[:, 0]  = T[:, 0]
    T_new[:, -1] = T[:, -1]

    T, T_new = T_new, T

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"PyTorch elapsed: {elapsed * 1000:.2f} ms")
print(f"Device: {device}")