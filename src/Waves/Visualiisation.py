import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.animation import FuncAnimation
import natsort

# --------------------
# Parameters
# --------------------
N = 1000
p0 = 100000.0          # reference pressure
A = 6000.0             # color scale amplitude (Pa)
time_total = 0.1       # total simulated time (s)

# --------------------
# Load frames
# --------------------
bin_files = natsort.natsorted(
    glob.glob(r"C:\Users\giley\CLionProjects\2D_Wawe_Spread\Bins\frame_*.bin")
)

print(f"Found {len(bin_files)} BIN files")

frames = []

for file in bin_files:
    arr = np.fromfile(file, dtype=np.float32)
    if arr.size != N * N:
        print(f"Warning: {file} has wrong size {arr.size}")
        continue

    grid = arr.reshape((N, N))
    frames.append(grid)

frames = np.array(frames)
num_frames = len(frames)

print(f"Loaded {num_frames} frames")

if num_frames == 0:
    raise RuntimeError("No valid frames loaded")

# Convert to pressure deviation once (important)
frames = frames - p0

dt = time_total / num_frames

# --------------------
# Figure setup
# --------------------
fig, ax = plt.subplots(figsize=(6, 6))

img = ax.imshow(
    frames[0],
    origin="lower",
    cmap="seismic",
    extent=[0, N, 0, N],
    vmin=-A,
    vmax=A,
    interpolation="nearest"
)

cbar = plt.colorbar(img, ax=ax, pad=0.02)
cbar.set_label("Δp [Pa]")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("2D acoustic wave propagation")

ax.set_aspect("equal")
ax.set_xlim(0, N)
ax.set_ylim(0, N)

# --------------------
# Animation update
# --------------------

#time_text = ax.text(0.5, 1.05, '', transform=ax.transAxes,
                   # ha='center', fontsize=12)


def update(frame_index):
    img.set_data(frames[frame_index])
    t = frame_index * dt
    #print(f"Frame {frame_index}: dt={dt}, t={t}, t_ms={t * 1e3}")
    ax.set_title(f"time of the simulation t = {t * 1e3:.6f} ms")
    #time_text.set_text(f"t = {t * 1e3:.6f} ms")
    return (img)

anim = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=45,
    blit=False,
    repeat=True
)

plt.tight_layout()
plt.show()

