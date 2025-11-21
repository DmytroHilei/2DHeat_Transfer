import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.animation import FuncAnimation
import natsort

N = 1000
bin_files = natsort.natsorted(glob.glob(r"C:\Users\giley\2DHeatTransferBins\frame_*.bin"))


print(f"Found {len(bin_files)} BIN files")

frames = []

for file in bin_files:
    try:
        arr = np.fromfile(file, dtype=np.float32)   # real = float => float32
        if arr.size != N*N:
            print(f"Warning: {file} has wrong size {arr.size}")
            continue

        grid = arr.reshape((N, N))
        frames.append(grid)
    except Exception as e:
        print(f"Error loading {file}: {e}")

print(f"Loaded {len(frames)} frames.")
if len(frames) > 0:
    print(f"Frame shape: {frames[0].shape}")
else:
    print("No valid frames found. Check your CSV format.")

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

if frames:
    img = ax.imshow(frames[0], origin='lower', cmap='hot',
                    extent=[0, N, 0, N], vmin=0, vmax=100)
else:
    img = ax.imshow([[0]], origin='lower', cmap='hot')

plt.colorbar(img, ax=ax, label='Temperature (°C)')

# Update function for animation
time_per_frame = 200 / len(frames)

def update(frame_index):
    img.set_data(frames[frame_index])
    current_time = frame_index * time_per_frame
    ax.set_title(f"Heat Diffusion — t = {current_time:.2f} s")
    return [img]


anim = FuncAnimation(fig, update, frames=len(frames), interval=50, repeat=True)
plt.tight_layout()
plt.show()



