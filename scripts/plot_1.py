import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
file_path = "/home/samriddhi/VelGrasp/force_comparison_log.csv"
df = pd.read_csv(file_path)

# === Focused range config ===
time_min = 0      # Start time (s)
time_max = 5      # End time (s)
force_limit = 50  # Y-axis force limit (N)
num_yticks = 11   # More points on Y-axis

# Filter by time
df = df[(df["time"] >= time_min) & (df["time"] <= time_max)]

# Labels and structure
components = ["fx", "fy", "fz"]
sides = ["left", "right"]

plt.figure(figsize=(12, 8))
plot_idx = 1

for side in sides:
    for comp in components:
        col_measured = f"f_{side}_{comp}"
        col_lambda = f"lambda_{side}_{comp}"

        plt.subplot(2, 3, plot_idx)
        plt.plot(df["time"], df[col_measured], label="Measured", linestyle='--')
        plt.plot(df["time"], df[col_lambda], label="LP (lambda)", linestyle='-')
        plt.title(f"{side.capitalize()} {comp.upper()}")
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.ylim(-force_limit, force_limit)
        plt.yticks(np.linspace(-force_limit, force_limit, num_yticks))
        plt.grid(True)

        if plot_idx == 1:
            plt.legend()

        plot_idx += 1

plt.tight_layout()
plt.suptitle(f"Measured vs LP Forces — Time {time_min}s to {time_max}s", fontsize=16, y=1.02)
plt.show()
