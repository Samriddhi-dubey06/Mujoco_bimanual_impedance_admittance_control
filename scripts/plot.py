import pandas as pd
import matplotlib.pyplot as plt

# === Load the CSV ===
file_path = "/home/samriddhi/VelGrasp/force_comparison_log.csv"
df = pd.read_csv(file_path)

# === Force component labels ===
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
        plt.grid(True)
        if plot_idx == 1:
            plt.legend()
        plot_idx += 1

plt.tight_layout()
plt.suptitle("Measured vs LP Forces (Single CSV)", fontsize=16, y=1.02)
plt.show()
