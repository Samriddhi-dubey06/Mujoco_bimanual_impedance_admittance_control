#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation as anim
from pathlib import Path

# ---------- Paper-style setup ----------
mpl.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.dpi": 300,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.constrained_layout.use": True,
})

# ---------- Load Data ----------
CSV_PATH = "/home/samriddhi/VelGrasp/videos_samriddhi/videos_bimanual/lifting_weight_compensation/object_position_vs_time.csv"
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

# Build time (seconds)
t = df["t_since_start"].to_numpy()
measured = df["pose_z"].to_numpy()
desired = np.full_like(measured, 0.17, dtype=float)  # constant desired height

# ---------- Output file ----------
out_dir = Path(CSV_PATH).parent
stem = Path(CSV_PATH).stem
mp4_path = out_dir / f"{stem}_height_live.mp4"
gif_path = out_dir / f"{stem}_height_live.gif"

# ---------- Live Plot Function ----------
def live_plot(t, measured, desired, interval=50, window=20.0, ylim=(0.0, 0.3),
              save_mp4=None, save_gif=None, xlim_fixed=(0, 30)):
    """
    Live plot of measured vs desired height (m). Saves MP4 (ffmpeg) or GIF.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    plt.subplots_adjust(left=0.16, bottom=0.18)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height (m)")
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)

    (line_meas,) = ax.plot([], [], label="Measured (pose_z)", lw=1.8)
    (line_des,)  = ax.plot([], [], label=f"Desired (0.17 m)", lw=1.8)
    ax.legend(frameon=False)

    # Fixed x-limits 0–30 s
    if xlim_fixed is not None:
        ax.set_xlim(*xlim_fixed)

    def update(frame):
        t_now = t[:frame]
        m_now = measured[:frame]
        d_now = desired[:frame]

        if t_now.size == 0:
            return line_meas, line_des

        line_meas.set_data(t_now, m_now)
        line_des.set_data(t_now, d_now)
        return line_meas, line_des

    frames = np.arange(1, len(t) + 1)
    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False, repeat=False)

    # ---------- Save ----------
    if save_mp4:
        fps = max(1, int(round(1000.0 / max(1, interval))))
        try:
            writer = anim.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
            print(f"[info] Saving MP4 to {save_mp4}")
            ani.save(str(save_mp4), writer=writer, dpi=300,
                     savefig_kwargs={"bbox_inches": "tight", "pad_inches": 0.05})
            print(f"[info] MP4 saved successfully.")
        except Exception as e:
            print(f"[warn] MP4 save failed ({e}); trying GIF…")
            if save_gif:
                writer = anim.PillowWriter(fps=fps)
                ani.save(str(save_gif), writer=writer, dpi=200,
                         savefig_kwargs={"bbox_inches": "tight", "pad_inches": 0.05})
                print(f"[info] GIF saved to {save_gif}")

    # ---------- Show live animation ----------
    plt.show()

# ---------- Run ----------
if __name__ == "__main__":
    live_plot(
        t, measured, desired,
        interval=50, window=20, ylim=(0.0, 0.3),
        save_mp4=mp4_path, save_gif=gif_path,
        xlim_fixed=(0, 30)  # fixed 0–30s view
    )
