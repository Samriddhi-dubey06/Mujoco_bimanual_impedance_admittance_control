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
    # help layout automatically
    "figure.constrained_layout.use": True,
})

# ---------- Load Data ----------
CSV_PATH = "/home/samriddhi/VelGrasp/videos_samriddhi/videos_bimanual/lifting_weight_compensation/object_position_vs_time.csv"
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

# Build time from first sample (seconds)
t0 = df["time"].iloc[0]
df["t_sec"] = df["time"] - t0

# Extract arrays for live plotting
t = df["t_sec"].to_numpy()
F_meas_z = df["meas_Fy"].to_numpy()
F_des_z_series = df["pose_z"].to_numpy()

# ---------- Output file ----------
out_dir = Path(CSV_PATH).parent
stem = Path(CSV_PATH).stem
mp4_path = out_dir / f"{stem}_live.mp4"
gif_path = out_dir / f"{stem}_live.gif"

# ---------- Live Plot Function ----------
def live_plot(t, F_meas, F_des, interval=50, window=20, ylim=(-40, 40),
              save_path_mp4=None, save_path_gif=None,
              xlim_fixed=None):
    """
    Animate Measured vs Desired force traces.
    Also save as MP4 (ffmpeg) or fallback GIF.
    """
    # Use constrained_layout=True to avoid clipping; still add a tiny manual pad as safety.
    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    # Safety padding in case constrained layout is not perfect on your backend
    plt.subplots_adjust(left=0.16, bottom=0.18)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)

    (line_meas,) = ax.plot([], [], label="Measured Force", lw=1.8)
    (line_des,)  = ax.plot([], [], label="Desired Force",  lw=1.8)
    ax.legend(frameon=False)

    # Optional: fix x-limits (e.g., (2, 17)) if you want a constant view window
    if xlim_fixed is not None:
        ax.set_xlim(*xlim_fixed)

    def update(frame):
        t_now = t[:frame]
        meas_now = F_meas[:frame]
        des_now = F_des[:frame]

        if t_now.size == 0:
            return line_meas, line_des

        if window is not None and xlim_fixed is None:
            tmax = t_now[-1]
            mask = t_now >= (tmax - window)
            t_now = t_now[mask]
            meas_now = meas_now[mask]
            des_now = des_now[mask]

        line_meas.set_data(t_now, meas_now)
        line_des.set_data(t_now, des_now)

        # Auto x-limits only if not fixed
        if xlim_fixed is None:
            xmin = float(t_now[0])
            xmax = float(t_now[-1])
            if xmin == xmax:
                xmax = xmin + 1e-6
            headroom = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
            ax.set_xlim(xmin, xmax + headroom)

        return line_meas, line_des

    ani = FuncAnimation(
        fig, update,
        frames=np.arange(1, len(t) + 1),
        interval=interval,
        blit=False,
        repeat=False
    )

    # ---------- Save ----------
    if save_path_mp4:
        fps = max(1, int(round(1000.0 / interval)))
        try:
            writer = anim.FFMpegWriter(
                fps=fps, codec="libx264",
                extra_args=["-pix_fmt", "yuv420p"]
            )
            print(f"[info] Saving MP4 to {save_path_mp4}")
            # savefig_kwargs helps reduce clipping in some Matplotlib versions
            ani.save(str(save_path_mp4), writer=writer, dpi=300,
                     savefig_kwargs={"bbox_inches": "tight", "pad_inches": 0.05})
            print("[info] MP4 saved successfully.")
        except Exception as e:
            print(f"[warn] MP4 save failed ({e}), trying GIF instead…")
            if save_path_gif:
                writer = anim.PillowWriter(fps=fps)
                ani.save(str(save_path_gif), writer=writer, dpi=200,
                         savefig_kwargs={"bbox_inches": "tight", "pad_inches": 0.05})
                print(f"[info] GIF saved to {save_path_gif}")

    # ---------- Show live animation ----------
    plt.show()

# ---------- Run ----------
if __name__ == "__main__":
    # If you want fixed time range visible (no sliding), set xlim_fixed=(2, 17)
    live_plot(
        t, F_meas_z, F_des_z_series,
        interval=50, window=20, ylim=(-40, 40),
        save_path_mp4=mp4_path, save_path_gif=gif_path,
        xlim_fixed=(1, 30) # or e.g., (2, 17)
    )
