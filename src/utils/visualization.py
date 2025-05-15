import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import time


class Visualizer:
    def __init__(self, config):
        self.cfg = config
        self.interval = config["visualization"]["plot_interval"]
        self.levels = config["visualization"]["contour_levels"]
        self.spacing = config["visualization"]["quiver_spacing"]
        self.t0 = time.time()

        plt.style.use("default")
        mpl.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "axes.grid": True,
                "grid.alpha": 0.3,
                "grid.linestyle": "--",
                "axes.labelsize": 12,
                "axes.titlesize": 14,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        )

        self.colormap = plt.cm.get_cmap("viridis").copy()
        self.colormap.set_bad("white")

    def _format_time(self, seconds_elapsed):
        if seconds_elapsed < 60:
            return f"{seconds_elapsed:.1f}s"
        elif seconds_elapsed < 3600:
            return f"{seconds_elapsed / 60:.1f}min"
        else:
            return f"{seconds_elapsed / 3600:.1f}hr"

    def animate(self, solver):
        print("Preparing visualization window...")
        draw_vectors = self.cfg["visualization"].get("show_velocities", False)

        x = np.linspace(0, solver.length, solver.nx)
        y = np.linspace(0, solver.length, solver.ny)
        X, Y = np.meshgrid(x, y)

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor("white")
        plt.tight_layout(pad=3.0)

        contour = ax.contourf(
            X, Y, solver.stream, levels=self.levels, cmap=self.colormap
        )

        if draw_vectors:
            qx, qy = X, Y
            u_display, v_display = solver.u, solver.v
        else:
            qx = X[:: self.spacing, :: self.spacing]
            qy = Y[:: self.spacing, :: self.spacing]
            u_display = solver.u[:: self.spacing, :: self.spacing]
            v_display = solver.v[:: self.spacing, :: self.spacing]

        arrows = ax.quiver(
            qx,
            qy,
            u_display,
            v_display,
            scale=50,
            color="white",
            alpha=0.8,
            width=0.004,
            headwidth=4,
            headlength=5,
        )

        colorbar = fig.colorbar(contour, ax=ax, pad=0.02)
        colorbar.set_label("Stream Function", fontsize=12)
        colorbar.ax.tick_params(labelsize=10)

        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_title("Lid-Driven Cavity Flow", fontsize=16, pad=20, weight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlim(-0.05, solver.length + 0.05)
        ax.set_ylim(-0.05, solver.length + 0.05)

        def update(frame):
            print(f"Frame {frame} in progress...", end="\r")

            solver.impose_boundary_conditions()
            solver.solve_stream_function()
            solver.update_vorticity()
            solver.update_velocity_field()

            elapsed = time.time() - self.t0
            formatted = self._format_time(elapsed)

            for c in ax.collections:
                c.remove()
            new_contour = ax.contourf(
                X, Y, solver.stream, levels=self.levels, cmap=self.colormap
            )

            if draw_vectors:
                arrows.set_UVC(solver.u, solver.v)
            else:
                arrows.set_UVC(
                    solver.u[:: self.spacing, :: self.spacing],
                    solver.v[:: self.spacing, :: self.spacing],
                )

            ax.set_title(
                f"Lid-Driven Cavity Flow\n"
                f"Re = {solver.reynolds:.0f}, t = {frame * solver.dt:.3f}s\n"
                f"Grid: {solver.nx}×{solver.ny}\n"
                f"Elapsed Time: {formatted}",
                fontsize=14,
                pad=20,
                weight="bold",
            )

            return [new_contour] + [arrows]

        print("Launching animation loop...")
        animation = FuncAnimation(
            fig, update, frames=solver.steps, interval=self.interval, blit=False
        )
        return animation
