import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import time


class Visualizer:
    def __init__(self, config):
        self.interval = config["visualization"]["plot_interval"]
        self.levels = config["visualization"]["contour_levels"]
        self.quiver_spacing = config["visualization"]["quiver_spacing"]
        self.start_time = time.time()

        # Set up professional style
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

        # Custom colormap for fluid simulation
        self.cmap = plt.cm.get_cmap("viridis").copy()
        self.cmap.set_bad(color="white")

    def format_time(self, elapsed_seconds):
        """Format elapsed time in appropriate units"""
        if elapsed_seconds < 60:
            return f"{elapsed_seconds:.1f}s"
        elif elapsed_seconds < 3600:
            minutes = elapsed_seconds / 60
            return f"{minutes:.1f}min"
        else:
            hours = elapsed_seconds / 3600
            return f"{hours:.1f}hr"

    def animate(self, solver):
        print("Setting up visualization...")
        # Prepare grid
        x = np.linspace(0, solver.L, solver.nx)
        y = np.linspace(0, solver.L, solver.ny)
        X, Y = np.meshgrid(x, y)

        # Set up figure and initial plots with improved styling
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor("white")
        plt.tight_layout(pad=3.0)

        # Initial plot with enhanced styling
        contour = ax.contourf(X, Y, solver.psi, levels=self.levels, cmap=self.cmap)
        quiver = ax.quiver(
            X[:: self.quiver_spacing, :: self.quiver_spacing],
            Y[:: self.quiver_spacing, :: self.quiver_spacing],
            solver.u[:: self.quiver_spacing, :: self.quiver_spacing],
            solver.v[:: self.quiver_spacing, :: self.quiver_spacing],
            scale=50,
            color="white",
            alpha=0.8,
            width=0.004,
            headwidth=4,
            headlength=5,
        )

        # Enhanced colorbar
        cbar = fig.colorbar(contour, ax=ax, pad=0.02)
        cbar.set_label("Stream Function", fontsize=12, labelpad=10)
        cbar.ax.tick_params(labelsize=10)

        # Enhanced labels and title
        ax.set_xlabel("X", fontsize=12, labelpad=10)
        ax.set_ylabel("Y", fontsize=12, labelpad=10)
        ax.set_title("Lid-Driven Cavity Flow", fontsize=16, pad=20, weight="bold")

        # Add grid with custom styling
        ax.grid(True, linestyle="--", alpha=0.3)

        # Set axis limits with padding
        ax.set_xlim(-0.05, solver.L + 0.05)
        ax.set_ylim(-0.05, solver.L + 0.05)

        def update(frame):
            print(f"Updating frame {frame}...", end="\r")
            solver.apply_boundary_conditions()
            solver.solve_streamfunction()
            solver.solve_vorticity_transport()
            solver.calculate_velocity()

            # Calculate elapsed time
            elapsed_time = time.time() - self.start_time
            formatted_time = self.format_time(elapsed_time)

            # Clear previous contour
            for coll in ax.collections:
                coll.remove()

            # Draw new contour with enhanced styling
            new_contour = ax.contourf(
                X, Y, solver.psi, levels=self.levels, cmap=self.cmap
            )

            # Update quiver with enhanced styling
            quiver.set_UVC(
                solver.u[:: self.quiver_spacing, :: self.quiver_spacing],
                solver.v[:: self.quiver_spacing, :: self.quiver_spacing],
            )

            # Enhanced title with more information including elapsed time
            ax.set_title(
                f"Lid-Driven Cavity Flow\n"
                f"Re = {solver.Re:.0f}, t = {frame*solver.dt:.3f}s\n"
                f"Grid: {solver.nx}×{solver.ny}\n"
                f"Elapsed Time: {formatted_time}",
                fontsize=14,
                pad=20,
                weight="bold",
            )

            return [new_contour] + [quiver]

        print("Starting animation...")
        anim = FuncAnimation(
            fig, update, frames=solver.max_iter, interval=self.interval, blit=False
        )
        return anim
