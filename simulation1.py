import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class LidDrivenCavitySolver:
    def __init__(self, Re, U, L, nx, ny, dt, max_iter):
        self.Re = Re  # Reynolds number
        self.U = U  # Lid velocity
        self.L = L  # Cavity size
        self.nx = nx  # Grid points in x
        self.ny = ny  # Grid points in y
        self.dt = dt  # Time step
        self.max_iter = max_iter  # Maximum iterations

        # Grid spacing
        self.dx = L / (nx - 1)
        self.dy = L / (ny - 1)

        # Initialize fields
        self.psi = np.zeros((ny, nx))  # Stream function
        self.vort = np.zeros((ny, nx))  # Vorticity
        self.u = np.zeros((ny, nx))  # x-velocity
        self.v = np.zeros((ny, nx))  # y-velocity

        # Coefficients
        self.alpha = 1 / (Re * self.dx**2)
        self.beta = 1 / (Re * self.dy**2)
        self.gamma = 1 / (2 * self.dx)
        self.delta = 1 / (2 * self.dy)

        # Add CFL number
        self.CFL = 0.5  # CFL number should be less than 1 for stability

    def apply_boundary_conditions(self):
        """Apply boundary conditions for stream function and vorticity"""
        # Top wall (moving lid)
        self.psi[-1, :] = 0
        self.vort[-1, :] = (
            self.psi[-1, :] - self.psi[-2, :]
        ) / self.dy**2 - 2 * self.U / self.dy

        # Bottom wall
        self.psi[0, :] = 0
        self.vort[0, :] = (self.psi[0, :] - self.psi[1, :]) / self.dy**2

        # Left wall
        self.psi[:, 0] = 0
        self.vort[:, 0] = (self.psi[:, 0] - self.psi[:, 1]) / self.dx**2

        # Right wall
        self.psi[:, -1] = 0
        self.vort[:, -1] = (self.psi[:, -1] - self.psi[:, -2]) / self.dx**2

    def solve_streamfunction(self):
        """Solve Poisson equation for stream function using finite differences"""
        for _ in range(self.max_iter):
            psi_old = self.psi.copy()

            # Interior points
            self.psi[1:-1, 1:-1] = (
                self.dy**2 * (self.psi[1:-1, 2:] + self.psi[1:-1, :-2])
                + self.dx**2 * (self.psi[2:, 1:-1] + self.psi[:-2, 1:-1])
                - self.dx**2 * self.dy**2 * self.vort[1:-1, 1:-1]
            ) / (2 * (self.dx**2 + self.dy**2))

            # Check convergence
            if np.max(np.abs(self.psi - psi_old)) < 1e-6:
                break

    def solve_vorticity_transport(self):
        """Solve vorticity transport equation using FTCS scheme"""
        vort_new = self.vort.copy()

        # Calculate velocities first
        self.calculate_velocity()

        # Calculate CFL condition
        max_u = np.max(np.abs(self.u))
        max_v = np.max(np.abs(self.v))
        dt_cfl = min(self.dx / max_u, self.dy / max_v) * self.CFL

        # Use the smaller of the two time steps
        dt_actual = min(self.dt, dt_cfl)

        # Rest of the vorticity transport calculation
        dvort_dx = (self.vort[1:-1, 2:] - self.vort[1:-1, :-2]) / (2 * self.dx)
        dvort_dy = (self.vort[2:, 1:-1] - self.vort[:-2, 1:-1]) / (2 * self.dy)
        laplacian_vort = (
            self.vort[1:-1, 2:] - 2 * self.vort[1:-1, 1:-1] + self.vort[1:-1, :-2]
        ) / self.dx**2 + (
            self.vort[2:, 1:-1] - 2 * self.vort[1:-1, 1:-1] + self.vort[:-2, 1:-1]
        ) / self.dy**2

        # Add artificial viscosity for stability
        eps = 0.1  # artificial viscosity coefficient

        vort_new[1:-1, 1:-1] = self.vort[1:-1, 1:-1] + dt_actual * (
            -self.u[1:-1, 1:-1] * dvort_dx
            - self.v[1:-1, 1:-1] * dvort_dy
            + (1 / self.Re + eps) * laplacian_vort
        )

        self.vort = vort_new

    def calculate_velocity(self):
        """Calculate velocity components from stream function"""
        self.u[1:-1, 1:-1] = (self.psi[1:-1, 2:] - self.psi[1:-1, :-2]) / (2 * self.dy)
        self.v[1:-1, 1:-1] = -(self.psi[2:, 1:-1] - self.psi[:-2, 1:-1]) / (2 * self.dx)

    def simulate(self):
        """Main simulation loop"""
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Initialize plot
        x = np.linspace(0, self.L, self.nx)
        y = np.linspace(0, self.L, self.ny)
        X, Y = np.meshgrid(x, y)

        contour = ax.contourf(X, Y, self.psi, levels=20, cmap="viridis")
        ax.quiver(X[::2, ::2], Y[::2, ::2], self.u[::2, ::2], self.v[::2, ::2])
        plt.colorbar(contour)

        def update(frame):
            self.apply_boundary_conditions()
            self.solve_streamfunction()
            self.solve_vorticity_transport()
            self.calculate_velocity()

            # Update plot
            ax.clear()
            contour = ax.contourf(X, Y, self.psi, levels=20, cmap="viridis")
            ax.quiver(X[::2, ::2], Y[::2, ::2], self.u[::2, ::2], self.v[::2, ::2])
            ax.set_title(f"Lid-Driven Cavity Flow (t = {frame*self.dt:.2f}s)")
            return (contour,)

        ani = FuncAnimation(fig, update, frames=self.max_iter, interval=50, blit=False)
        plt.show()


class SimulationCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Lid-Driven Cavity Flow Simulation"
        )
        self.parser.add_argument(
            "--Re", type=float, default=100, help="Reynolds number"
        )
        self.parser.add_argument("--U", type=float, default=1.0, help="Lid velocity")
        self.parser.add_argument("--L", type=float, default=1.0, help="Cavity size")
        self.parser.add_argument("--nx", type=int, default=41, help="Grid points in x")
        self.parser.add_argument("--ny", type=int, default=41, help="Grid points in y")
        self.parser.add_argument("--dt", type=float, default=0.001, help="Time step")
        self.parser.add_argument(
            "--max_iter", type=int, default=1000, help="Maximum iterations"
        )

    def run(self):
        args = self.parser.parse_args()
        solver = LidDrivenCavitySolver(
            Re=args.Re,
            U=args.U,
            L=args.L,
            nx=args.nx,
            ny=args.ny,
            dt=args.dt,
            max_iter=args.max_iter,
        )
        solver.simulate()


if __name__ == "__main__":
    cli = SimulationCLI()
    cli.run()
