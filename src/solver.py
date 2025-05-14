import numpy as np
from .utils.visualization import Visualizer
import matplotlib.pyplot as plt


class LidDrivenCavitySolver:
    def __init__(self, config):
        # Simulation parameters
        self.Re = config["simulation"]["reynolds_number"]
        self.U = config["simulation"]["lid_velocity"]
        self.L = config["simulation"]["cavity_size"]
        self.nx = config["simulation"]["grid_points_x"]
        self.ny = config["simulation"]["grid_points_y"]
        self.dt = config["simulation"]["time_step"]
        self.max_iter = config["simulation"]["max_iterations"]

        # Stability parameters
        self.CFL = config["stability"]["cfl_number"]
        self.eps = config["stability"]["artificial_viscosity"]

        # Grid spacing
        self.dx = self.L / (self.nx - 1)
        self.dy = self.L / (self.ny - 1)

        # Initialize fields
        self.psi = np.zeros((self.ny, self.nx))
        self.vort = np.zeros((self.ny, self.nx))
        self.u = np.zeros((self.ny, self.nx))
        self.v = np.zeros((self.ny, self.nx))

        # Coefficients
        self.alpha = 1 / (self.Re * self.dx**2)
        self.beta = 1 / (self.Re * self.dy**2)
        self.gamma = 1 / (2 * self.dx)
        self.delta = 1 / (2 * self.dy)

        # Initialize visualizer
        self.visualizer = Visualizer(config)

    def apply_boundary_conditions(self):
        """Apply boundary conditions for stream function and vorticity"""
        self._apply_top_wall_conditions()
        self._apply_bottom_wall_conditions()
        self._apply_side_wall_conditions()

    def _apply_top_wall_conditions(self):
        """Apply conditions for the moving lid (top wall)"""
        self.psi[-1, :] = 0
        self.vort[-1, :] = (
            self.psi[-1, :] - self.psi[-2, :]
        ) / self.dy**2 - 2 * self.U / self.dy

    def _apply_bottom_wall_conditions(self):
        """Apply conditions for the bottom wall"""
        self.psi[0, :] = 0
        self.vort[0, :] = (self.psi[0, :] - self.psi[1, :]) / self.dy**2

    def _apply_side_wall_conditions(self):
        """Apply conditions for the left and right walls"""
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
            self._update_streamfunction()

            if self._check_streamfunction_convergence(psi_old):
                break

    def _update_streamfunction(self):
        """Update stream function values for interior points"""
        self.psi[1:-1, 1:-1] = (
            self.dy**2 * (self.psi[1:-1, 2:] + self.psi[1:-1, :-2])
            + self.dx**2 * (self.psi[2:, 1:-1] + self.psi[:-2, 1:-1])
            - self.dx**2 * self.dy**2 * self.vort[1:-1, 1:-1]
        ) / (2 * (self.dx**2 + self.dy**2))

    def _check_streamfunction_convergence(self, psi_old):
        """Check if stream function solution has converged"""
        return np.max(np.abs(self.psi - psi_old)) < 1e-6

    def solve_vorticity_transport(self):
        """Solve vorticity transport equation using FTCS scheme"""
        vort_new = self.vort.copy()

        # Calculate velocities and time step
        self.calculate_velocity()
        dt_actual = self._calculate_stable_timestep()

        # Update vorticity
        vort_new[1:-1, 1:-1] = self._compute_vorticity_update(dt_actual)
        self.vort = vort_new

    def _calculate_stable_timestep(self):
        """Calculate stable time step based on CFL condition"""
        max_u = np.max(np.abs(self.u))
        max_v = np.max(np.abs(self.v))

        # Prevent divide by zero
        if max_u == 0 or max_v == 0:
            return self.dt

        dt_cfl = min(self.dx / max_u, self.dy / max_v) * self.CFL
        return min(self.dt, dt_cfl)

    def _compute_vorticity_update(self, dt):
        """Compute vorticity update for interior points"""
        # Calculate spatial derivatives
        dvort_dx = (self.vort[1:-1, 2:] - self.vort[1:-1, :-2]) / (2 * self.dx)
        dvort_dy = (self.vort[2:, 1:-1] - self.vort[:-2, 1:-1]) / (2 * self.dy)

        # Calculate Laplacian
        laplacian_vort = self._compute_vorticity_laplacian()

        # Compute update
        return self.vort[1:-1, 1:-1] + dt * (
            -self.u[1:-1, 1:-1] * dvort_dx
            - self.v[1:-1, 1:-1] * dvort_dy
            + (1 / self.Re + self.eps) * laplacian_vort
        )

    def _compute_vorticity_laplacian(self):
        """Compute Laplacian of vorticity"""
        return (
            self.vort[1:-1, 2:] - 2 * self.vort[1:-1, 1:-1] + self.vort[1:-1, :-2]
        ) / self.dx**2 + (
            self.vort[2:, 1:-1] - 2 * self.vort[1:-1, 1:-1] + self.vort[:-2, 1:-1]
        ) / self.dy**2

    def calculate_velocity(self):
        """Calculate velocity components from stream function"""
        self.u[1:-1, 1:-1] = (self.psi[1:-1, 2:] - self.psi[1:-1, :-2]) / (2 * self.dy)
        self.v[1:-1, 1:-1] = -(self.psi[2:, 1:-1] - self.psi[:-2, 1:-1]) / (2 * self.dx)

    def simulate(self):
        """Main simulation loop"""
        print("Starting simulation...")
        anim = self.visualizer.animate(self)
        print("Animation created, displaying...")
        return anim
