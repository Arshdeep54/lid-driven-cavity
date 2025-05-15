import numpy as np
import matplotlib.pyplot as plt
from .utils.visualization import Visualizer


class LidDrivenCavitySolver:
    def __init__(self, config):
        # Extract simulation settings from config
        sim = config["simulation"]
        self.reynolds = sim["reynolds_number"]
        self.lid_speed = sim["lid_velocity"]
        self.length = sim["cavity_size"]
        self.nx = sim["grid_points_x"]
        self.ny = sim["grid_points_y"]
        self.dt = sim["time_step"]
        self.steps = sim["max_iterations"]

        # Grid size
        self.dx = self.length / (self.nx - 1)
        self.dy = self.length / (self.ny - 1)

        # Initialize flow fields
        self.stream = np.zeros((self.ny, self.nx))
        self.vorticity = np.zeros((self.ny, self.nx))
        self.u = np.zeros((self.ny, self.nx))  # horizontal velocity
        self.v = np.zeros((self.ny, self.nx))  # vertical velocity

        # Visualizer
        self.visualizer = Visualizer(config)

    def impose_boundary_conditions(self):
        """Set stream function and vorticity on all four walls"""
        self._top_lid()
        self._bottom_wall()
        self._side_walls()

    def _top_lid(self):
        """Top wall moves at constant speed (lid)"""
        self.stream[-1, :] = 0
        self.vorticity[-1, :] = (
            (self.stream[-1, :] - self.stream[-2, :]) / self.dy**2
            - 2 * self.lid_speed / self.dy
        )

    def _bottom_wall(self):
        """Bottom wall is stationary"""
        self.stream[0, :] = 0
        self.vorticity[0, :] = (self.stream[0, :] - self.stream[1, :]) / self.dy**2

    def _side_walls(self):
        """Left and right walls (stationary)"""
        self.stream[:, 0] = 0
        self.vorticity[:, 0] = (self.stream[:, 0] - self.stream[:, 1]) / self.dx**2

        self.stream[:, -1] = 0
        self.vorticity[:, -1] = (self.stream[:, -1] - self.stream[:, -2]) / self.dx**2

    def solve_stream_function(self):
        """Use Jacobi iteration to solve the Poisson equation"""
        for _ in range(self.steps):
            previous = self.stream.copy()

            self.stream[1:-1, 1:-1] = (
                self.dy**2 * (self.stream[1:-1, 2:] + self.stream[1:-1, :-2])
                + self.dx**2 * (self.stream[2:, 1:-1] + self.stream[:-2, 1:-1])
                - self.dx**2 * self.dy**2 * self.vorticity[1:-1, 1:-1]
            ) / (2 * (self.dx**2 + self.dy**2))

            if np.max(np.abs(self.stream - previous)) < 1e-6:
                break

    def update_vorticity(self):
        """Advance vorticity using Forward Euler in time and central difference in space"""
        omega = self.vorticity.copy()
        self.update_velocity_field()

        u = self.u
        v = self.v

        # Compute spatial derivatives
        dwdx = (omega[1:-1, 2:] - omega[1:-1, :-2]) / (2 * self.dx)
        dwdy = (omega[2:, 1:-1] - omega[:-2, 1:-1]) / (2 * self.dy)

        laplacian = (
            (omega[1:-1, 2:] - 2 * omega[1:-1, 1:-1] + omega[1:-1, :-2]) / self.dx**2
            + (omega[2:, 1:-1] - 2 * omega[1:-1, 1:-1] + omega[:-2, 1:-1]) / self.dy**2
        )

        omega[1:-1, 1:-1] += self.dt * (
            -u[1:-1, 1:-1] * dwdx - v[1:-1, 1:-1] * dwdy + (1 / self.reynolds) * laplacian
        )

        self.vorticity = omega

    def update_velocity_field(self):
        """Compute velocity from stream function gradients"""
        self.u[1:-1, 1:-1] = (
            self.stream[1:-1, 2:] - self.stream[1:-1, :-2]
        ) / (2 * self.dy)
        self.v[1:-1, 1:-1] = (
            -self.stream[2:, 1:-1] + self.stream[:-2, 1:-1]
        ) / (2 * self.dx)

    def run(self):
        """Run the entire simulation and animate it"""
        print("Simulation started...")
        animation = self.visualizer.animate(self)
        print("Rendering complete.")
        return animation
