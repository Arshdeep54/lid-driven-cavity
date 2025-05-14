import numpy as np
import matplotlib.pyplot as plt
import argparse

class LidDrivenCavity:
    def __init__(self, N=64, Re=1000, U=1.0, dt=0.001, max_iter=10000):
        self.N = N
        self.Re = Re
        self.U = U
        self.dt = dt
        self.max_iter = max_iter

        self.dx = 1.0 / N
        self.dy = 1.0 / N

        self.psi = np.zeros((N + 2, N + 2))
        self.omega = np.zeros((N + 2, N + 2))

        self.x = np.linspace(0, 1, N + 2)
        self.y = np.linspace(0, 1, N + 2)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def apply_boundary_conditions(self):
        N, dx, dy, U = self.N, self.dx, self.dy, self.U

        self.omega[1:N+1, N+1] = -2 * (self.psi[1:N+1, N] - dy * U) / dy**2
        self.omega[1:N+1, 0] = -2 * self.psi[1:N+1, 1] / dy**2
        self.omega[0, 1:N+1] = -2 * self.psi[1, 1:N+1] / dx**2
        self.omega[N+1, 1:N+1] = -2 * self.psi[N, 1:N+1] / dx**2

    def solve_streamfunction(self, tol=1e-6, max_iter=10000):
        N, dx, dy = self.N, self.dx, self.dy
        psi = self.psi.copy()

        for _ in range(max_iter):
            psi_old = psi.copy()
            psi[1:N+1, 1:N+1] = 0.25 * (
                psi_old[2:N+2, 1:N+1] + psi_old[0:N, 1:N+1] +
                psi_old[1:N+1, 2:N+2] + psi_old[1:N+1, 0:N] +
                self.omega[1:N+1, 1:N+1] * dx**2
            )
            if np.linalg.norm(psi - psi_old, ord=np.inf) < tol:
                break

        self.psi = psi

    def step(self):
        N, dx, dy, dt, Re = self.N, self.dx, self.dy, self.dt, self.Re
        omega = self.omega.copy()
        psi = self.psi

        u = (psi[1:N+1, 2:N+2] - psi[1:N+1, 0:N]) / (2 * dy)
        v = -(psi[2:N+2, 1:N+1] - psi[0:N, 1:N+1]) / (2 * dx)

        omega[1:N+1, 1:N+1] += dt * (
            -u * (omega[1:N+1, 2:N+2] - omega[1:N+1, 0:N]) / (2 * dy)
            -v * (omega[2:N+2, 1:N+1] - omega[0:N, 1:N+1]) / (2 * dx)
            + (1 / Re) * (
                (omega[2:N+2, 1:N+1] - 2 * omega[1:N+1, 1:N+1] + omega[0:N, 1:N+1]) / dx**2 +
                (omega[1:N+1, 2:N+2] - 2 * omega[1:N+1, 1:N+1] + omega[1:N+1, 0:N]) / dy**2
            )
        )

        self.omega = omega

    def run(self):
        for it in range(self.max_iter):
            self.apply_boundary_conditions()
            self.solve_streamfunction()
            self.step()
            if it % 100 == 0:
                print(f"Iteration: {it}")

    def plot(self):
        plt.contourf(self.X, self.Y, self.psi.T, levels=50, cmap='viridis')
        plt.colorbar(label='Streamfunction')
        plt.title('Lid Driven Cavity Flow (Streamfunction)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal')
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Lid Driven Cavity Simulation")
    parser.add_argument("--u", type=float, default=1.0, help="Top lid velocity")
    parser.add_argument("--n", type=int, default=64, help="Grid size")
    parser.add_argument("--re", type=float, default=1000.0, help="Reynolds number")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step")
    parser.add_argument("--iter", type=int, default=1000, help="Max iterations")
    return parser.parse_args()


def main():
    args = parse_args()
    sim = LidDrivenCavity(N=args.n, Re=args.re, U=args.u, dt=args.dt, max_iter=args.iter)
    sim.run()
    sim.plot()


if __name__ == "__main__":
    main()