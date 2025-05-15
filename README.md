# Lid-Driven Cavity Flow Simulation

## Overview

This project simulates the lid-driven cavity flow problem using the Navier–Stokes equations in the vorticity-streamfunction formulation. The simulation visualizes the flow of an incompressible fluid inside a square cavity where the top lid moves at a constant velocity, creating complex flow patterns.

## Mathematical Background

The Navier–Stokes equations for incompressible flow are:

\[
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u}
\]

\[
\nabla \cdot \mathbf{u} = 0
\]

In the vorticity-streamfunction formulation, we introduce the streamfunction \(\psi\) and vorticity \(\omega\):

\[
\omega = \nabla \times \mathbf{u}
\]

\[
\mathbf{u} = \nabla \times \psi
\]

This reduces the problem to solving two coupled equations:

1. **Poisson equation for the streamfunction:**
   \[
   \nabla^2 \psi = -\omega
   \]

2. **Vorticity transport equation:**
   \[
   \frac{\partial \omega}{\partial t} + (\mathbf{u} \cdot \nabla) \omega = \nu \nabla^2 \omega
   \]

## Key Components

### 1. Solver (`src/solver.py`)

The `LidDrivenCavitySolver` class implements the numerical solution:

- **Initialization:** Sets up the grid, flow fields, and simulation parameters.
- **Boundary Conditions:** Imposes no-slip conditions on the walls and moving lid.
- **Streamfunction Solver:** Uses Jacobi iteration to solve the Poisson equation for \(\psi\).
- **Vorticity Transport:** Advances the vorticity field using a finite-difference scheme.
- **Velocity Update:** Computes the velocity field from the streamfunction gradients.

### 2. Visualization (`src/utils/visualization.py`)

The `Visualizer` class handles the animation:

- **Setup:** Prepares the plot, contour, and velocity vectors.
- **Animation Loop:** Updates the simulation and refreshes the plot for each frame.
- **Time Formatting:** Converts elapsed time into a human-readable string.

## Running the Simulation

### Prerequisites

- Python 3.6+
- Required packages: `numpy`, `matplotlib`, `toml`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Arshdeep54/lid-driven-cavity.git
   cd lid-driven-cavity
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Edit the `config.toml` file to adjust simulation parameters:

```toml
[simulation]
reynolds_number = 1000
lid_velocity = 1.0
cavity_size = 1.0
grid_points_x = 100
grid_points_y = 100
time_step = 0.001
max_iterations = 1000

[visualization]
plot_interval = 50
contour_levels = 20
quiver_spacing = 5
show_velocities = true  # Set to false to hide velocity vectors
```

### Running the Simulation

Execute the main script:

```bash
python src/main.py
```

### Command-Line Options

- **`--show_velocities`:** Toggle visibility of velocity vectors (default: `true`).
- **`--reynolds`:** Set the Reynolds number (default: `1000`).

Example:

```bash
python src/main.py --show_velocities false --reynolds 500
```

## Results

The simulation produces an animated visualization of the lid-driven cavity flow, showing the streamfunction contour and velocity vectors. The title displays the Reynolds number, simulation time, grid size, and elapsed runtime.

