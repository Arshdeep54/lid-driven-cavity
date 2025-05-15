import argparse
import tomli
from pathlib import Path


class SimulationCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Lid-Driven Cavity Flow Simulation"
        )
        self.parser.add_argument(
            "--config",
            type=str,
            default="config/config.toml",
            help="Path to configuration file",
        )
        # Add arguments for each config parameter
        self.parser.add_argument(
            "--reynolds_number", type=float, help="Reynolds number"
        )
        self.parser.add_argument("--lid_velocity", type=float, help="Lid velocity")
        self.parser.add_argument("--cavity_size", type=float, help="Cavity size")
        self.parser.add_argument("--grid_points_x", type=int, help="Grid points in x")
        self.parser.add_argument("--grid_points_y", type=int, help="Grid points in y")
        self.parser.add_argument("--time_step", type=float, help="Time step")
        self.parser.add_argument(
            "--max_iterations", type=int, help="Maximum iterations"
        )
        self.parser.add_argument("--cfl_number", type=float, help="CFL number")
        self.parser.add_argument(
            "--artificial_viscosity", type=float, help="Artificial viscosity"
        )
        self.parser.add_argument("--plot_interval", type=int, help="Plot interval")
        self.parser.add_argument("--contour_levels", type=int, help="Contour levels")
        self.parser.add_argument("--quiver_spacing", type=int, help="Quiver spacing")
        self.parser.add_argument(
            "--show_velocities",
            action="store_true",
            help="Show velocity vectors at every cell",
        )

    def load_config(self, config_path):
        # Load default config
        default_config = {
            "simulation": {
                "reynolds_number": 100,
                "lid_velocity": 1.0,
                "cavity_size": 1.0,
                "grid_points_x": 41,
                "grid_points_y": 41,
                "time_step": 0.001,
                "max_iterations": 1000,
            },
            "stability": {"cfl_number": 0.5, "artificial_viscosity": 0.1},
            "visualization": {
                "plot_interval": 50,
                "contour_levels": 20,
                "quiver_spacing": 2,
            },
        }

        # Load user config if exists
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, "rb") as f:
                user_config = tomli.load(f)
                # Update default config with user values
                for section, values in user_config.items():
                    if section in default_config:
                        default_config[section].update(values)

        return default_config

    def update_config_from_args(self, config, args):
        """Update config with command line arguments if provided"""
        # Map argument names to config paths
        arg_to_config = {
            "reynolds_number": ("simulation", "reynolds_number"),
            "lid_velocity": ("simulation", "lid_velocity"),
            "cavity_size": ("simulation", "cavity_size"),
            "grid_points_x": ("simulation", "grid_points_x"),
            "grid_points_y": ("simulation", "grid_points_y"),
            "time_step": ("simulation", "time_step"),
            "max_iterations": ("simulation", "max_iterations"),
            "cfl_number": ("stability", "cfl_number"),
            "artificial_viscosity": ("stability", "artificial_viscosity"),
            "plot_interval": ("visualization", "plot_interval"),
            "contour_levels": ("visualization", "contour_levels"),
            "quiver_spacing": ("visualization", "quiver_spacing"),
        }

        # Update config with provided arguments
        for arg_name, (section, key) in arg_to_config.items():
            value = getattr(args, arg_name)
            if value is not None:
                config[section][key] = value
                print(f"Updated {section}.{key} to {value}")

        return config

    def run(self):
        args = self.parser.parse_args()
        config = self.load_config(args.config)
        config = self.update_config_from_args(config, args)
        
        print("\nFinal configuration:")
        for section, values in config.items():
            print(f"\n[{section}]")
            for key, value in values.items():
                print(f"{key} = {value}")

        from .solver import LidDrivenCavitySolver

        solver = LidDrivenCavitySolver(config)
        return solver.run()
