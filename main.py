from src.cli import SimulationCLI
import matplotlib.pyplot as plt
import time


def main():
    print("Starting Lid-Driven Cavity Flow Simulation...")
    cli = SimulationCLI()

    print("Initializing solver...")
    anim = cli.run()  # Get the animation object

    print("Simulation complete. Displaying results...")
    plt.show(block=True)  # Block until window is closed


if __name__ == "__main__":
    main()
