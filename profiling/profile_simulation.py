import cProfile
import pstats
import sys
from pathlib import Path
from rtd_simulator.controller.rtd_controller import RTDController
from rtd_simulator.model.base import get_rtd_model
from rtd_simulator.view.plotting import RTDPlotter


def profile_simulation(model_type: str = "Simplified", output: str = "simulation.prof"):
    """
    Profile a simulation run using cProfile and save the results.
    Args:
        model_type: Name of the RTD model to use ("Simplified" or "Schulman")
        output: Path to save the .prof file
    """
    # Set up a dummy plotter (no GUI)
    plotter = RTDPlotter()
    controller = RTDController(plotter=plotter)
    controller.update_parameters(model_type=model_type)
    # Typical simulation parameters
    t_end = 1e-6 if model_type == "Schulman" else 100.0
    dt = 1e-9 if model_type == "Schulman" else 0.01
    vbias = 3.0 if model_type == "Schulman" else 0.0
    
    def run():
        controller.run_simulation(
            t_end=t_end,
            dt=dt,
            vbias=vbias,
            pulse_amplitude=0.1,
            pulse_frequency=0.04,
            duty_cycle=0.5,
            pulse_type="square",
            offset=0.0
        )
    
    print(f"Profiling {model_type} simulation...")
    cProfile.runctx("run()", globals(), locals(), output)
    print(f"Profile saved to {output}")
    stats = pstats.Stats(output)
    stats.strip_dirs().sort_stats("cumulative").print_stats(20)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Profile RTD simulation with cProfile.")
    parser.add_argument("--model", type=str, default="Simplified", help="Model type: Simplified or Schulman")
    parser.add_argument("--output", type=str, default="simulation.prof", help="Output .prof file")
    args = parser.parse_args()
    profile_simulation(model_type=args.model, output=args.output) 