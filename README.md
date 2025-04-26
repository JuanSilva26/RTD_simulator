# RTD Simulator

A Python-based simulation platform for Resonant Tunneling Diodes (RTDs) with support for multiple physical models.

## Available Models

### 1. Simplified Model (Dimensionless)
A basic dimensionless model suitable for studying RTD dynamics and basic behaviors:
- Simplified IV characteristic with cubic-like function
- Normalized parameters for educational purposes
- Ideal for understanding basic RTD behavior

### 2. Schulman Model (Physical)
A physically accurate model based on Schulman's equations:

**Device Dynamics:**
```math
dV/dt = 1/C * (I - F(V))
dI/dt = 1/L * (Vbias - V - R*I)
```
where F(V) is the Schulman IV characteristic:
```math
F(V) = J1 * J2 + J3
```
with components:
- J1 = a * ln((1 + term_up)/(1 + term_down))
- J2 = π/2 + arctan((c - n1*V)/d)
- J3 = h * (exp(qe/(k*T) * (n2*V)) - 1)
- term_up = exp(qe/(k*T) * (b - c + n1*V))
- term_down = exp(qe/(k*T) * (b - c - n1*V))

**Default Fitting Parameters:**
- a = 6.715e-4 A (Current scale)
- b = 6.499e-2 V (Voltage offset)
- c = 9.709e-2 V (Peak voltage)
- d = 2.213e-2 V (Width parameter)
- h = 1.664e-4 A (Current scale)
- n1 = 3.106e-2 (Voltage coefficient)
- n2 = 1.721e-2 (Voltage coefficient)

**Physical Parameters and Ranges:**
- C: Device capacitance [pF range]
- L: Circuit inductance [nH to µH range]
- R: Circuit resistance [Ω range]
- Vbias: Bias voltage [0 to few V]
- T: Temperature (default 300K)

**Simulation Parameters:**
- Time scales: nanoseconds to milliseconds
- Pulse amplitude: microamps to amps
- Pulse cycle time: nanoseconds to milliseconds
- k: Boltzmann constant (1.380649e-23 J/K)
- qe: Electron charge (1.602176634e-19 C)

## Overview

This project implements a simulation environment for studying the behavior of Resonant Tunneling Diodes (RTDs), which are quantum devices exhibiting unique N-shaped current-voltage (IV) characteristics. The platform allows for real-time parameter adjustment and visualization of RTD dynamics, making it an ideal tool for research and educational purposes.

## RTD Device Characteristics

Resonant Tunneling Diodes are quantum devices that exhibit:
- N-shaped IV curve with two Positive Differential Conductance (PDC) regions
- One Negative Differential Resistance (NDR) region
- Self-sustained oscillations in the NDR region
- Triggered oscillations when biased near NDR with sufficient perturbation
- Dimensionless model with frequency and amplitude independence from bias point
- Threshold-based oscillation activation

## Mathematical Model

The RTD dynamics are described by the following coupled differential equations:

```
dv/dt = (i - f(v))/m
di/dt = m(vbias - v - ri)
```

Where:
- v: voltage across the device
- i: current through the device
- m: stiffness parameter
- r: resistance
- f(v) = kv - h*arctan(v/w): IV function
- k, h, w: fitting parameters (5.92, 15.76, 2.259)

## Features

- Real-time simulation of RTD dynamics
- Interactive parameter adjustment
- Visualization of IV characteristics
- Time-domain waveform visualization
- Phase space analysis
- Customizable perturbation pulses
- Export capabilities for simulation data
- Performance-optimized for Apple Silicon (M-series processors)

## Requirements

- Python 3.8+
- Anaconda environment (INL_env)
- PyTorch
- NumPy
- Matplotlib
- PyQt6 (for GUI)
- SciPy
- Jupyter (for documentation and examples)

## Installation

1. Clone this repository
2. Activate your Anaconda environment:
   ```bash
   conda activate INL_env
   ```
3. Install additional required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

[To be added as the project develops]

## Project Structure

```
rtd_simulator/
├── src/
│   ├── core/           # Core simulation engine
│   ├── gui/            # Graphical user interface
│   ├── visualization/  # Plotting and visualization tools
│   └── utils/          # Utility functions
├── tests/              # Unit tests and validation
├── docs/               # Documentation
├── examples/           # Example scripts and notebooks
└── data/              # Sample data and configurations
```

## Contributing

[To be added]

## License

[To be added]

## Acknowledgments

[To be added] 