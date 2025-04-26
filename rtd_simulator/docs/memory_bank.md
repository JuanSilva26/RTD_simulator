# RTD Simulator - Memory Bank

## Project Overview

The RTD (Resonant Tunneling Diode) Simulator is a Python-based application designed to simulate and analyze the behavior of RTD devices. The project implements two different models of RTD behavior:

1. Simplified Model: A basic model for educational purposes
2. Schulman Model: A more complex model based on physical equations

## Core Components

### 1. Model Layer

Located in `rtd_simulator/model/`

#### RTD Models

- `RTDModel` (Abstract Base Class)

  - Defines the interface for all RTD models
  - Key methods:
    - `iv_characteristic`: Calculates current-voltage relationship
    - `simulate`: Runs time-domain simulation

- `SimplifiedRTDModel`

  - Implements a basic RTD model
  - Parameters:
    - `m`: Stiffness parameter
    - `r`: Resistance parameter
  - Voltage range: -3V to 3V

- `SchulmanRTDModel`
  - Implements the Schulman equation for RTD
  - Fixed parameters:
    - `a, b, c, d, h, n1, n2`: Fitting parameters
    - `T`: Temperature (300K)
  - Adjustable parameters:
    - `R`: Circuit resistance
    - `L`: Circuit inductance
    - `C`: Device capacitance
  - Voltage range: 0V to 4.5V

### 2. View Layer

Located in `rtd_simulator/view/`

#### Main Components

- `MainWindow`

  - Central application window
  - Manages all UI components
  - Handles model switching and parameter updates

- `Parameter Sections`

  - `RTDParameterSection`: Controls for RTD model parameters
  - `SimulationParameterSection`: Controls for simulation settings
  - `PulseParameterSection`: Controls for pulse signal generation

- `Plotting`
  - `RTDPlotter`: Handles all plotting functionality
  - Features:
    - IV curve plotting
    - Time-domain simulation results
    - Phase space analysis
    - Frequency analysis

### 3. Controller Layer

Located in `rtd_simulator/controller/`

#### RTDController

- Manages interaction between model and view
- Key responsibilities:
  - Parameter updates
  - Simulation execution
  - Data analysis
  - Plot generation
  - Export functionality

## Key Functionality

### 1. Model Switching

- Users can switch between Simplified and Schulman models
- Each model has its own set of parameters
- UI adapts to show relevant controls

### 2. Simulation

- Time-domain simulation of RTD behavior
- Supports:
  - DC bias voltage
  - Pulse signals (square, sine, triangle, sawtooth)
  - Customizable simulation duration and time step

### 3. Analysis

- IV curve analysis
  - Peak detection
  - Curve fitting
  - Parameter extraction
- Dynamics analysis
  - Phase space plots
  - Frequency analysis
  - Time-domain behavior

### 4. Visualization

- Real-time plotting
- Multiple plot types:
  - IV characteristics
  - Voltage vs Time
  - Current vs Time
  - Phase space
  - Frequency spectrum

### 5. Data Export

- Export simulation data to CSV
- Export plots in various formats
- High-quality plot generation for publications

## Project Structure

```
rtd_simulator/
├── controller/
│   └── rtd_controller.py
├── model/
│   ├── rtd_model.py
│   └── curve_analysis.py
├── view/
│   ├── main_window.py
│   ├── parameter_sections.py
│   ├── plotting.py
│   └── custom_widgets.py
├── docs/
│   └── memory_bank.md
└── tests/
    └── test_model.py
```

## Dependencies

- PyQt6: GUI framework
- NumPy: Numerical computations
- Matplotlib: Plotting
- SciPy: Scientific computations
- Pandas: Data handling

## Usage Flow

1. User selects RTD model type
2. Sets model parameters
3. Configures simulation settings
4. Runs simulation
5. Views results in plots
6. Can perform analysis and export data

## Key Features

- Interactive parameter adjustment
- Real-time visualization
- Multiple analysis tools
- Export capabilities
- Educational and research applications

## Future Enhancements

- Additional RTD models
- More analysis tools
- Batch simulation capabilities
- Parameter optimization
- Machine learning integration

## Notes for AI Understanding

- The project follows MVC architecture
- Models are physics-based implementations
- UI is built with PyQt6 for cross-platform compatibility
- All numerical computations use NumPy for efficiency
- Plotting is handled by Matplotlib with custom styling
- The codebase is well-documented and type-hinted
