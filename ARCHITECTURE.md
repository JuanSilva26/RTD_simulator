# RTD Simulator Architecture

## Overview

The RTD Simulator follows a Model-View-Controller (MVC) architecture with a plugin system for extensibility.

```python
rtd_simulator/
├── models/                 # Model layer
│   ├── __init__.py
│   ├── base.py            # Abstract base classes
│   ├── rtd_model.py       # Core RTD models
│   └── plugins/           # Plugin RTD models
├── views/                 # View layer
│   ├── __init__.py
│   ├── main_window.py    # Main application window
│   ├── plot_view.py      # Real-time plotting
│   └── widgets/          # Custom Qt widgets
├── controllers/          # Controller layer
│   ├── __init__.py
│   ├── simulation.py     # Simulation controller
│   └── analysis.py       # Analysis controller
├── utils/               # Shared utilities
│   ├── __init__.py
│   ├── profiling.py     # Profiling tools
│   └── config.py        # Configuration handling
└── app.py              # Application entry point
```

## Components

### Model Layer

- Abstract base classes for RTD models
- Core implementations (Simplified, Schulman)
- Plugin system for extensible models
- Data structures and numerical methods

### View Layer

- Qt-based UI components
- Real-time plotting with Matplotlib
- Custom widgets for parameter input
- Progress reporting

### Controller Layer

- Simulation management
- Analysis coordination
- Thread handling
- Event processing

### Utils

- Profiling tools
- Configuration management
- Common utilities

## Plugin System

- Dynamic model loading
- Registry for models and analyses
- Hot-reloading capability

## Threading Model

- Background simulation execution
- UI responsiveness
- Progress reporting
- Cancellation support

## Data Flow

1. User inputs parameters via View
2. Controller validates and initializes Model
3. Model performs simulation in background thread
4. Results streamed to View via Controller
5. Real-time updates through signal/slot

## Next Implementation Steps

1. Reorganize files according to MVC structure
2. Move simulation to background threads
3. Add progress reporting
4. Implement real-time plotting optimizations
