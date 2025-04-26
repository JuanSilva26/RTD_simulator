# RTD Simulator Roadmap

## Phase 1: Core Architecture & Robustness (Completed)

### 1.1 Project Structure & Packaging

- [x] Implement MVC architecture
- [x] Modernize packaging system
  - [x] Migrate from `setup.py` to `pyproject.toml`
  - [x] Consolidate dependencies in `pyproject.toml`
  - [x] Add proper entry points for CLI
  - [x] Define proper package metadata
- [x] Fix import paths and consistency
  - [x] Update all relative imports to use correct paths
  - [x] Ensure consistent naming (`models/` vs `model/`)
  - [x] Add proper `__init__.py` files
  - [x] Fix circular imports

### 1.2 Code Quality & Maintainability

- [x] Implement comprehensive type hints
  - [x] Add return type hints to base model methods
  - [x] Add proper type annotations for base model data structures
  - [x] Use `typing` module for complex types in base model
  - [x] Add type aliases for better readability
  - [x] Add type hints to SimplifiedRTDModel implementation
  - [x] Add type hints to SchulmanRTDModel implementation
  - [x] Add type hints to controller methods
  - [x] Add type hints to view components
- [x] Improve error handling and logging
  - [x] Create centralized logging configuration
  - [x] Replace print statements with proper logging
  - [x] Add structured error handling in critical paths
  - [x] Implement log rotation and formatting
- [x] Code style and formatting
  - [x] Add Black for consistent formatting
  - [x] Implement flake8 for linting
  - [x] Add pre-commit hooks

## Phase 2: Core Functionality & Performance (Completed)

### 2.1 Physics & Numerics

- [x] Centralize numerical methods
  - [x] Move RK4 implementation to models layer
  - [x] Create abstract numerical methods interface
  - [x] Implement proper time-stepping abstraction
- [x] Performance optimization
  - [x] Implement Numba-accelerated simulation
  - [x] Add vectorized operations where possible
  - [x] Implement lazy IV-curve caching
  - [x] Add performance benchmarks

### 2.2 Model Architecture

- [x] Implement plugin system
- [x] Add plugin documentation
- [x] Create validation framework for plugins
- [x] Add simulation boundary checks
  - [x] Implement voltage range validation
  - [x] Add current limit checks
  - [x] Add stability boundary detection
- [x] Create example plugins
  - [x] Add basic RTD model example
  - [x] Add visualization plugin

## Phase 3: UI & User Experience (Current Focus)

### 3.1 UI Improvements

- [x] Modernize UI components
  - [x] Improve responsive layout
  - [x] Add unit selection for pulse parameters
    - [x] Amplitude units (V, mV, µV)
    - [x] Frequency units (Hz, kHz, MHz, GHz)
- [ ] Enhanced visualization
  - [ ] Add real-time parameter effects preview
  - [ ] Implement interactive plot controls

### 3.2 User Experience Enhancements

- [x] Improved unit handling
  - [x] Smart unit conversion for all parameters
  - [x] Consistent unit display across UI
  - [x] Unit-aware value validation
- [ ] Parameter presets
  - [ ] Save/load parameter configurations
  - [ ] Default presets for common scenarios
- [ ] Contextual help
  - [ ] Parameter tooltips with physical explanations
  - [ ] Quick reference guides

## Phase 5: Advanced Features

### 5.1 Analysis Tools

- [x] Add peak detection
  - [ ] Implement curve fitting
  - [ ] Add statistical analysis
- [ ] Data export
  - [ ] Add multiple format support
  - [ ] Implement batch export
  - [ ] Add metadata support

### 5.2 Extensibility

- [ ] Plugin framework
  - [ ] Add plugin configuration
  - [ ] Implement plugin dependencies
  - [ ] Add plugin versioning
- [ ] API
  - [ ] Create Python API
  - [ ] Add REST API support
  - [ ] Implement remote control

## Current Focus

1. UI Modernization:
   - [ ] Improve responsive layout
   - [ ] Add real-time parameter effects preview
   - [ ] Implement interactive plot controls
2. Configuration System:
   - [ ] Align config schema with model parameters
   - [ ] Add config validation
   - [ ] Implement config migration

## Future Considerations

- Web interface using Pyodide
- Cloud integration for distributed computing
- Real-time collaboration features
- Plugin marketplace
- Machine learning integration for parameter optimization
