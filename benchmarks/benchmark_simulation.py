"""
Benchmark suite for RTD simulation performance.

This module provides comprehensive benchmarks for measuring the performance
of different simulation methods and configurations.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path

from rtd_simulator.models.rtd_model import SimplifiedRTDModel, SchulmanRTDModel
from rtd_simulator.models.base import RTDModel

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method: str
    model_type: str
    parameters: Dict[str, float]
    execution_time: float
    memory_usage: float
    error: Optional[str] = None

def benchmark_simulation(
    model: RTDModel,
    t_end: float,
    dt: float,
    vbias: float,
    methods: List[str] = ["original", "vectorized", "numba"],
    num_runs: int = 5
) -> List[BenchmarkResult]:
    """
    Benchmark different simulation methods.
    
    Args:
        model: RTD model to benchmark
        t_end: End time for simulation
        dt: Time step
        vbias: Bias voltage
        methods: List of methods to benchmark
        num_runs: Number of runs per method
        
    Returns:
        List of benchmark results
    """
    results = []
    
    for method in methods:
        # Skip if method not available
        if method == "numba" and not hasattr(model, "simulate_vectorized"):
            continue
            
        # Run benchmark
        for _ in range(num_runs):
            start_time = time.time()
            try:
                if method == "original":
                    t, v, i = model.simulate(t_end, dt, vbias, v0=0.0, i0=0.0)
                elif method == "vectorized":
                    t, v, i = model.simulate_vectorized(t_end, dt, vbias, v0=0.0, i0=0.0)
                elif method == "numba":
                    t, v, i = model.simulate_vectorized(t_end, dt, vbias, v0=0.0, i0=0.0)
                else:
                    raise ValueError(f"Unknown method: {method}")
                    
                execution_time = time.time() - start_time
                
                # Estimate memory usage (rough approximation)
                memory_usage = (t.nbytes + v.nbytes + i.nbytes) / 1024  # KB
                
                results.append(BenchmarkResult(
                    method=method,
                    model_type=model.__class__.__name__,
                    parameters=model.parameters,
                    execution_time=execution_time,
                    memory_usage=memory_usage
                ))
                
            except Exception as e:
                results.append(BenchmarkResult(
                    method=method,
                    model_type=model.__class__.__name__,
                    parameters=model.parameters,
                    execution_time=0.0,
                    memory_usage=0.0,
                    error=str(e)
                ))
    
    return results

def benchmark_caching(
    model: RTDModel,
    voltage_range: np.ndarray,
    num_runs: int = 5
) -> List[BenchmarkResult]:
    """
    Benchmark IV-curve caching performance.
    
    Args:
        model: RTD model to benchmark
        voltage_range: Array of voltage values
        num_runs: Number of runs
        
    Returns:
        List of benchmark results
    """
    results = []
    
    # First run (no cache)
    model.disable_cache()
    for _ in range(num_runs):
        start_time = time.time()
        try:
            current = model.iv_characteristic(voltage_range)
            execution_time = time.time() - start_time
            memory_usage = current.nbytes / 1024  # KB
            
            results.append(BenchmarkResult(
                method="no_cache",
                model_type=model.__class__.__name__,
                parameters=model.parameters,
                execution_time=execution_time,
                memory_usage=memory_usage
            ))
        except Exception as e:
            results.append(BenchmarkResult(
                method="no_cache",
                model_type=model.__class__.__name__,
                parameters=model.parameters,
                execution_time=0.0,
                memory_usage=0.0,
                error=str(e)
            ))
    
    # Subsequent runs (with cache)
    model.enable_cache()
    for _ in range(num_runs):
        start_time = time.time()
        try:
            current = model.iv_characteristic(voltage_range)
            execution_time = time.time() - start_time
            memory_usage = current.nbytes / 1024  # KB
            
            results.append(BenchmarkResult(
                method="with_cache",
                model_type=model.__class__.__name__,
                parameters=model.parameters,
                execution_time=execution_time,
                memory_usage=memory_usage
            ))
        except Exception as e:
            results.append(BenchmarkResult(
                method="with_cache",
                model_type=model.__class__.__name__,
                parameters=model.parameters,
                execution_time=0.0,
                memory_usage=0.0,
                error=str(e)
            ))
    
    return results

def plot_benchmark_results(
    results: List[BenchmarkResult],
    output_dir: str = "benchmark_results"
) -> None:
    """
    Plot and save benchmark results.
    
    Args:
        results: List of benchmark results
        output_dir: Directory to save plots
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Group results by model type
    model_results: Dict[str, List[BenchmarkResult]] = {}
    for result in results:
        if result.model_type not in model_results:
            model_results[result.model_type] = []
        model_results[result.model_type].append(result)
    
    # Plot execution time comparison
    plt.figure(figsize=(12, 6))
    
    for model_type, model_results_list in model_results.items():
        # Group by method
        method_times: Dict[str, List[float]] = {}
        for result in model_results_list:
            if result.error is None:
                if result.method not in method_times:
                    method_times[result.method] = []
                method_times[result.method].append(result.execution_time)
        
        # Calculate mean and std for each method
        methods = list(method_times.keys())
        means = [np.mean(method_times[m]) for m in methods]
        stds = [np.std(method_times[m]) for m in methods]
        
        # Plot
        x = np.arange(len(methods))
        plt.bar(x, means, yerr=stds, label=model_type)
        plt.xticks(x, methods)
    
    plt.xlabel("Simulation Method")
    plt.ylabel("Execution Time (s)")
    plt.title("Simulation Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path / "execution_time.png")
    plt.close()
    
    # Plot memory usage comparison
    plt.figure(figsize=(12, 6))
    
    for model_type, model_results_list in model_results.items():
        # Group by method
        method_memory: Dict[str, List[float]] = {}
        for result in model_results_list:
            if result.error is None:
                if result.method not in method_memory:
                    method_memory[result.method] = []
                method_memory[result.method].append(result.memory_usage)
        
        # Calculate mean and std for each method
        methods = list(method_memory.keys())
        means = [np.mean(method_memory[m]) for m in methods]
        stds = [np.std(method_memory[m]) for m in methods]
        
        # Plot
        x = np.arange(len(methods))
        plt.bar(x, means, yerr=stds, label=model_type)
        plt.xticks(x, methods)
    
    plt.xlabel("Simulation Method")
    plt.ylabel("Memory Usage (KB)")
    plt.title("Memory Usage Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path / "memory_usage.png")
    plt.close()

def main():
    """Run all benchmarks."""
    # Create models
    simplified_model = SimplifiedRTDModel()
    schulman_model = SchulmanRTDModel()
    
    # Benchmark simulation methods
    print("Benchmarking simulation methods...")
    sim_results = []
    
    # Simplified model benchmarks
    sim_results.extend(benchmark_simulation(
        simplified_model,
        t_end=1.0,
        dt=0.01,
        vbias=1.0
    ))
    
    # Schulman model benchmarks
    sim_results.extend(benchmark_simulation(
        schulman_model,
        t_end=1e-6,
        dt=1e-9,
        vbias=0.1
    ))
    
    # Benchmark caching
    print("Benchmarking caching...")
    cache_results = []
    
    # Simplified model caching
    voltage_range = np.linspace(-3.0, 3.0, 1000)
    cache_results.extend(benchmark_caching(simplified_model, voltage_range))
    
    # Schulman model caching
    voltage_range = np.linspace(-0.5, 0.5, 1000)
    cache_results.extend(benchmark_caching(schulman_model, voltage_range))
    
    # Plot results
    print("Plotting results...")
    plot_benchmark_results(sim_results + cache_results)
    
    print("Benchmarking complete!")

if __name__ == "__main__":
    main() 