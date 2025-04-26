"""
Analysis utilities for RTD curves.

This module provides analysis tools for RTD IV curves and time series data,
including peak detection, curve fitting, and frequency analysis.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

class CurveAnalyzer:
    """Analyzer class for RTD curves.
    
    This class provides static methods for analyzing RTD IV curves and time series data,
    including peak detection, curve fitting, and statistical analysis.
    """
    
    @staticmethod
    def find_peaks_and_valleys(v: NDArray[np.float64], 
                             i: NDArray[np.float64]) -> Dict[str, NDArray[np.float64]]:
        """Find peaks and valleys in the IV curve.
        
        Args:
            v: Voltage array
            i: Current array
            
        Returns:
            Dictionary containing:
                - peak_indices: Indices of peaks
                - peak_voltages: Voltage values at peaks
                - peak_currents: Current values at peaks
                - valley_indices: Indices of valleys
                - valley_voltages: Voltage values at valleys
                - valley_currents: Current values at valleys
        """
        # Find peaks (local maxima)
        peak_indices, _ = find_peaks(i)
        
        # Find valleys (local minima)
        valley_indices, _ = find_peaks(-i)
        
        return {
            'peak_indices': peak_indices,
            'peak_voltages': v[peak_indices],
            'peak_currents': i[peak_indices],
            'valley_indices': valley_indices,
            'valley_voltages': v[valley_indices],
            'valley_currents': i[valley_indices]
        }
    
    @staticmethod
    def fit_iv_curve(v: NDArray[np.float64], 
                     i: NDArray[np.float64]) -> Tuple[NDArray[np.float64], Dict[str, float]]:
        """Fit IV curve using a polynomial function.
        
        Args:
            v: Voltage array
            i: Current array
            
        Returns:
            Tuple containing:
                - Fitted current array
                - Dictionary of fitting parameters:
                    - a: Cubic term coefficient
                    - b: Quadratic term coefficient
                    - c: Linear term coefficient
                    - d: Constant term
                    - r_squared: R-squared value of fit
        """
        # Define polynomial fitting function (3rd order)
        def poly_func(x: float, a: float, b: float, c: float, d: float) -> float:
            return a * x**3 + b * x**2 + c * x + d
        
        # Fit curve
        popt, _ = curve_fit(poly_func, v, i)
        
        # Generate fitted curve
        i_fit = poly_func(v, *popt)
        
        # Calculate R-squared
        residuals = i - i_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((i - np.mean(i))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Return fitted curve and parameters
        params = {
            'a': float(popt[0]),  # cubic term
            'b': float(popt[1]),  # quadratic term
            'c': float(popt[2]),  # linear term
            'd': float(popt[3]),  # constant term
            'r_squared': float(r_squared)
        }
        
        return i_fit, params
    
    @staticmethod
    def calculate_statistics(v: NDArray[np.float64], 
                           i: NDArray[np.float64]) -> Dict[str, float]:
        """Calculate basic statistics for the IV curve.
        
        Args:
            v: Voltage array
            i: Current array
            
        Returns:
            Dictionary containing statistical measures:
                - v_mean: Mean voltage
                - v_std: Voltage standard deviation
                - v_min: Minimum voltage
                - v_max: Maximum voltage
                - i_mean: Mean current
                - i_std: Current standard deviation
                - i_min: Minimum current
                - i_max: Maximum current
                - peak_to_peak_v: Peak-to-peak voltage
                - peak_to_peak_i: Peak-to-peak current
        """
        return {
            'v_mean': float(np.mean(v)),
            'v_std': float(np.std(v)),
            'v_min': float(np.min(v)),
            'v_max': float(np.max(v)),
            'i_mean': float(np.mean(i)),
            'i_std': float(np.std(i)),
            'i_min': float(np.min(i)),
            'i_max': float(np.max(i)),
            'peak_to_peak_v': float(np.max(v) - np.min(v)),
            'peak_to_peak_i': float(np.max(i) - np.min(i))
        }
    
    @staticmethod
    def analyze_frequency(t: NDArray[np.float64], 
                         signal: NDArray[np.float64]) -> Dict[str, Union[NDArray[np.float64], float]]:
        """Perform frequency analysis on time series data.
        
        Args:
            t: Time array
            signal: Signal array (voltage or current)
            
        Returns:
            Dictionary containing:
                - frequencies: Array of frequency values
                - magnitudes: Array of magnitude values
                - dominant_frequency: Frequency with highest magnitude
        """
        # Calculate sampling frequency and perform FFT
        fs = 1 / (t[1] - t[0])
        n = len(signal)
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n, 1/fs)
        
        # Get positive frequencies and their magnitudes
        pos_freq_mask = freqs >= 0
        frequencies = freqs[pos_freq_mask]
        magnitudes = np.abs(fft)[pos_freq_mask]
        
        # Find dominant frequency
        dominant_idx = np.argmax(magnitudes)
        dominant_freq = float(frequencies[dominant_idx])
        
        return {
            'frequencies': frequencies,
            'magnitudes': magnitudes,
            'dominant_frequency': dominant_freq
        } 