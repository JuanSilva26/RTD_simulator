"""
Curve analysis utilities for RTD models.
"""

from typing import Dict, Any, Tuple, List, Union
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

class CurveAnalyzer:
    """Analyzes IV curves and extracts key parameters."""
    
    def __init__(self):
        """Initialize the curve analyzer."""
        pass
        
    def analyze_iv_curve(self, v_range: NDArray, i_values: NDArray) -> Dict[str, Any]:
        """
        Analyze an IV curve and extract key parameters.
        
        Args:
            v_range: Array of voltage values
            i_values: Array of current values
            
        Returns:
            Dictionary containing analysis results
        """
        # Find peak and valley points
        peak_idx = np.argmax(i_values)
        valley_idx = np.argmin(i_values[peak_idx:]) + peak_idx
        
        # Calculate peak-to-valley ratio
        pvr = i_values[peak_idx] / i_values[valley_idx]
        
        # Calculate differential resistance
        dv = np.diff(v_range)
        di = np.diff(i_values)
        dr = dv / di
        
        # Find negative differential resistance region
        ndr_mask = dr < 0
        ndr_start = np.argmax(ndr_mask)
        ndr_end = len(dr) - np.argmax(ndr_mask[::-1])
        
        return {
            'peak_voltage': v_range[peak_idx],
            'peak_current': i_values[peak_idx],
            'valley_voltage': v_range[valley_idx],
            'valley_current': i_values[valley_idx],
            'peak_to_valley_ratio': pvr,
            'ndr_region': (v_range[ndr_start], v_range[ndr_end]),
            'differential_resistance': dr
        }
        
    def find_oscillation_points(self, v: NDArray, i: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Find points where the system is likely to oscillate.
        
        Args:
            v: Array of voltage values
            i: Array of current values
            
        Returns:
            Tuple of (voltage_points, current_points) where oscillations occur
        """
        # Calculate second derivative
        d2i = np.gradient(np.gradient(i, v), v)
        
        # Find points where second derivative is large
        threshold = np.std(d2i) * 2
        oscillation_mask = np.abs(d2i) > threshold
        
        return v[oscillation_mask], i[oscillation_mask]
        
    def fit_iv_curve(self, v: NDArray, i: NDArray) -> Dict[str, Any]:
        """
        Fit a polynomial to the IV curve.
        
        Args:
            v: Array of voltage values
            i: Array of current values
            
        Returns:
            Dictionary containing fit parameters
        """
        # Fit a 4th degree polynomial
        coeffs = np.polyfit(v, i, 4)
        return {
            'coefficients': coeffs,
            'polynomial': np.poly1d(coeffs)
        }
        
    def find_peaks_and_valleys(self, v: NDArray, i: NDArray) -> Dict[str, List[Tuple[float, float]]]:
        """
        Find all peaks and valleys in the IV curve.
        
        Args:
            v: Array of voltage values
            i: Array of current values
            
        Returns:
            Dictionary containing lists of peak and valley points
        """
        # Find peaks
        peaks, _ = find_peaks(i)
        valleys, _ = find_peaks(-i)
        
        return {
            'peaks': [(float(v[p]), float(i[p])) for p in peaks],
            'valleys': [(float(v[v]), float(i[v])) for v in valleys]
        }
        
    def analyze_frequency(self, t: NDArray, signal: NDArray) -> Dict[str, Any]:
        """
        Analyze the frequency content of a signal.
        
        Args:
            t: Array of time values
            signal: Array of signal values
            
        Returns:
            Dictionary containing frequency analysis results
        """
        # Calculate sampling frequency
        dt = float(t[1] - t[0])
        fs = 1.0 / dt
        
        # Compute FFT
        n = len(signal)
        yf = fft(signal)
        xf = fftfreq(n, dt)
        
        # Find dominant frequency
        idx = np.argmax(np.abs(yf))
        dominant_freq = float(abs(xf[idx]))
        
        return {
            'frequencies': xf,
            'amplitudes': np.abs(yf),
            'dominant_frequency': dominant_freq,
            'sampling_frequency': fs
        } 