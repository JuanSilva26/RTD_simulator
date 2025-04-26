"""
Curve analysis utilities for RTD models.
"""

from typing import Dict, Any, Tuple, List, Union, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks, peak_widths, peak_prominences
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
        
        # Ensure indices are valid
        valid_peaks = [p for p in peaks if p < len(v)]
        valid_valleys = [val for val in valleys if val < len(v)]
        
        return {
            'peaks': [(float(v[p]), float(i[p])) for p in valid_peaks],
            'valleys': [(float(v[val]), float(i[val])) for val in valid_valleys]
        }
    
    def advanced_peak_detection(
            self, 
            x: NDArray, 
            y: NDArray,
            height: Optional[float] = None,
            threshold: Optional[float] = None,
            distance: Optional[int] = None,
            prominence: Optional[float] = None, 
            width: Optional[float] = None,
            wlen: Optional[int] = None,
            rel_height: float = 0.5,
            plateau_size: Optional[float] = None
        ) -> Dict[str, Any]:
        """
        Advanced peak detection with detailed peak properties.
        
        Args:
            x: X-axis values (e.g., voltage)
            y: Y-axis values (e.g., current)
            height: Required height of peaks (absolute height)
            threshold: Required threshold of peaks relative to neighboring samples
            distance: Required minimal horizontal distance between neighboring peaks
            prominence: Required prominence of peaks
            width: Required width of peaks in samples
            wlen: Window length for prominence calculation
            rel_height: Relative height for width calculation (0-1)
            plateau_size: Size of flat plateau tops to consider as peaks
            
        Returns:
            Dictionary containing comprehensive peak analysis results
        """
        # Ensure arrays are numpy arrays
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        
        # Calculate y range for scaling
        y_range = np.max(y_arr) - np.min(y_arr)
        
        # Determine appropriate default values for peaks if not provided
        if height is None:
            height = 0.01 * y_range  # 1% of range
        
        if distance is None:
            distance = max(len(y_arr) // 50, 3)  # 2% of data length or at least 3 points
            
        if prominence is None:
            prominence = 0.01 * y_range  # 1% of range
            
        # Find peaks with configurable parameters
        peaks, peak_props = find_peaks(
            y_arr, 
            height=height,
            threshold=threshold,
            distance=distance,
            prominence=prominence,
            width=width,
            wlen=wlen,
            plateau_size=plateau_size
        )
        
        # Use much more sensitive parameters for valley detection
        valley_height = None  # Remove height restriction for valleys
        valley_prominence = 0.001 * y_range if prominence is None else prominence/10  # Much more sensitive prominence
        
        # Find valleys (using -y to find peaks)
        valleys, valley_props = find_peaks(
            -y_arr, 
            height=valley_height,
            threshold=threshold,
            distance=distance,
            prominence=valley_prominence,
            width=width,
            wlen=wlen,
            plateau_size=plateau_size
        )
        
        # Compute peak widths at specified relative height
        if len(peaks) > 0:
            widths, width_heights, left_ips, right_ips = peak_widths(
                y_arr, peaks, rel_height=rel_height
            )
            
            # Convert x-indices to x-values for left and right width positions
            left_bases = np.interp(left_ips, np.arange(len(x_arr)), x_arr)
            right_bases = np.interp(right_ips, np.arange(len(x_arr)), x_arr)
            
            # Calculate actual widths in x units (instead of samples)
            x_widths = right_bases - left_bases
        else:
            widths = np.array([])
            width_heights = np.array([])
            left_bases = np.array([])
            right_bases = np.array([])
            x_widths = np.array([])
        
        # Calculate derivatives for peak identification
        dy_dx = np.gradient(y_arr, x_arr, edge_order=2)
        d2y_dx2 = np.gradient(dy_dx, x_arr, edge_order=2)
        
        # Identify sharp vs. rounded peaks based on second derivative
        peak_sharpness = []
        if len(peaks) > 0:
            for peak in peaks:
                # Calculate average second derivative around peak (negative means sharp)
                window_size = max(1, min(5, peak, len(d2y_dx2) - peak - 1))
                if window_size > 0 and peak < len(d2y_dx2):
                    avg_d2 = np.mean(d2y_dx2[peak-window_size:peak+window_size+1])
                    peak_sharpness.append(float(avg_d2))
                else:
                    peak_sharpness.append(0.0)
        
        # Prepare results with peak properties
        peak_results = []
        if len(peaks) > 0:
            for i, peak in enumerate(peaks):
                if peak < len(x_arr) and peak < len(y_arr):
                    result = {
                        'x': float(x_arr[peak]),
                        'y': float(y_arr[peak]),
                        'index': int(peak),
                        'prominence': float(peak_props['prominences'][i]) if 'prominences' in peak_props and i < len(peak_props['prominences']) else None,
                        'width': float(widths[i]) if i < len(widths) else None,
                        'width_x_units': float(x_widths[i]) if i < len(x_widths) else None,
                        'width_left': float(left_bases[i]) if i < len(left_bases) else None,
                        'width_right': float(right_bases[i]) if i < len(right_bases) else None,
                        'height': float(peak_props['peak_heights'][i]) if 'peak_heights' in peak_props and i < len(peak_props['peak_heights']) else None,
                        'sharpness': peak_sharpness[i] if i < len(peak_sharpness) else None,
                    }
                    peak_results.append(result)
        
        # Prepare results with valley properties
        valley_results = []
        if len(valleys) > 0:
            for i, valley in enumerate(valleys):
                if valley < len(x_arr) and valley < len(y_arr):
                    result = {
                        'x': float(x_arr[valley]),
                        'y': float(y_arr[valley]),
                        'index': int(valley),
                        'prominence': float(valley_props['prominences'][i]) if 'prominences' in valley_props and i < len(valley_props['prominences']) else None,
                        'height': float(-valley_props['peak_heights'][i]) if 'peak_heights' in valley_props and i < len(valley_props['peak_heights']) else None,
                    }
                    valley_results.append(result)
        
        # Calculate peak-to-valley ratios if we have both
        peak_to_valley_ratios = []
        if len(peak_results) > 0 and len(valley_results) > 0:
            # Find pairs of peaks and valleys
            for i, peak in enumerate(peak_results):
                # Find the closest valley to the right of this peak
                right_valleys = [v for v in valley_results if v['index'] > peak['index']]
                if right_valleys:
                    closest_valley = min(right_valleys, key=lambda v: v['index'] - peak['index'])
                    ratio = abs(peak['y'] / closest_valley['y']) if closest_valley['y'] != 0 else float('inf')
                    peak_to_valley_ratios.append({
                        'peak_index': peak['index'],
                        'valley_index': closest_valley['index'],
                        'peak_x': peak['x'],
                        'valley_x': closest_valley['x'],
                        'ratio': ratio
                    })
        
        return {
            'peaks': peak_results,
            'valleys': valley_results,
            'peak_to_valley_ratios': peak_to_valley_ratios,
            'dy_dx': dy_dx,  # First derivative
            'd2y_dx2': d2y_dx2,  # Second derivative
            'peak_count': len(peak_results),
            'valley_count': len(valley_results),
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
    
    def calculate_statistics(self, x: NDArray, y: NDArray) -> Dict[str, float]:
        """
        Calculate statistical properties of x and y data.
        
        Args:
            x: Array of x values (e.g., voltage)
            y: Array of y values (e.g., current)
            
        Returns:
            Dictionary containing statistical analysis
        """
        return {
            'x_mean': float(np.mean(x)),
            'x_std': float(np.std(x)),
            'x_min': float(np.min(x)),
            'x_max': float(np.max(x)),
            'y_mean': float(np.mean(y)),
            'y_std': float(np.std(y)),
            'y_min': float(np.min(y)),
            'y_max': float(np.max(y)),
            'peak_to_peak_x': float(np.max(x) - np.min(x)),
            'peak_to_peak_y': float(np.max(y) - np.min(y)),
        } 