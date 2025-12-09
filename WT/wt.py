import numpy as np
import pywt
from scipy import signal
import time

class AIC_Arrival:
    """Python implementation of AIC algorithm for P-wave arrival time detection"""
    
    def __init__(self):
        self.arrival_time = -1
    
    def process(self, input_signal, index=-1):
        """
        Process single component signal to find arrival time using AIC algorithm
        
        Args:
            input_signal: 1D numpy array containing signal data
            index: optional signal index for logging
            
        Returns:
            arrival time index or -1 if detection fails
        """
        start_time = time.time()
        
        if len(input_signal) == 0:
            print(f"[Signal index: {index}] | [Input signal is empty]")
            return -1
        
        self.arrival_time = -1
        
        # Data Preprocessing
        x = np.array(input_signal.copy())
        
        # Remove median
        median = np.median(x)
        x -= median
        
        # Handle peak mode (find maximum amplitude and truncate signal there)
        mode = 1  # "to_peak" mode
        if mode == 1:
            peak_idx = np.argmax(np.abs(x))
            x = x[:peak_idx+1]
        
        # AIC Calculation Core
        n = len(x)
        if n < 2:
            print(f"[Signal index: {index}] | [Signal too short after preprocessing]")
            return -1
            
        aic = np.zeros(n-1)
        
        for k in range(n-1):
            # First segment
            if k > 0:
                var1 = np.var(x[:k+1], ddof=1)  # Use N-1 denominator
            else:
                var1 = 0
                
            # Second segment
            if n-k-1 > 1:
                var2 = np.var(x[k+1:], ddof=1)  # Use N-1 denominator
            else:
                var2 = 0
                
            # Calculate AIC value
            aic[k] = (k+1) * (np.log(var1) if var1 > 0 else 0) + \
                     (n-k-1) * (np.log(var2) if var2 > 0 else 0)
        
        # Result Processing
        if len(aic) > 0:
            ind = np.argmin(aic)
            self.arrival_time = ind if ind < len(x) else -1
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print(f"[Signal index: {index}] | [Processed] Time cost: {elapsed_ms:.1f} ms | Arrival time: {self.arrival_time}")
        
        return self.arrival_time


class WaveletDenoise:
    """Python implementation of wavelet denoising algorithm"""
    
    # Threshold types
    HARD = 1
    SOFT = 2
    SEMISOFT = 3
    
    def __init__(self, wavelet_name="db4", level=8, threshold_type=SEMISOFT):
        """
        Initialize wavelet denoising with specified parameters
        
        Args:
            wavelet_name: wavelet to use (default: 'db4')
            level: decomposition level (default: 8)
            threshold_type: thresholding method (HARD, SOFT, or SEMISOFT)
        """
        self.wavelet_name = wavelet_name
        self.level = level
        self.threshold_type = threshold_type
    
    def process(self, input_signal, index=-1):
        """
        Apply wavelet denoising to the input signal
        
        Args:
            input_signal: 1D numpy array containing signal data
            index: optional signal index for logging
            
        Returns:
            denoised signal as numpy array
        """
        start_time = time.time()
        
        if len(input_signal) == 0:
            print(f"[Signal index: {index}] | [Input signal is empty]")
            return input_signal
        
        # Create a copy to avoid modifying the original
        output = np.array(input_signal).copy()
        signal_length = len(output)
        
        # Calculate maximum possible decomposition level
        max_possible_level = int(np.log2(signal_length))
        actual_level = min(self.level, max_possible_level)
        
        if actual_level < 1:
            print(f"[Signal index: {index}] | Invalid decomposition level.")
            return output
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(output, self.wavelet_name, level=actual_level)
        
        # Skip approximation coefficients (first element)
        for i in range(1, len(coeffs)):
            # Calculate threshold for this level
            detail_coeffs = coeffs[i]
            threshold = self.calculate_threshold(detail_coeffs)
            
            # Apply thresholding
            coeffs[i] = self.apply_thresholding(detail_coeffs, threshold)
        
        # Reconstruct the signal
        output = pywt.waverec(coeffs, self.wavelet_name)
        
        # Ensure the result has the same length as input
        if len(output) > signal_length:
            output = output[:signal_length]
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        # print(f"[Signal index: {index}] | Processed. Time: {elapsed_ms:.1f} ms | Length: {len(output)}")
        
        return output
    
    def calculate_threshold(self, coeffs):
        """Calculate threshold using median absolute deviation (MAD)"""
        if len(coeffs) == 0:
            return 0
            
        # Calculate median
        median = np.median(coeffs)
        
        # Calculate median absolute deviation
        mad = np.median(np.abs(coeffs - median))
        
        # Estimate noise level
        sigma = mad / 0.6745
        
        # Calculate threshold (universal threshold)
        threshold = sigma * np.sqrt(2 * np.log(len(coeffs)))
        
        return threshold
    
    def apply_thresholding(self, coeffs, threshold):
        """Apply thresholding to wavelet coefficients"""
        result = coeffs.copy()
        
        if self.threshold_type == self.HARD:
            # Hard thresholding
            result[np.abs(result) <= threshold] = 0
            
        elif self.threshold_type == self.SOFT:
            # Soft thresholding
            result = np.sign(result) * np.maximum(np.abs(result) - threshold, 0)
            
        elif self.threshold_type == self.SEMISOFT:
            # Semi-soft thresholding (using logistic function for smooth transition)
            abs_val = np.abs(result)
            mask = abs_val > threshold
            
            # For values above threshold
            alpha = 1.0 / (1.0 + np.exp(-(abs_val[mask] - threshold) / threshold))
            result[mask] = np.sign(result[mask]) * (abs_val[mask] - threshold) / (1 + alpha * threshold)
            
            # For values below or equal to threshold
            result[~mask] = 0
            
        else:
            print("Invalid threshold type")
        
        return result
    
    def set_parameters(self, level=None, threshold_type=None, wavelet_name=None):
        """Update parameters of the denoiser"""
        if level is not None:
            self.level = level
        if threshold_type is not None:
            self.threshold_type = threshold_type
        if wavelet_name is not None:
            self.wavelet_name = wavelet_name 