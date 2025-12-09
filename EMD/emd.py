import numpy as np
from PyEMD import EMD
import warnings

class EMDDenoise:
    """
    Signal denoising class based on Empirical Mode Decomposition (EMD).
    """
    
    def __init__(self, imf_threshold=0.3, noise_modes='auto', max_modes=None):
        """
        Initialize EMD denoiser.
        
        Args:
            imf_threshold: Energy threshold ratio for determining if an IMF contains noise.
            noise_modes: Number of modes to remove, can be an integer or 'auto'.
            max_modes: Maximum number of modes for EMD decomposition, None means no limit.
        """
        self.imf_threshold = imf_threshold
        self.noise_modes = noise_modes
        self.max_modes = max_modes
        self.emd = EMD()
        
        # Set EMD parameters
        if max_modes is not None:
            self.emd.MAX_ITERATION = max_modes * 10  # Set maximum iterations
        
        # Ignore warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
    def process(self, signal, signal_idx=None):
        """
        Perform EMD denoising on the input signal.
        
        Args:
            signal: Input noisy signal.
            signal_idx: Signal index (for logging).
            
        Returns:
            Denoised signal.
        """
        # Ensure signal is a 1D array
        signal = np.ravel(signal)
        
        # Apply EMD decomposition
        try:
            imfs = self.emd(signal)
        except Exception as e:
            print(f"Warning: EMD decomposition failed for signal {signal_idx}: {e}")
            return signal  # Return original signal if decomposition fails
        
        # Determine the number of noise modes to remove
        if self.noise_modes == 'auto':
            # Automatically determine the number of noise modes
            noise_modes = self._auto_select_noise_modes(imfs, signal)
        else:
            # Use user-specified number of modes
            noise_modes = min(int(self.noise_modes), len(imfs))
        
        # Reconstruct signal, removing noise modes
        if noise_modes > 0 and noise_modes < len(imfs):
            # Remove high-frequency noise modes
            denoised_signal = np.sum(imfs[noise_modes:], axis=0)
        else:
            # Return original signal if no noise modes are identified
            denoised_signal = signal
            
        # if signal_idx is not None and signal_idx % 50 == 0:
        #     print(f"Signal {signal_idx} EMD decomposition obtained {len(imfs)} IMFs, removed {noise_modes} noise modes")
            
        return denoised_signal
    
    def _auto_select_noise_modes(self, imfs, original_signal):
        """
        Automatically select the number of noise modes to remove.
        Based on energy distribution and correlation analysis.
        
        Args:
            imfs: Decomposed IMFs.
            original_signal: Original signal.
            
        Returns:
            Number of modes to remove.
        """
        # Calculate energy of each IMF
        imf_energies = np.array([np.sum(imf**2) for imf in imfs])
        total_energy = np.sum(imf_energies)
        
        # Calculate energy ratio
        energy_ratio = imf_energies / total_energy
        
        # Cumulative energy
        cum_energy = np.cumsum(energy_ratio)
        
        # Select noise modes based on energy threshold
        noise_modes = np.sum(cum_energy < self.imf_threshold)
        
        # Ensure at least one mode is selected as noise (if there are multiple IMFs)
        if noise_modes == 0 and len(imfs) > 1:
            noise_modes = 1
            
        # At most remove len(imfs)-1 modes
        if noise_modes >= len(imfs):
            noise_modes = len(imfs) - 1
            
        return noise_modes 