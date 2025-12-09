import numpy as np
from PyEMD import EEMD
import warnings

class EEMDDenoise:
    """
    Signal denoising class based on Ensemble Empirical Mode Decomposition (EEMD).
    EEMD solves the mode mixing problem of basic EMD by adding white noise, providing better denoising for non-stationary signals like microseismic signals.
    """
    
    def __init__(self, imf_threshold=0.8, noise_modes=1, max_modes=None, 
                 noise_width=0.05, trials=100, random_seed=42, selection_method='correlation'):
        """
        Initialize EEMD denoiser.
        
        Args:
            imf_threshold: Energy threshold ratio for determining if an IMF contains noise.
            noise_modes: Number of modes to remove, can be an integer or 'auto'.
            max_modes: Maximum number of modes for EEMD decomposition, None means no limit.
            noise_width: Intensity of added white noise, typically between 0.05-0.2.
            trials: Number of ensemble trials for EEMD, larger values give more stable results but increase computation.
            random_seed: Random seed for reproducibility.
            selection_method: IMF selection method, 'energy' (energy-based) or 'correlation' (correlation-based).
        """
        self.imf_threshold = imf_threshold
        self.noise_modes = noise_modes
        self.max_modes = max_modes
        self.noise_width = noise_width
        self.trials = trials
        self.random_seed = random_seed
        self.selection_method = selection_method
        
        # Create EEMD instance
        self.eemd = EEMD(trials=self.trials)
        self.eemd.noise_seed(self.random_seed)
        self.eemd.noise_width = self.noise_width
        
        # Configure EMD parameters (EMD used internally by EEMD)
        if max_modes is not None:
            self.eemd.EMD.MAX_ITERATION = max_modes * 10  # Set maximum iterations
        
        # Set extrema detection method to parabolic, which usually yields more accurate results
        self.eemd.EMD.extrema_detection = "parabol"
        
        # Ignore warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
    def process(self, signal, signal_idx=None):
        """
        Perform EEMD denoising on the input signal.
        
        Args:
            signal: Input noisy signal.
            signal_idx: Signal index (for logging).
            
        Returns:
            Denoised signal.
        """
        # Ensure signal is a 1D array
        signal = np.ravel(signal)
        
        # Apply EEMD decomposition
        try:
            imfs = self.eemd.eemd(signal)
        except Exception as e:
            print(f"Warning: EEMD decomposition failed for signal {signal_idx}: {e}")
            return signal  # Return original signal if decomposition fails
        
        # Select IMFs to retain
        if self.selection_method == 'correlation':
            # Select IMFs based on correlation
            denoised_signal = self._select_by_correlation(imfs, signal, signal_idx)
        else:
            # Select IMFs based on energy (default)
            denoised_signal = self._select_by_energy(imfs, signal, signal_idx)
            
        return denoised_signal
    
    def _select_by_energy(self, imfs, original_signal, signal_idx=None):
        """
        Select IMF components based on energy threshold.
        
        Args:
            imfs: Decomposed IMF components.
            original_signal: Original signal.
            signal_idx: Signal index (for logging).
            
        Returns:
            Reconstructed signal.
        """
        # Determine the number of noise modes to remove
        if self.noise_modes == 'auto':
            # Automatically determine the number of noise modes
            noise_modes = self._auto_select_noise_modes(imfs, original_signal)
        else:
            # Use user-specified number of modes
            noise_modes = min(int(self.noise_modes), len(imfs))
        
        # Reconstruct signal, removing noise modes
        if noise_modes > 0 and noise_modes < len(imfs):
            # Remove high-frequency noise modes
            denoised_signal = np.sum(imfs[noise_modes:], axis=0)
        else:
            # Return original signal if no noise modes are identified
            denoised_signal = original_signal
        
        if signal_idx is not None and signal_idx % 50 == 0:
            print(f"Signal {signal_idx} EEMD decomposition obtained {len(imfs)} IMFs, removed {noise_modes} noise modes")
            
        return denoised_signal
    
    def _select_by_correlation(self, imfs, original_signal, signal_idx=None):
        """
        Select IMF components based on correlation.
        
        Args:
            imfs: Decomposed IMF components.
            original_signal: Original signal.
            signal_idx: Signal index (for logging).
            
        Returns:
            Reconstructed signal.
        """
        # Calculate correlation between each IMF and the original signal
        correlations = []
        for imf in imfs:
            corr = np.corrcoef(original_signal, imf)[0, 1]
            correlations.append(abs(corr))  # Use absolute value of correlation
        
        # Select IMFs with correlation greater than threshold (threshold set to 0.8 * mean correlation)
        threshold = np.mean(correlations) * 0.8  # Lower threshold to retain more relevant IMFs
        selected_imfs = []
        
        for i, corr in enumerate(correlations):
            # Skip the first IMF as it usually contains high-frequency noise
            if i == 0 and self.noise_modes != 'auto':
                continue
                
            if corr > threshold:
                selected_imfs.append(imfs[i])
        
        # If no IMF is selected, select the one with the highest correlation (except the first one)
        if not selected_imfs and len(imfs) > 1:
            # Find the highest correlation starting from the second IMF
            if self.noise_modes != 'auto' and len(imfs) > 1:
                corr_slice = correlations[1:]
                max_idx = np.argmax(corr_slice) + 1  # Add 1 because we skipped the first IMF
            else:
                max_idx = np.argmax(correlations)
            selected_imfs.append(imfs[max_idx])
        
        # Reconstruct signal
        if selected_imfs:
            denoised_signal = np.sum(selected_imfs, axis=0)
        else:
            denoised_signal = original_signal
        
        if signal_idx is not None and signal_idx % 50 == 0:
            print(f"Signal {signal_idx} EEMD decomposition obtained {len(imfs)} IMFs, selected {len(selected_imfs)} IMFs based on correlation")
        
        return denoised_signal
    
    def _auto_select_noise_modes(self, imfs, original_signal):
        """
        Automatically select the number of noise modes to remove.
        Based on energy distribution analysis.
        
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
        
    def get_imfs(self, signal):
        """
        Return IMF decomposition results of the signal for visualization analysis.
        
        Args:
            signal: Input signal.
            
        Returns:
            Array of IMF components.
        """
        signal = np.ravel(signal)
        try:
            imfs = self.eemd.eemd(signal)
            return imfs
        except Exception as e:
            print(f"Warning: EEMD decomposition failed for signal: {e}")
            return np.array([signal])  # Return original signal as a single IMF 