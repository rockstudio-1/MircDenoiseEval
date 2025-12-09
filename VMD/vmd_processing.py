import numpy as np
from vmd import vmd


class VMDDenoise:
    """
    Signal denoising class based on Variational Mode Decomposition (VMD)
    """
    
    def __init__(self, alpha=2000, tau=0, K=5, DC=False, init=1, tol=1e-7, 
                 is_multivariate=False, noise_modes='auto', mode_selection='energy'):
        """
        Initialize VMD denoiser
        
        Args:
            alpha: Bandwidth constraint parameter, larger value means narrower mode bandwidth
            tau: Dual ascent step size, usually 0 (noise slack)
            K: Number of modes to extract
            DC: Whether to fix the first mode at DC component (0 frequency)
            init: Center frequency initialization method (0=all zeros, 1=uniform distribution, 2=random distribution)
            tol: Convergence tolerance, usually around 1e-6
            is_multivariate: Whether it is multivariate signal processing
            noise_modes: Number of noise modes to remove, 'auto' for automatic determination
            mode_selection: Mode selection method, 'energy' based on energy, 'correlation' based on correlation
        """
        self.alpha = alpha
        self.tau = tau
        self.K = K
        self.DC = DC
        self.init = init
        self.tol = tol
        self.is_multivariate = is_multivariate
        self.noise_modes = noise_modes
        self.mode_selection = mode_selection
        
        # Store IMFs from last decomposition
        self.last_imfs = None
        
    def process(self, signal, signal_idx=None):
        """
        Process signal, apply VMD denoising
        
        Args:
            signal: Input signal
            signal_idx: Signal index (for logging)
            
        Returns:
            Denoised signal
        """
        # Ensure signal is an array
        signal = np.asarray(signal)
        
        # Check if it is a multivariate signal
        if self.is_multivariate and len(signal.shape) > 1:
            # Multivariate VMD decomposition
            u, _, _ = mvvmd(signal, self.alpha, self.tau, self.K, self.DC, self.init, self.tol)
        else:
            # Univariate VMD decomposition
            if len(signal.shape) > 1:
                signal = signal.flatten()  # Ensure it is a 1D signal
            u, _, _ = vmd(signal, self.alpha, self.tau, self.K, self.DC, self.init, self.tol)
        
        # Save IMFs for visualization
        self.last_imfs = u
        
        # Select IMFs based on mode selection strategy
        if self.mode_selection == 'correlation':
            return self._select_by_correlation(u, signal, signal_idx)
        else:  # Default use energy selection
            return self._select_by_energy(u, signal_idx)
    
    def _select_by_energy(self, imfs, signal_idx=None):
        """
        Select IMF components based on energy
        
        Args:
            imfs: Decomposed IMF components
            signal_idx: Signal index (for logging)
            
        Returns:
            Reconstructed signal
        """
        # Calculate energy of each IMF
        if self.is_multivariate:
            # Multivariate case, calculate average energy of each mode across all channels
            energies = np.zeros(self.K)
            for k in range(self.K):
                for c in range(imfs.shape[2]):
                    energies[k] += np.sum(imfs[k, :, c]**2)
            energies /= imfs.shape[2]  # Average over channels
        else:
            # Univariate case, calculate energy of each mode directly
            energies = np.array([np.sum(imf**2) for imf in imfs])
        
        # Calculate total energy
        total_energy = np.sum(energies)
        
        # Sort by energy (descending)
        sorted_indices = np.argsort(energies)[::-1]
        
        # Determine number of noise modes to remove
        if self.noise_modes == 'auto':
            # Auto determination: find point where cumulative energy reaches threshold
            cumulative_energy = 0
            threshold_ratio = 0.95  # Keep 95% of energy
            num_modes_to_keep = 0
            
            for idx in sorted_indices:
                cumulative_energy += energies[idx] / total_energy
                num_modes_to_keep += 1
                if cumulative_energy >= threshold_ratio:
                    break
            
            num_noise_modes = self.K - num_modes_to_keep
        else:
            # Use specified number of noise modes
            num_noise_modes = self.noise_modes
        
        # Ensure number of noise modes is reasonable
        num_noise_modes = max(0, min(num_noise_modes, self.K - 1))
        
        if signal_idx is not None:
            print(f"Signal {signal_idx}: Remove {num_noise_modes} noise modes, keep {self.K - num_noise_modes} modes")
        
        # Select modes to keep based on energy sorting
        modes_to_keep = sorted_indices[:self.K - num_noise_modes]
        
        # Reconstruct signal
        if self.is_multivariate:
            # Multivariate case
            reconstructed = np.zeros_like(imfs[0, :, :])
            for idx in modes_to_keep:
                reconstructed += imfs[idx, :, :]
        else:
            # Univariate case
            reconstructed = np.zeros_like(imfs[0, :])
            for idx in modes_to_keep:
                reconstructed += imfs[idx, :]
        
        return reconstructed
    
    def _select_by_correlation(self, imfs, original_signal, signal_idx=None):
        """
        Select IMF components based on correlation
        
        Args:
            imfs: Decomposed IMF components
            original_signal: Original signal
            signal_idx: Signal index (for logging)
            
        Returns:
            Reconstructed signal
        """
        if self.is_multivariate:
            # Multivariate case, calculate average correlation of each mode with each channel of original signal
            correlations = np.zeros(self.K)
            
            # Ensure imfs dimensions are correct
            if len(imfs.shape) == 3:  # K x L x C format
                num_channels = imfs.shape[2]
                
                for k in range(self.K):
                    # Determine how to extract channel signal based on original signal shape
                    for c in range(num_channels):
                        # Extract original signal of current channel
                        if len(original_signal.shape) == 2:
                            if original_signal.shape[0] == num_channels:
                                channel_signal = original_signal[c, :]  # C x L format
                            else:
                                channel_signal = original_signal[:, c]  # L x C format
                        else:
                            # If original signal is 1D, might be incorrect call, use same signal
                            channel_signal = original_signal
                        
                        # Ensure length matches
                        signal_len = min(len(channel_signal), imfs.shape[1])
                        channel_signal = channel_signal[:signal_len]
                        imf_channel = imfs[k, :signal_len, c]
                        
                        # Calculate correlation coefficient
                        try:
                            corr = np.corrcoef(channel_signal, imf_channel)[0, 1]
                            if not np.isnan(corr):
                                correlations[k] += abs(corr)  # Use absolute value of correlation
                        except:
                            # If calculation fails, skip this channel
                            if signal_idx is not None:
                                print(f"Warning: Signal {signal_idx}, Mode {k}, Channel {c} correlation calculation failed")
                
                # Average
                correlations /= num_channels
            else:
                # If imfs is not in expected 3D format, fallback to univariate processing
                if signal_idx is not None:
                    print(f"Warning: Signal {signal_idx} multivariate IMF format incorrect, using univariate processing")
                return self._select_by_correlation_single(imfs, original_signal, signal_idx)
        else:
            # Univariate case, call univariate processing function
            return self._select_by_correlation_single(imfs, original_signal, signal_idx)
        
        # Select IMFs with correlation greater than threshold
        threshold = np.mean(correlations) * 0.8  # Use 80% of average correlation as threshold
        
        # Select modes with correlation higher than threshold
        modes_to_keep = []
        for k in range(self.K):
            if correlations[k] > threshold:
                modes_to_keep.append(k)
        
        # If no mode selected, select the one with highest correlation
        if not modes_to_keep:
            modes_to_keep = [np.argmax(correlations)]
        
        if signal_idx is not None:
            pass
            # print(f"Signal {signal_idx}: Selected {len(modes_to_keep)} modes based on correlation: {modes_to_keep}")
        
        # Reconstruct signal
        if self.is_multivariate and len(imfs.shape) == 3:
            # Multivariate case
            reconstructed = np.zeros_like(imfs[0, :, :])
            for idx in modes_to_keep:
                reconstructed += imfs[idx, :, :]
        else:
            # Univariate case or incorrect imfs format
            reconstructed = np.zeros_like(imfs[0, :])
            for idx in modes_to_keep:
                reconstructed += imfs[idx, :]
        
        return reconstructed
    
    def _select_by_correlation_single(self, imfs, original_signal, signal_idx=None):
        """
        Select IMF components based on correlation for univariate signal
        
        Args:
            imfs: Decomposed IMF components
            original_signal: Original signal
            signal_idx: Signal index (for logging)
            
        Returns:
            Reconstructed signal
        """
        # Univariate case, calculate correlation of each mode with original signal
        correlations = np.zeros(self.K)
        for k in range(self.K):
            # Ensure length matches
            signal_len = min(len(original_signal), imfs.shape[1])
            signal_to_correlate = original_signal[:signal_len]
            imf_to_correlate = imfs[k, :signal_len]
            
            # Calculate correlation coefficient
            try:
                corr = np.corrcoef(signal_to_correlate, imf_to_correlate)[0, 1]
                if not np.isnan(corr):
                    correlations[k] = abs(corr)  # Use absolute value of correlation
            except:
                # If calculation fails, set correlation of this mode to 0
                if signal_idx is not None:
                    print(f"Warning: Signal {signal_idx}, Mode {k} correlation calculation failed")
                correlations[k] = 0
        
        # Select IMFs with correlation greater than threshold
        threshold = np.mean(correlations) * 0.8  # Use 80% of average correlation as threshold
        
        # Select modes with correlation higher than threshold
        modes_to_keep = []
        for k in range(self.K):
            if correlations[k] > threshold:
                modes_to_keep.append(k)
        
        # If no mode selected, select the one with highest correlation
        if not modes_to_keep:
            modes_to_keep = [np.argmax(correlations)]
        
        if signal_idx is not None:
            pass
            # print(f"Signal {signal_idx}: Selected {len(modes_to_keep)} modes based on correlation: {modes_to_keep}")
        
        # Reconstruct signal
        reconstructed = np.zeros_like(imfs[0, :])
        for idx in modes_to_keep:
            reconstructed += imfs[idx, :]
        
        return reconstructed
    
    def get_imfs(self, signal=None):
        """
        Get IMFs from last processing, or process new signal and return IMFs
        
        Args:
            signal: If provided, process this signal and return IMFs; otherwise return IMFs from last processing
            
        Returns:
            IMF components
        """
        if signal is not None:
            # Process new signal and return IMFs
            if self.is_multivariate:
                imfs, _, _ = mvvmd(signal, self.alpha, self.tau, self.K, self.DC, self.init, self.tol)
            else:
                if len(signal.shape) > 1:
                    signal = signal.flatten()  # Ensure it is a 1D signal
                imfs, _, _ = vmd(signal, self.alpha, self.tau, self.K, self.DC, self.init, self.tol)
            self.last_imfs = imfs
            return imfs
        elif self.last_imfs is not None:
            # Return IMFs from last processing
            return self.last_imfs
        else:
            raise ValueError("No IMF components available. Please call process method or provide signal first.") 