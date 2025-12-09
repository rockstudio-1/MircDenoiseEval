import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift


def vmd(signal, alpha, tau, K, DC, init, tol):
    """
    Variational Mode Decomposition
    
    Python implementation of the algorithm by Konstantin Dragomiretskiy and Dominique Zosso.
    
    Parameters
    ----------
    signal : ndarray
        The time domain signal (1D) to be decomposed
    alpha : float
        The balancing parameter of the data-fidelity constraint
    tau : float
        Time-step of the dual ascent (pick 0 for noise-slack)
    K : int
        The number of modes to be recovered
    DC : bool
        True if the first mode is put and kept at DC (0-freq)
    init : int
        0 = all omegas start at 0
        1 = all omegas start uniformly distributed
        2 = all omegas initialized randomly
    tol : float
        Tolerance of convergence criterion; typically around 1e-6
    
    Returns
    -------
    u : ndarray
        The collection of decomposed modes
    u_hat : ndarray
        Spectra of the modes
    omega : ndarray
        Estimated mode center-frequencies
    
    References
    ----------
    K. Dragomiretskiy, D. Zosso, Variational Mode Decomposition, IEEE Trans.
    on Signal Processing, vol. 62, no. 3, pp. 531-544, Feb. 2014.
    """
    # Period and sampling frequency of input signal
    save_T = len(signal)
    fs = 1 / save_T
    
    # Extend the signal by mirroring
    T = save_T
    f_mirror = np.zeros(2 * T)
    f_mirror[:T//2] = signal[T//2-1::-1]
    f_mirror[T//2:3*T//2] = signal
    f_mirror[3*T//2:2*T] = signal[T-1:T//2-1:-1]
    f = f_mirror
    
    # Time Domain 0 to T (of mirrored signal)
    T = len(f)
    t = np.arange(1, T + 1) / T
    
    # Spectral Domain discretization
    freqs = t - 0.5 - 1/T
    
    # Maximum number of iterations (if not converged yet, then it won't anyway)
    N = 500
    
    # For future generalizations: individual alpha for each mode
    Alpha = alpha * np.ones(K)
    
    # Construct and center f_hat
    f_hat = fftshift(fft(f))
    f_hat_plus = f_hat.copy()
    f_hat_plus[:T//2] = 0
    
    # Matrix keeping track of every iterant
    u_hat_plus = np.zeros((N, len(freqs), K), dtype=complex)
    
    # Initialization of omega_k
    omega_plus = np.zeros((N, K))
    if init == 1:
        for i in range(K):
            omega_plus[0, i] = (0.5 / K) * i
    elif init == 2:
        omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(K)))
    else:  # init == 0
        omega_plus[0, :] = 0
    
    # If DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0, 0] = 0
    
    # Start with empty dual variables
    lambda_hat = np.zeros((N, len(freqs)), dtype=complex)
    
    # Other inits
    uDiff = tol + np.finfo(float).eps  # Update step
    n = 0  # Loop counter
    sum_uk = 0  # Accumulator
    
    # ----------- Main loop for iterative updates
    while uDiff > tol and n < N - 1:  # Not converged and below iterations limit
        # Update first mode accumulator
        k = 0
        sum_uk = u_hat_plus[n, :, K-1] + sum_uk - u_hat_plus[n, :, 0]
        
        # Update spectrum of first mode through Wiener filter of residuals
        u_hat_plus[n+1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :]/2) / (1 + Alpha[k] * (freqs - omega_plus[n, k])**2)
        
        # Update first omega if not held at 0
        if not DC:
            omega_plus[n+1, k] = np.sum(freqs[T//2:T] * np.abs(u_hat_plus[n+1, T//2:T, k])**2) / np.sum(np.abs(u_hat_plus[n+1, T//2:T, k])**2)
        else:
            omega_plus[n+1, k] = 0
        
        # Update of any other mode
        for k in range(1, K):
            # Accumulator
            sum_uk = u_hat_plus[n+1, :, k-1] + sum_uk - u_hat_plus[n, :, k]
            
            # Mode spectrum
            u_hat_plus[n+1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :]/2) / (1 + Alpha[k] * (freqs - omega_plus[n, k])**2)
            
            # Center frequencies
            omega_plus[n+1, k] = np.sum(freqs[T//2:T] * np.abs(u_hat_plus[n+1, T//2:T, k])**2) / np.sum(np.abs(u_hat_plus[n+1, T//2:T, k])**2)
        
        # Dual ascent
        lambda_hat[n+1, :] = lambda_hat[n, :] + tau * (np.sum(u_hat_plus[n+1, :, :], axis=1) - f_hat_plus)
        
        # Loop counter
        n = n + 1
        
        # Converged yet?
        uDiff = np.finfo(float).eps
        for i in range(K):
            diff = u_hat_plus[n, :, i] - u_hat_plus[n-1, :, i]
            uDiff = uDiff + 1/T * np.sum(diff * np.conj(diff))
        uDiff = np.abs(uDiff)
    
    # ------ Postprocessing and cleanup
    
    # Discard empty space if converged early
    N = min(N, n+1)
    omega = omega_plus[:N, :]
    
    # Signal reconstruction
    u_hat = np.zeros((T, K), dtype=complex)
    for k in range(K):
        u_hat[T//2:T, k] = u_hat_plus[N-1, T//2:T, k]
    
    # Fix dimension error in lower half spectrum reconstruction
    for k in range(K):
        u_hat[1:T//2, k] = np.conj(u_hat_plus[N-1, T//2-1:0:-1, k])
    u_hat[0, :] = np.conj(u_hat[-1, :])
    
    u = np.zeros((K, len(t)))
    
    for k in range(K):
        u[k, :] = np.real(ifft(ifftshift(u_hat[:, k])))
    
    # Remove mirror part
    u = u[:, T//4:3*T//4]
    
    # Recompute spectrum
    u_hat = np.zeros((T//2, K), dtype=complex)
    for k in range(K):
        u_hat[:, k] = fftshift(fft(u[k, :]))
    
    return u, u_hat, omega 