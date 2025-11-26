"""
Variational Mode Decomposition (VMD)

Implementation of VMD algorithm using ADMM (Alternating Direction Method of Multipliers)
for decomposing a signal into K narrowband intrinsic mode functions (IMFs).

Reference:
Dragomiretskiy, K., & Zosso, D. (2014). "Variational mode decomposition."
IEEE Transactions on Signal Processing, 62(3), 531-544.
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq


def vmd(signal, alpha=2000, tau=0.0, K=3, DC=False, init=1, tol=1e-7, max_iter=500):
    """
    Variational Mode Decomposition

    Decomposes a signal into K narrowband modes using ADMM optimization in frequency domain.

    Parameters:
    -----------
    signal : array_like
        Input signal to decompose (real-valued time series)
    alpha : float
        Bandwidth penalty parameter (2000-5000 typical)
        Higher values enforce narrower spectral support for each mode
    tau : float
        Noise tolerance (Lagrangian update step size)
        0 for clean signals, small positive for noisy signals
    K : int
        Number of modes to extract (3-5 typical)
    DC : bool
        If True, first mode is forced to have zero mean (DC component)
    init : int
        Initialization method for center frequencies omega
        0 = all start at 0
        1 = uniformly distributed across spectrum (recommended)
        2 = random initialization
    tol : float
        Convergence tolerance for ADMM (1e-6 to 1e-7 typical)
    max_iter : int
        Maximum number of ADMM iterations

    Returns:
    --------
    u : ndarray, shape (K, len(signal))
        Decomposed modes (time domain)
    u_hat : ndarray, shape (K, len(signal))
        Decomposed modes (frequency domain)
    omega : ndarray, shape (K,)
        Center frequencies of each mode

    Algorithm:
    ----------
    VMD solves the variational problem:
        min Σ_k ||∂_t[(δ(t) + j/(πt)) * u_k(t)]e^(-jω_k t)||²
        subject to Σ_k u_k = f

    Using ADMM, this is converted to an augmented Lagrangian and solved by alternating:
    1. Update u_k in frequency domain (Wiener filtering)
    2. Update ω_k as center of gravity of |u_k|²
    3. Update dual variable λ (gradient ascent on constraint)
    """

    
    
    

    
    signal = np.squeeze(signal)
    if signal.ndim != 1:
        raise ValueError("Input signal must be 1D")

    T = len(signal)

    
    
    f_mirror = np.concatenate([signal[:T//2][::-1], signal, signal[T//2:][::-1]])
    T_extended = len(f_mirror)

    
    freqs = fftfreq(T_extended, d=1.0)

    
    
    omega_axis = 2 * np.pi * freqs[freqs >= 0]
    N = len(omega_axis)

    
    f_hat = fft(f_mirror)
    f_hat_plus = f_hat[:N].copy()  

    
    
    

    
    u_hat_plus = np.zeros((K, N), dtype=complex)

    
    omega_plus = np.zeros(K)

    if init == 0:
        
        omega_plus[:] = 0
    elif init == 1:
        
        omega_plus = np.linspace(0, omega_axis[-1], K + 2)[1:-1]
    elif init == 2:
        
        omega_plus = np.sort(np.random.rand(K) * omega_axis[-1])
    else:
        raise ValueError(f"Invalid init method: {init}. Use 0, 1, or 2")

    
    lambda_hat = np.zeros(N, dtype=complex)

    
    if tau == 0:
        tau = 1e-10

    
    
    

    uDiff = tol + 1  
    n_iter = 0

    
    u_hat_plus_prev = u_hat_plus.copy()

    while uDiff > tol and n_iter < max_iter:

        
        
        
        for k in range(K):
            
            sum_uk = np.sum(u_hat_plus, axis=0) - u_hat_plus[k, :]

            
            residual = f_hat_plus - sum_uk + lambda_hat / 2

            
            
            denominator = 1 + alpha * (omega_axis - omega_plus[k])**2

            
            u_hat_plus[k, :] = residual / denominator

            
            if DC and k == 0:
                u_hat_plus[k, 0] = 0

        
        
        
        for k in range(K):
            
            

            power_spectrum = np.abs(u_hat_plus[k, :])**2
            numerator = np.dot(omega_axis, power_spectrum)
            denominator = np.sum(power_spectrum)

            if denominator > 1e-10:
                omega_plus[k] = numerator / denominator
            else:
                
                pass

        
        
        
        
        lambda_hat += tau * (np.sum(u_hat_plus, axis=0) - f_hat_plus)

        
        
        
        
        uDiff = np.linalg.norm(u_hat_plus - u_hat_plus_prev, ord='fro')**2
        uDiff /= (np.linalg.norm(u_hat_plus_prev, ord='fro')**2 + 1e-10)

        u_hat_plus_prev = u_hat_plus.copy()
        n_iter += 1

    
    
    

    
    
    u_hat = np.zeros((K, T_extended), dtype=complex)

    for k in range(K):
        
        u_hat[k, :N] = u_hat_plus[k, :]

        
        
        u_hat[k, -N+1:] = np.conj(u_hat_plus[k, 1:N][::-1])

    
    u = np.zeros((K, T_extended))
    for k in range(K):
        u[k, :] = np.real(ifft(u_hat[k, :]))

    
    
    u = u[:, T//2:T//2+T]

    
    
    
    omega = omega_plus / (2 * np.pi)

    return u, u_hat, omega


def select_respiration_mode(modes, omega, fs, breathing_freq_min=0.1, breathing_freq_max=0.5):
    """
    Select the mode that best represents respiration signal.

    Selects the mode whose dominant spectral peak lies within the expected
    respiration frequency band.

    Parameters:
    -----------
    modes : ndarray, shape (K, N)
        VMD decomposed modes (time domain)
    omega : ndarray, shape (K,)
        Center frequencies of each mode (normalized: cycles per sample)
    fs : float
        Sampling rate (Hz)
    breathing_freq_min : float
        Minimum breathing frequency (Hz)
    breathing_freq_max : float
        Maximum breathing frequency (Hz)

    Returns:
    --------
    respiration_mode : ndarray
        The mode selected as respiration signal
    mode_index : int
        Index of selected mode
    mode_info : dict
        Information about all modes for comparison
    """

    K = modes.shape[0]

    
    omega_hz = omega * fs

    
    candidates = []
    mode_info = []

    for k in range(K):
        
        N = len(modes[k])
        mode_fft = fft(modes[k])
        freqs = fftfreq(N, d=1.0/fs)

        
        positive_idx = freqs > 0
        freqs_pos = freqs[positive_idx]
        power = np.abs(mode_fft[positive_idx])**2

        
        peak_idx = np.argmax(power)
        peak_freq = freqs_pos[peak_idx]
        peak_power = power[peak_idx]

        
        total_power = np.sum(power)

        
        breathing_idx = (freqs_pos >= breathing_freq_min) & (freqs_pos <= breathing_freq_max)
        breathing_power = np.sum(power[breathing_idx])

        info = {
            'mode_index': k,
            'center_freq': omega_hz[k],
            'peak_freq': peak_freq,
            'peak_power': peak_power,
            'total_power': total_power,
            'breathing_power': breathing_power,
            'breathing_power_ratio': breathing_power / (total_power + 1e-10)
        }
        mode_info.append(info)

        
        if breathing_freq_min <= omega_hz[k] <= breathing_freq_max:
            candidates.append((k, breathing_power))
        elif breathing_freq_min <= peak_freq <= breathing_freq_max:
            
            candidates.append((k, breathing_power))

    
    if len(candidates) > 0:
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        mode_index = candidates[0][0]
    else:
        
        
        powers = [info['total_power'] for info in mode_info]
        mode_index = np.argmax(powers)

    respiration_mode = modes[mode_index]

    return respiration_mode, mode_index, mode_info
