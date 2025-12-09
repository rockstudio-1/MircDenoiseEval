import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.stats import pearsonr
import random
from tqdm import tqdm

# Import VMD denoise class
from vmd_processing import VMDDenoise

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval, return mean and margin of error."""
    n = len(data)
    if n < 2:
        return np.mean(data), 0  # Too few samples to calculate confidence interval
    
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error
    
    # Calculate confidence interval using t-distribution
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_value * std_err
    
    return mean, margin_error

def calculate_snr(clean_signal, noisy_signal):
    """
    Calculate Signal-to-Noise Ratio (SNR).
    
    Args:
        clean_signal: Clean signal.
        noisy_signal: Noisy signal.
        
    Returns:
        SNR value (dB).
    """
    # Ensure signal lengths are consistent
    min_len = min(len(clean_signal), len(noisy_signal))
    clean_signal = clean_signal[:min_len]
    noisy_signal = noisy_signal[:min_len]
    
    noise = noisy_signal - clean_signal
    signal_power = np.sum(clean_signal ** 2)
    noise_power = np.sum(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_mse(clean_signal, processed_signal):
    """
    Calculate Mean Squared Error (MSE).
    
    Args:
        clean_signal: Clean signal.
        processed_signal: Processed signal.
        
    Returns:
        MSE value.
    """
    # Ensure signal lengths are consistent
    min_len = min(len(clean_signal), len(processed_signal))
    clean_signal = clean_signal[:min_len]
    processed_signal = processed_signal[:min_len]
    
    mse = np.mean((clean_signal - processed_signal) ** 2)
    return mse

def calculate_correlation(clean_signal, processed_signal):
    """
    Calculate Correlation Coefficient.
    
    Args:
        clean_signal: Clean signal.
        processed_signal: Processed signal.
        
    Returns:
        Correlation coefficient value.
    """
    # Ensure signal lengths are consistent
    min_len = min(len(clean_signal), len(processed_signal))
    clean_signal = clean_signal[:min_len]
    processed_signal = processed_signal[:min_len]
    
    correlation, _ = pearsonr(clean_signal, processed_signal)
    return correlation

def calculate_spectral_distortion(clean_signal, processed_signal, fs=100):
    """
    Calculate Spectral Distortion.
    
    Args:
        clean_signal: Clean signal.
        processed_signal: Processed signal.
        fs: Sampling frequency.
        
    Returns:
        Spectral distortion value.
    """
    # Ensure signal lengths are consistent
    min_len = min(len(clean_signal), len(processed_signal))
    clean_signal = clean_signal[:min_len]
    processed_signal = processed_signal[:min_len]
    
    # Calculate Power Spectral Density
    f, Pxx_clean = signal.welch(clean_signal, fs=fs, nperseg=min(256, min_len))
    f, Pxx_processed = signal.welch(processed_signal, fs=fs, nperseg=min(256, min_len))
    
    # Prevent division by zero error
    Pxx_clean = np.maximum(Pxx_clean, 1e-10)
    Pxx_processed = np.maximum(Pxx_processed, 1e-10)
    
    # Calculate Spectral Distortion
    sd = np.mean((10 * np.log10(Pxx_processed / Pxx_clean) )** 2)
    return np.sqrt(sd)

def calculate_esn(clean_signal, processed_signal):
    """
    Calculate Energy Signal-to-Noise Ratio (ESN) Percentage.
    Percentage of energy of the denoised signal to the energy of the original signal.
    
    Args:
        clean_signal: Clean signal.
        processed_signal: Processed signal.
        
    Returns:
        ESN value (%).
    """
    # Ensure signal lengths are consistent
    min_len = min(len(clean_signal), len(processed_signal))
    clean_signal = clean_signal[:min_len]
    processed_signal = processed_signal[:min_len]
    
    # Calculate original signal energy
    energy_clean = np.sum(clean_signal ** 2)
    
    # Calculate processed signal energy
    energy_processed = np.sum(processed_signal ** 2)
    
    # Energy percentage
    if energy_clean == 0:
        return 0
    
    esn = (energy_processed / energy_clean) * 100
    return esn

def plot_signals(clean_signal, noisy_signal, denoised_signal, title, save_path, snr_before=None, snr_after=None):
    """
    Plot signal comparison.
    
    Args:
        clean_signal: Clean signal.
        noisy_signal: Noisy signal.
        denoised_signal: Denoised signal.
        title: Plot title.
        save_path: Path to save the plot.
        snr_before: SNR before denoising.
        snr_after: SNR after denoising.
    """
    # Ensure signal lengths are consistent
    min_len = min(len(clean_signal), len(noisy_signal), len(denoised_signal))
    clean_signal = clean_signal[:min_len]
    noisy_signal = noisy_signal[:min_len]
    denoised_signal = denoised_signal[:min_len]
    
    # Create time axis
    t = np.arange(min_len)
    
    plt.figure(figsize=(12, 9))
    
    # Plot clean signal
    plt.subplot(3, 1, 1)
    plt.plot(t, clean_signal)
    plt.title('Clean Signal')
    plt.xlabel('Sample Points')
    plt.ylabel('Amplitude')
    
    # Plot noisy signal
    plt.subplot(3, 1, 2)
    plt.plot(t, noisy_signal)
    snr_text = f'SNR: {snr_before:.2f} dB' if snr_before is not None else ''
    plt.title(f'Noisy Signal {snr_text}')
    plt.xlabel('Sample Points')
    plt.ylabel('Amplitude')
    
    # Plot denoised signal
    plt.subplot(3, 1, 3)
    plt.plot(t, denoised_signal)
    snr_text = f'SNR: {snr_after:.2f} dB' if snr_after is not None else ''
    plt.title(f'Denoised Signal (VMD) {snr_text}')
    plt.xlabel('Sample Points')
    plt.ylabel('Amplitude')
    
    # Add SNR improvement info to the title if SNR values are provided
    if snr_before is not None and snr_after is not None:
        snr_improvement = snr_after - snr_before
        main_title = f"{title}\nSNR Improvement: {snr_improvement:.2f} dB"
    else:
        main_title = title
        
    plt.suptitle(main_title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_modes(signal, modes, title, save_path, clean_signal=None):
    """
    Plot VMD modes.
    
    Args:
        signal: Original signal (noisy).
        modes: VMD decomposed modes.
        title: Plot title.
        save_path: Path to save the plot.
        clean_signal: Clean signal (if provided, noise will be calculated and plotted).
    """
    n_modes = len(modes)
    
    # If clean signal is provided, add a subplot to display noise
    extra_plots = 1
    if clean_signal is not None:
        extra_plots = 2
    
    plt.figure(figsize=(12, (n_modes + extra_plots) * 1.5 + 2))
    
    # Ensure signal lengths are consistent
    if clean_signal is not None:
        min_len = min(len(signal), len(clean_signal))
        signal_plot = signal[:min_len]
        clean_signal_plot = clean_signal[:min_len]
        # Calculate noise
        noise = signal_plot - clean_signal_plot
    else:
        signal_plot = signal
    
    # Plot original signal
    plt.subplot(n_modes + extra_plots, 1, 1)
    plt.plot(signal_plot)
    plt.title('Original Signal (Noisy)')
    plt.xlabel('Sample Points')
    plt.ylabel('Amplitude')
    
    # If clean signal is provided, plot noise
    if clean_signal is not None:
        plt.subplot(n_modes + extra_plots, 1, 2)
        plt.plot(noise, 'r')
        plt.title('Noise Component')
        plt.xlabel('Sample Points')
        plt.ylabel('Amplitude')
    
    # Plot each mode
    for i, mode in enumerate(modes):
        plt.subplot(n_modes + extra_plots, 1, i + extra_plots + 1)
        # Ensure mode length does not exceed signal length
        mode_plot = mode[:len(signal_plot)] if len(mode) > len(signal_plot) else mode
        plt.plot(mode_plot)
        plt.title(f'Mode {i+1}')
        plt.xlabel('Sample Points')
        plt.ylabel('Amplitude')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_denoised_signals_csv(denoised_results, output_dir="results"):
    """
    Quickly save denoised signal data to a CSV file.
    
    Args:
        denoised_results: List containing tuples of (denoised_signal, idx, snr_improvement).
        output_dir: Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not denoised_results:
        print("No denoised results to save")
        return
    
    # Pre-allocate numpy array to improve performance
    num_signals = len(denoised_results)
    signal_length = len(denoised_results[0][0])  # Assume all signals have the same length
    
    # Create array to store denoised signal data
    denoised_signals = np.zeros((num_signals, signal_length), dtype=np.float32)
    indices = np.zeros(num_signals, dtype=int)
    snr_improvements = np.zeros(num_signals, dtype=np.float32)
    
    # Batch fill data
    for i, (denoised_signal, idx, snr_improvement) in enumerate(denoised_results):
        denoised_signals[i] = denoised_signal
        indices[i] = idx
        snr_improvements[i] = snr_improvement
    
    # Create column names, following CSV format: first column blank, then 0,1,2...4000,Ptime,SNR_Improvement
    signal_columns = [str(i) for i in range(signal_length)]
    
    # Create DataFrame, following CSV format
    denoised_df = pd.DataFrame(denoised_signals, columns=signal_columns)
    
    # Add Ptime column (set to -1)
    denoised_df['Ptime'] = -1
    
    # Add SNR_Improvement column
    denoised_df['SNR_Improvement'] = snr_improvements
    
    # Insert index column at the beginning (corresponding to the first column)
    denoised_df.insert(0, '', indices)  # First column header is blank
    
    # Save denoised signal data
    denoised_df.to_csv(os.path.join(output_dir, "denoised_signals.csv"), index=False)
    
    print(f"VMD denoised signal data saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate VMD denoising algorithm performance")
    parser.add_argument("--noisy_file", type=str, default=None, help="Noisy signal CSV file path")
    parser.add_argument("--clean_file", type=str, default=None, help="Clean signal CSV file path")
    parser.add_argument("--output_dir", type=str, default="VMD/vmd_results", help="Output directory")
    
    # VMD parameters
    parser.add_argument("--alpha", type=float, default=2000, help="Bandwidth constraint parameter")
    parser.add_argument("--tau", type=float, default=0, help="Dual ascent step size")
    parser.add_argument("--K", type=int, default=5, help="Number of modes to extract")
    parser.add_argument("--DC", action="store_true", help="Whether to fix the first mode at DC")
    parser.add_argument("--init", type=int, default=1, choices=[0, 1, 2], help="Center frequency initialization method")
    parser.add_argument("--tol", type=float, default=1e-7, help="Convergence tolerance")
    parser.add_argument("--noise_modes", default='auto', help="Number of noise modes to remove, 'auto' for automatic determination")
    parser.add_argument("--mode_selection", type=str, default='correlation', choices=['energy', 'correlation'], 
                        help="Mode selection method: energy (energy-based) or correlation (correlation-based)")
    
    # Visualization parameters
    parser.add_argument("--plot_samples", type=int, default=0, help="Number of samples to plot comparison")
    parser.add_argument("--plot_modes", action="store_true", help="Whether to plot VMD modes")
    parser.add_argument("--plot_mode_samples", type=int, default=0, help="Number of samples to plot VMD modes")
    
    # Data selection parameters
    parser.add_argument("--num_signals", type=int, default=256, help="Number of signals to process, default processes all if None")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducible random selection")
    parser.add_argument("--specific_indices", type=str, default=None, help="Specific signal indices to process, comma separated, e.g., '0,1,5,10'")
    
    # CSV save parameters
    parser.add_argument("--save_csv", action="store_true", help="Whether to save signal data to CSV file")
    
    args = parser.parse_args()
    # Set random seed
    random.seed(args.random_seed)
    
    # Convert noise_modes parameter
    if args.noise_modes != 'auto':
        args.noise_modes = int(args.noise_modes)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read signal data
    print("Reading data...")
    noisy_df = pd.read_csv(args.noisy_file)
    clean_df = pd.read_csv(args.clean_file)
    
    # Extract signal data (skip first column)
    noisy_signals = noisy_df.iloc[:, 1:].values if noisy_df.shape[1] > 1 else noisy_df.values
    clean_signals = clean_df.iloc[:, 1:].values if clean_df.shape[1] > 1 else clean_df.values
    
    # Ensure data dimensions are correct
    if len(noisy_signals.shape) == 1:
        noisy_signals = noisy_signals.reshape(1, -1)
    if len(clean_signals.shape) == 1:
        clean_signals = clean_signals.reshape(1, -1)
    
    # Create VMD denoiser
    print(f"Using VMD method for denoising, parameters: alpha={args.alpha}, K={args.K}, noise_modes={args.noise_modes}, mode_selection={args.mode_selection}")
    
    denoiser = VMDDenoise(
        alpha=args.alpha,
        tau=args.tau,
        K=args.K,
        DC=args.DC,
        init=args.init,
        tol=args.tol,
        noise_modes=args.noise_modes,
        mode_selection=args.mode_selection
    )
    
    # Determine signal indices to process
    available_signals = min(len(noisy_signals), len(clean_signals))
    
    if args.specific_indices:
        # Parse user-specified indices
        try:
            specific_indices = [int(idx.strip()) for idx in args.specific_indices.split(',')]
            # Filter out indices that are out of range
            specific_indices = [idx for idx in specific_indices if 0 <= idx < available_signals]
            if not specific_indices:
                print("Warning: Specified indices are out of valid range, random signals will be selected")
                if args.num_signals is not None and args.num_signals > 0:
                    num_to_process = min(available_signals, args.num_signals)
                else:
                    num_to_process = available_signals
                signal_indices = sorted(random.sample(range(available_signals), num_to_process))
                print(f"Randomly selecting {num_to_process} signals for processing (Random seed: {args.random_seed})")
            else:
                signal_indices = specific_indices
                print(f"Will only process {len(signal_indices)} specified signals: {signal_indices}")
        except ValueError:
            print("Warning: Index format error, random signals will be selected")
            if args.num_signals is not None and args.num_signals > 0:
                num_to_process = min(available_signals, args.num_signals)
            else:
                num_to_process = available_signals
            signal_indices = sorted(random.sample(range(available_signals), num_to_process))
            print(f"Randomly selecting {num_to_process} signals for processing (Random seed: {args.random_seed})")
    else:
        # If number of signals to process is specified, limit the number
        if args.num_signals is not None and args.num_signals > 0:
            num_to_process = min(available_signals, args.num_signals)
        else:
            num_to_process = available_signals
        
        # Determine signal indices to process
        if num_to_process < available_signals:
            # Randomly select specified number of signals
            signal_indices = sorted(random.sample(range(available_signals), num_to_process))
            print(f"Randomly selecting {num_to_process} signals for processing (Random seed: {args.random_seed})")
        else:
            # Process all signals
            signal_indices = list(range(num_to_process))
            print(f"Processing all {num_to_process} signals")
    
    # Select signal indices for plotting
    if args.specific_indices and specific_indices:
        # If specific indices are specified, generate charts for these indices (but not exceeding plot_samples)
        plot_indices = signal_indices[:min(len(signal_indices), args.plot_samples)]
    else:
        # Otherwise randomly select
        plot_indices = random.sample(signal_indices, min(args.plot_samples, len(signal_indices)))
    
    # If plotting mode charts is required, select some samples
    if args.plot_modes:
        if args.specific_indices and specific_indices:
            # If specific indices are specified, generate mode charts for these indices (but not exceeding plot_mode_samples)
            mode_plot_indices = signal_indices[:min(len(signal_indices), args.plot_mode_samples)]
        else:
            # Otherwise randomly select
            mode_plot_indices = random.sample(signal_indices, min(args.plot_mode_samples, len(signal_indices)))
        # Create mode plot directory
        mode_dir = os.path.join(args.output_dir, "mode_plots")
        os.makedirs(mode_dir, exist_ok=True)
    
    # Initialize results list
    results = []
    denoised_results = []  # To save denoised signal data
    
    # Process selected signals
    print(f"Starting to process {len(signal_indices)} signals...")
    for i, idx in enumerate(tqdm(signal_indices)):
        noisy_signal = noisy_signals[idx]
        clean_signal = clean_signals[idx]
        
        # Apply VMD denoising
        denoised_signal = denoiser.process(noisy_signal, idx)
        
        # Calculate performance metrics
        # 1. SNR before denoising
        snr_before = calculate_snr(clean_signal, noisy_signal)
        
        # 2. SNR after denoising
        snr_after = calculate_snr(clean_signal, denoised_signal)
        
        # 3. SNR improvement
        snr_improvement = snr_after - snr_before
        
        # 4. MSE
        mse_before = calculate_mse(clean_signal, noisy_signal)
        mse_after = calculate_mse(clean_signal, denoised_signal)
        mse_improvement = mse_before - mse_after
        
        # 5. Correlation coefficient
        correlation = calculate_correlation(clean_signal, denoised_signal)
        
        # 6. Spectral distortion
        spectral_distortion = calculate_spectral_distortion(clean_signal, denoised_signal)
        
        # 7. Energy percentage ESN
        esn = calculate_esn(clean_signal, denoised_signal)
        
        # Save results
        results.append({
            "Signal Index": idx,
            "SNR Before (dB)": snr_before,
            "SNR After (dB)": snr_after,
            "SNR Improvement (dB)": snr_improvement,
            "MSE Before": mse_before,
            "MSE After": mse_after,
            "MSE Improvement": mse_improvement,
            "Correlation Coefficient": correlation,
            "Spectral Distortion": spectral_distortion,
            "Energy Percentage ESN (%)": esn
        })
        
        # If saving CSV is required, collect denoised signal data
        if not args.save_csv:
            denoised_results.append((denoised_signal, idx, snr_improvement))
        
        # If plotting comparison for this signal is required
        if idx in plot_indices:
            plot_path = os.path.join(args.output_dir, f"comparison_signal_{idx}.png")
            plot_signals(clean_signal, noisy_signal, denoised_signal, 
                       f"Signal {idx} Comparison (VMD)", 
                       plot_path, 
                       snr_before=snr_before, 
                       snr_after=snr_after)
            
        # If plotting modes is required
        if args.plot_modes and idx in mode_plot_indices:
            # Get modes and plot
            try:
                modes = denoiser.get_imfs(noisy_signal)
                mode_plot_path = os.path.join(mode_dir, f"mode_signal_{idx}.png")
                plot_modes(noisy_signal, modes, f"Signal {idx} VMD Modes", 
                          mode_plot_path, clean_signal=clean_signal)
            except Exception as e:
                print(f"Warning: Failed to plot modes for signal {idx}: {e}")
    
    # Save all performance metrics results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, "evaluation_results.csv"), index=False)
    
    # Optional: Save denoised signal data to CSV
    if not args.save_csv:
        print("Saving VMD denoised signal data to CSV file...")
        save_denoised_signals_csv(denoised_results, output_dir=args.output_dir)
    
    # Calculate average performance metrics
    avg_results = {
        "Average SNR Improvement (dB)": results_df["SNR Improvement (dB)"].mean(),
        "Average MSE Improvement": results_df["MSE Improvement"].mean(),
        "Average Correlation Coefficient": results_df["Correlation Coefficient"].mean(),
        "Average Spectral Distortion": results_df["Spectral Distortion"].mean(),
        "Average Energy Percentage ESN (%)": results_df["Energy Percentage ESN (%)"].mean()
    }
    
    # Save average performance metrics
    with open(os.path.join(args.output_dir, "average_results.txt"), "w") as f:
        f.write(f"VMD Parameters: alpha={args.alpha}, K={args.K}, mode_selection={args.mode_selection}\n")
        f.write(f"Noise Modes: {args.noise_modes}\n\n")
        
        for metric, value in avg_results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    # Calculate confidence interval
    snr_values = [result["SNR Improvement (dB)"] for result in results]
    mse_values = [result["MSE Improvement"] for result in results]
    corr_values = [result["Correlation Coefficient"] for result in results]
    spec_values = [result["Spectral Distortion"] for result in results]
    esn_values = [result["Energy Percentage ESN (%)"] for result in results]
    
    # Calculate mean and margin of error for 95% confidence interval
    snr_mean, snr_error = calculate_confidence_interval(snr_values)
    mse_mean, mse_error = calculate_confidence_interval(mse_values)
    corr_mean, corr_error = calculate_confidence_interval(corr_values)
    spec_mean, spec_error = calculate_confidence_interval(spec_values)
    esn_mean, esn_error = calculate_confidence_interval(esn_values)
    
    print(f"Evaluation completed. Results saved to {args.output_dir} directory")
    print("Average Performance Metrics (95% Confidence Interval):")
    print(f"Average SNR Improvement (dB): {snr_mean:.4f} ± {snr_error:.4f}")
    print(f"Average MSE Improvement: {mse_mean:.4f} ± {mse_error:.4f}")
    print(f"Average Correlation Coefficient: {corr_mean:.4f} ± {corr_error:.4f}")
    print(f"Average Spectral Distortion: {spec_mean:.4f} ± {spec_error:.4f}")
    print(f"Average Energy Percentage ESN (%): {esn_mean:.4f} ± {esn_error:.4f}")

if __name__ == "__main__":
    main()