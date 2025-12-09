import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.stats import pearsonr
import random
from tqdm import tqdm

# Import wavelet denoising class
from wt import WaveletDenoise

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval, return mean and margin of error"""
    n = len(data)
    if n < 2:
        return np.mean(data), 0  # Sample too small to calculate confidence interval
    
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error
    
    # Use t-distribution to calculate confidence interval
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_value * std_err
    
    return mean, margin_error

def calculate_snr(clean_signal, noisy_signal):
    """
    Calculate Signal-to-Noise Ratio (SNR)
    
    Args:
        clean_signal: Clean signal
        noisy_signal: Noisy signal
        
    Returns:
        SNR value (dB)
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
    Calculate Mean Squared Error (MSE)
    
    Args:
        clean_signal: Clean signal
        processed_signal: Processed signal
        
    Returns:
        MSE value
    """
    # Ensure signal lengths are consistent
    min_len = min(len(clean_signal), len(processed_signal))
    clean_signal = clean_signal[:min_len]
    processed_signal = processed_signal[:min_len]
    
    mse = np.mean((clean_signal - processed_signal) ** 2)
    return mse

def calculate_correlation(clean_signal, processed_signal):
    """
    Calculate Correlation Coefficient
    
    Args:
        clean_signal: Clean signal
        processed_signal: Processed signal
        
    Returns:
        Correlation coefficient value
    """
    # Ensure signal lengths are consistent
    min_len = min(len(clean_signal), len(processed_signal))
    clean_signal = clean_signal[:min_len]
    processed_signal = processed_signal[:min_len]
    
    correlation, _ = pearsonr(clean_signal, processed_signal)
    return correlation

def calculate_spectral_distortion(clean_signal, processed_signal, fs=100):
    """
    Calculate Spectral Distortion
    
    Args:
        clean_signal: Clean signal
        processed_signal: Processed signal
        fs: Sampling frequency
        
    Returns:
        Spectral distortion value
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
    
    # Spectral distortion calculation
    sd = np.mean((10 * np.log10(Pxx_processed / Pxx_clean) )** 2)
    return np.sqrt(sd)

def calculate_esn(clean_signal, processed_signal):
    """
    Calculate Energy Signal-to-Noise Ratio (ESN) Percentage
    Percentage of energy of the denoised signal to the original signal energy
    
    Args:
        clean_signal: Clean signal
        processed_signal: Processed signal
        
    Returns:
        ESN value (%)
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
    Plot signal comparison
    
    Args:
        clean_signal: Clean signal
        noisy_signal: Noisy signal
        denoised_signal: Denoised signal
        title: Plot title
        save_path: Save path
        snr_before: SNR before denoising
        snr_after: SNR after denoising
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
    plt.title(f'Denoised Signal (Wavelet) {snr_text}')
    plt.xlabel('Sample Points')
    plt.ylabel('Amplitude')
    
    # If SNR values provided, add improvement info to title
    if snr_before is not None and snr_after is not None:
        snr_improvement = snr_after - snr_before
        main_title = f"{title}\nSNR Improvement: {snr_improvement:.2f} dB"
    else:
        main_title = title
        
    plt.suptitle(main_title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_denoised_signals_csv(denoised_results, output_dir="results"):
    """
    Quickly save denoised signal data to CSV file
    
    Args:
        denoised_results: List containing (denoised_signal, idx, snr_improvement) tuples
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not denoised_results:
        print("No denoised results to save")
        return
    
    # Pre-allocate numpy array for performance
    num_signals = len(denoised_results)
    signal_length = len(denoised_results[0][0])  # Assume all signals have same length
    
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
    
    # Insert index column at the beginning (corresponding to first column)
    denoised_df.insert(0, '', indices)  # First column header is blank
    
    # Save denoised signal data
    denoised_df.to_csv(os.path.join(output_dir, "denoised_signals.csv"), index=False)
    
    print(f"Wavelet denoised signal data saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Wavelet Denoising Algorithm Performance")
    parser.add_argument("--noisy_file", type=str, default=None, help="Path to noisy signal CSV file")
    parser.add_argument("--clean_file", type=str, default=None, help="Path to clean signal CSV file")
    parser.add_argument("--output_dir", type=str, default="WT/wavelet_results", help="Output directory")
    parser.add_argument("--plot_samples", type=int, default=0, help="Number of comparison plots to generate")
    parser.add_argument("--wavelet", type=str, default="sym8", help="Wavelet type")
    parser.add_argument("--level", type=int, default=8, help="Decomposition level")
    parser.add_argument("--threshold_type", type=int, default=2, help="Threshold type: 1=Hard, 2=Soft, 3=Semi-soft")
    parser.add_argument("--specific_indices", type=str, default=None, help="Specific signal indices to process, comma separated, e.g., '0,1,5,10'")
    parser.add_argument("--save_csv", action="store_true", help="Whether to save signal data to CSV file")
    args = parser.parse_args()
    
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
    
    # Create wavelet denoiser
    denoiser = WaveletDenoise(wavelet_name=args.wavelet, 
                             level=args.level, 
                             threshold_type=args.threshold_type)
    
    # Initialize results list
    results = []
    denoised_results = []  # For saving denoised signal data
    
    # Determine signal indices to process
    num_signals = min(len(noisy_signals), len(clean_signals))
    
    if args.specific_indices:
        # Parse user specified indices
        try:
            specific_indices = [int(idx.strip()) for idx in args.specific_indices.split(',')]
            # Filter out indices out of range
            specific_indices = [idx for idx in specific_indices if 0 <= idx < num_signals]
            if not specific_indices:
                print("Warning: All specified indices are out of valid range, processing all signals")
                indices_to_process = list(range(num_signals))
            else:
                indices_to_process = specific_indices
                print(f"Will only process {len(indices_to_process)} specified signals: {indices_to_process}")
        except ValueError:
            print("Warning: Invalid index format, processing all signals")
            indices_to_process = list(range(num_signals))
    else:
        # Process all signals
        indices_to_process = list(range(num_signals))
    
    # Select signal indices for plotting
    if args.specific_indices:
        # If specific indices specified, generate plots for all of them
        plot_indices = indices_to_process[:min(len(indices_to_process), args.plot_samples)]
    else:
        # Otherwise random selection
        plot_indices = random.sample(indices_to_process, min(args.plot_samples, len(indices_to_process)))
    
    # Process selected signals
    print(f"Starting processing of {len(indices_to_process)} signals...")
    for idx in tqdm(indices_to_process):
        noisy_signal = noisy_signals[idx]
        clean_signal = clean_signals[idx]
        
        # Apply wavelet denoising
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
        
        # If need to save CSV, collect denoised signal data
        if not args.save_csv:
            denoised_results.append((denoised_signal, idx, snr_improvement))
        
        # If need to plot comparison for this signal
        if idx in plot_indices:
            plot_path = os.path.join(args.output_dir, f"comparison_signal_{idx}.png")
            plot_signals(clean_signal, noisy_signal, denoised_signal, 
                         f"Signal {idx} Comparison", plot_path,
                         snr_before=snr_before, 
                         snr_after=snr_after)
    
    # Save all performance metrics results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, "evaluation_results.csv"), index=False)
    
    # Optional: Save denoised signal data to CSV
    if not args.save_csv:
        print("Saving wavelet denoised signal data to CSV file...")
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