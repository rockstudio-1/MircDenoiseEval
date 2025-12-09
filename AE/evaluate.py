import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy import signal, stats
from scipy.stats import pearsonr
from tqdm import tqdm

from model import ConvAutoencoder, normalize_signal

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval, returning mean and margin of error."""
    n = len(data)
    if n < 2:
        return np.mean(data), 0  # Too few samples to calculate confidence interval
    
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error
    
    # Calculate confidence interval using t-distribution
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_value * std_err
    
    return mean, margin_error

def load_model(model_path, device='cuda'):
    """Load trained model."""
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_signal_pair(clean_file, mixed_file, indices=None, num_samples=None):
    """Load clean and mixed signal pairs."""
    # Read CSV files
    clean_df = pd.read_csv(clean_file)
    mixed_df = pd.read_csv(mixed_file)
    
    # Randomly select samples if indices are not specified but num_samples is
    if indices is None and num_samples is not None:
        total_samples = len(clean_df)
        indices = np.random.choice(total_samples, num_samples, replace=False)
    # Use all samples if neither is specified
    elif indices is None:
        indices = range(len(clean_df))
    
    signal_pairs = []
    
    for idx in indices:
        # Read clean and mixed signals (extract data from columns 1 to 4001)
        clean_signal = clean_df.iloc[idx, 1:4001].values
        mixed_signal = mixed_df.iloc[idx, 1:4001].values
        
        signal_pairs.append((clean_signal, mixed_signal, idx))
    
    return signal_pairs

def denoise_signals(model, signal_pairs, device='cuda', batch_size=None):
    """Denoise signals using the model."""
    results = []
    
    if batch_size is None or batch_size <= 1:
        # Process signals one by one
        for clean, noisy, idx in tqdm(signal_pairs, desc="Processing signals"):
            # Convert to Tensor and add batch dimension
            noisy_tensor = torch.FloatTensor(noisy).unsqueeze(0).to(device)
            
            # Denoise using model
            with torch.no_grad():
                denoised_tensor = model(noisy_tensor)
            
            # Convert back to numpy array
            denoised = denoised_tensor.cpu().squeeze(0).numpy()
            
            # Save results
            results.append((clean, noisy, denoised, idx))
    else:
        # Batch process signals
        for i in tqdm(range(0, len(signal_pairs), batch_size), desc="Batch processing"):
            batch_pairs = signal_pairs[i:i+batch_size]
            batch_noisy = [pair[1] for pair in batch_pairs]
            batch_clean = [pair[0] for pair in batch_pairs]
            batch_indices = [pair[2] for pair in batch_pairs]
            
            # Convert to Tensor
            # Convert list to single array first using numpy.array(), then to tensor for performance
            batch_noisy_array = np.array(batch_noisy)
            batch_noisy_tensor = torch.FloatTensor(batch_noisy_array).to(device)
            
            # Denoise using model
            with torch.no_grad():
                batch_denoised_tensor = model(batch_noisy_tensor)
            
            # Convert back to numpy array
            batch_denoised = batch_denoised_tensor.cpu().numpy()
            
            # Save results
            for j in range(len(batch_pairs)):
                results.append((batch_clean[j], batch_noisy[j], batch_denoised[j], batch_indices[j]))
    
    return results

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
        Spectral Distortion value.
    """
    # Ensure signal lengths are consistent
    min_len = min(len(clean_signal), len(processed_signal))
    clean_signal = clean_signal[:min_len]
    processed_signal = processed_signal[:min_len]
    
    # Calculate Power Spectral Density
    f, Pxx_clean = signal.welch(clean_signal, fs=fs, nperseg=min(256, min_len))
    f, Pxx_processed = signal.welch(processed_signal, fs=fs, nperseg=min(256, min_len))
    
    # Prevent division by zero errors
    Pxx_clean = np.maximum(Pxx_clean, 1e-10)
    Pxx_processed = np.maximum(Pxx_processed, 1e-10)
    
    # Calculate Spectral Distortion
    sd = np.mean((10 * np.log10(Pxx_processed / Pxx_clean) )** 2)
    return np.sqrt(sd)

def calculate_esn(clean_signal, processed_signal):
    """
    Calculate Energy Signal-to-Noise Ratio (ESN) Percentage.
    Percentage of energy of the denoised signal relative to the original signal.
    
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

def calculate_metrics(results):
    """
    Calculate all evaluation metrics.
    
    Args:
        results: List of tuples (clean, noisy, denoised, idx).
        
    Returns:
        List containing all metrics.
    """
    metrics_list = []
    
    for clean, noisy, denoised, idx in results:
        # Calculate SNR of noisy signal
        input_snr = calculate_snr(clean, noisy)
        
        # Calculate SNR of denoised signal
        output_snr = calculate_snr(clean, denoised)
        
        # Calculate SNR improvement
        snr_improvement = output_snr - input_snr
        
        # Calculate MSE
        input_mse = calculate_mse(clean, noisy)
        output_mse = calculate_mse(clean, denoised)
        mse_improvement = input_mse - output_mse
        
        # Calculate correlation coefficient
        correlation = calculate_correlation(clean, denoised)
        
        # Calculate spectral distortion
        spectral_distortion = calculate_spectral_distortion(clean, denoised)
        
        # Calculate energy percentage
        esn = calculate_esn(clean, denoised)
        
        metrics = {
            'index': idx,
            'input_snr': input_snr,
            'output_snr': output_snr,
            'snr_improvement': snr_improvement,
            'input_mse': input_mse,
            'output_mse': output_mse,
            'mse_improvement': mse_improvement,
            'correlation': correlation,
            'spectral_distortion': spectral_distortion,
            'energy_percentage': esn
        }
        
        metrics_list.append(metrics)
    
    return metrics_list

def plot_results(results, output_dir="results", max_plots=5):
    """
    Plot denoising results.
    
    Args:
        results: List of tuples (clean, noisy, denoised, idx).
        output_dir: Output directory.
        max_plots: Maximum number of signals to plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit number of plots
    plot_count = min(len(results), max_plots)
    
    for i in range(plot_count):
        clean, noisy, denoised, idx = results[i]
        
        plt.figure(figsize=(15, 10))
        
        # Plot time domain waveforms
        plt.subplot(3, 1, 1)
        plt.plot(clean)
        plt.title(f"Clean Signal (Index: {idx})")
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(noisy)
        plt.title("Noisy Signal")
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(denoised)
        plt.title("Denoised Signal")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"signal_{idx}_comparison.png"))
        plt.close()

def save_metrics(metrics_list, output_dir="results"):
    """
    Save evaluation metrics.
    
    Args:
        metrics_list: List of metric dictionaries.
        output_dir: Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Calculate averages
    avg_metrics = metrics_df.drop(columns=['index']).mean()
    
    # Save detailed metrics
    # metrics_df.to_csv(os.path.join(output_dir, "detailed_metrics.csv"), index=False)
    
    # Save average metrics
    # avg_metrics_df = pd.DataFrame([avg_metrics], columns=avg_metrics.index)
    # avg_metrics_df.to_csv(os.path.join(output_dir, "average_metrics.csv"), index=False)
    
    # Return average metrics
    return avg_metrics

def save_denoised_signals_csv(results, metrics_list, output_dir="results"):
    """
    Quickly save denoised signal data to CSV file.
    
    Args:
        results: List of tuples (clean, noisy, denoised, idx).
        metrics_list: List of metric dictionaries (including SNR improvement).
        output_dir: Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Pre-allocate numpy array for performance
    num_signals = len(results)
    signal_length = len(results[0][0])  # Assume all signals have the same length
    
    # Create array to store denoised signal data
    denoised_signals = np.zeros((num_signals, signal_length), dtype=np.float32)
    indices = np.zeros(num_signals, dtype=int)
    
    # Extract SNR improvement values from metrics_list and build index map
    snr_improvements = {}
    for metrics in metrics_list:
        snr_improvements[metrics['index']] = metrics['snr_improvement']
    
    # Batch fill data
    for i, (clean, noisy, denoised, idx) in enumerate(results):
        denoised_signals[i] = denoised
        indices[i] = idx
    
    # Create column names: first column blank, then 0,1,2...4000, Ptime, SNR_Improvement
    signal_columns = [str(i) for i in range(signal_length)]
    
    # Create DataFrame
    denoised_df = pd.DataFrame(denoised_signals, columns=signal_columns)
    
    # Add Ptime column (set to -1)
    denoised_df['Ptime'] = -1
    
    # Add SNR_Improvement column
    snr_improvement_values = [snr_improvements[idx] for idx in indices]
    denoised_df['SNR_Improvement'] = snr_improvement_values
    
    # Insert index column at the beginning (corresponding to the first column)
    denoised_df.insert(0, '', indices)  # First column header is blank
    
    # Save denoised signal data
    denoised_df.to_csv(os.path.join(output_dir, "denoised_signals.csv"), index=False)
    
    print(f"Denoised signal data saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Convolutional Autoencoder Denoising Model Performance")
    parser.add_argument("--model_path", type=str, default=None, help="Model path")
    parser.add_argument("--noisy_file", type=str, default=None, help="Noisy signal CSV file path")
    parser.add_argument("--clean_file", type=str, default=None, help="Clean signal CSV file path")
    parser.add_argument("--output_dir", type=str, default="AE/evaluation_results", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to evaluate, 0 for all")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch processing size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Processing device")
    parser.add_argument("--specific_indices", type=str, default=None, help="Specific signal indices to process, comma separated, e.g., '0,1,5,10'")
    parser.add_argument("--max_plots", type=int, default=0, help="Maximum number of signal comparison plots")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_csv", action="store_true", help="Whether to save signal data to CSV file")
    args = parser.parse_args()
    
    # Set random seed
    # np.random.seed(args.random_seed)
    # torch.manual_seed(args.random_seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model_path}")
    model = load_model(args.model_path, device=args.device)
    
    # Prepare data file paths
    clean_file = args.clean_file
    mixed_file = args.noisy_file
    
    # Determine signal indices to process
    indices = None
    if args.specific_indices:
        indices = [int(idx) for idx in args.specific_indices.split(',')]
        print(f"Processing {len(indices)} specified signal indices: {indices}")
    elif args.num_samples > 0:
        print(f"Randomly selecting {args.num_samples} signals for evaluation")
    else:
        print("Processing all signals")
    
    # Load signal pairs
    # When num_samples is 0, pass None to process all signals
    num_samples_param = None if args.num_samples == 0 else args.num_samples
    signal_pairs = load_signal_pair(clean_file, mixed_file, indices, num_samples_param)
    print(f"Loaded {len(signal_pairs)} signal pairs")
    
    # Denoise signals
    print("Starting denoising...")
    results = denoise_signals(model, signal_pairs, device=args.device, batch_size=args.batch_size)
    
    # Calculate evaluation metrics
    print("Calculating evaluation metrics...")
    metrics_list = calculate_metrics(results)
    
    # Save evaluation metrics
    avg_metrics = save_metrics(metrics_list, output_dir=args.output_dir)
    
    # Optional: Save signal data to CSV
    if not args.save_csv:
        print("Saving signal data to CSV file...")
        save_denoised_signals_csv(results, metrics_list, output_dir=args.output_dir)
    
    # Plot results
    print(f"Plotting results (max {args.max_plots})...")
    plot_results(results, output_dir=args.output_dir, max_plots=args.max_plots)
    
    # Calculate confidence intervals
    snr_values = [m['snr_improvement'] for m in metrics_list]
    mse_values = [m['mse_improvement'] for m in metrics_list]
    corr_values = [m['correlation'] for m in metrics_list]
    spec_values = [m['spectral_distortion'] for m in metrics_list]
    energy_values = [m['energy_percentage'] for m in metrics_list]
    
    # Calculate mean and error margin for 95% confidence interval
    snr_mean, snr_error = calculate_confidence_interval(snr_values)
    mse_mean, mse_error = calculate_confidence_interval(mse_values)
    corr_mean, corr_error = calculate_confidence_interval(corr_values)
    spec_mean, spec_error = calculate_confidence_interval(spec_values)
    energy_mean, energy_error = calculate_confidence_interval(energy_values)
    
    # Print average metrics
    print("\nAverage Evaluation Metrics (95% CI):")
    print(f"  Avg SNR Improvement: {snr_mean:.2f} \u00b1 {snr_error:.2f} dB")
    print(f"  Avg MSE Improvement: {mse_mean:.6f} \u00b1 {mse_error:.6f}")
    print(f"  Avg Correlation: {corr_mean:.4f} \u00b1 {corr_error:.4f}")
    print(f"  Avg Spectral Distortion: {spec_mean:.6f} \u00b1 {spec_error:.6f}")
    print(f"  Avg Energy Percentage: {energy_mean:.2f} \u00b1 {energy_error:.2f}%")
    
    print(f"\nEvaluation results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()