import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import argparse
from sklearn.model_selection import train_test_split
from scipy import signal

# Import model definition
from model import ConvAutoencoder, normalize_signal

# Timing function
def tic():
    global start_time
    start_time = time.time()

def toc():
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds.")
    return elapsed_time

def load_data(clean_file, noisy_file, test_size=0.2, random_state=42):
    """
    Load clean and noisy signal data.
    
    Args:
        clean_file: Path to clean signal CSV file.
        noisy_file: Path to noisy signal CSV file.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed.
        
    Returns:
        Training and testing data.
    """
    # Read CSV files
    clean_df = pd.read_csv(clean_file)
    noisy_df = pd.read_csv(noisy_file)
    
    # Extract signal data (specifically read data from column 1 to 4001)
    clean_data = clean_df.iloc[:, 1:4001].values  # Columns 1 to 4000 (0-based index)
    noisy_data = noisy_df.iloc[:, 1:4001].values
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        noisy_data, clean_data, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=100, device='cuda', save_path='models', 
                patience=10):
    """
    Train the model.
    
    Args:
        model: Model instance.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        num_epochs: Number of training epochs.
        device: Training device.
        save_path: Path to save the model.
        patience: Patience for early stopping.
        
    Returns:
        Trained model and training history.
    """
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'training_time': 0
    }
    
    # Early stopping setup
    best_val_loss = float('inf')
    no_improve_epochs = 0
    
    # Start timer
    tic()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training phase
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                val_loss = criterion(outputs, target)
                running_val_loss += val_loss.item()
                
        val_loss = running_val_loss / len(val_loader)
        history['val_loss'].append(val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.6f}, '
              f'Val Loss: {val_loss:.6f}')
        
        # Check if model needs to be saved
        if val_loss < best_val_loss and epoch > 9:
            best_val_loss = val_loss
            history['best_epoch'] = epoch
            no_improve_epochs = 0
            
            # Save model
            model_path = os.path.join(save_path, "conv_autoencoder.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved: {model_path}")
        else:
            no_improve_epochs += 1
            
        # Early stopping check
        if no_improve_epochs >= patience:
            print(f"Early stopping: No improvement for {patience} epochs, stopping training.")
            break
    
    # Calculate training time
    history['training_time'] = toc()
    
    # Load best model
    model_path = os.path.join(save_path, "conv_autoencoder.pt")
    model.load_state_dict(torch.load(model_path))
    
    return model, history

def plot_training_history(history, save_path='results'):
    """Plot training history."""
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.axvline(x=history['best_epoch'], color='r', linestyle='--', label='Best Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()

# Add signal-related loss function
class SignalLoss(nn.Module):
    """Signal loss function combining MSE and correlation coefficient."""
    def __init__(self, mse_weight=0.7, corr_weight=0.3):
        super(SignalLoss, self).__init__()
        self.mse_weight = mse_weight
        self.corr_weight = corr_weight
        self.mse = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        # MSE loss
        mse_loss = self.mse(y_pred, y_true)
        
        # Correlation coefficient loss (1 - correlation coefficient)
        batch_size = y_pred.size(0)
        corr_loss = torch.zeros(1, device=y_pred.device)
        
        for i in range(batch_size):
            x = y_true[i]
            y = y_pred[i]
            
            # Calculate mean
            mx = torch.mean(x)
            my = torch.mean(y)
            
            # Center data
            xm, ym = x - mx, y - my
            
            # Calculate correlation coefficient
            r_num = torch.sum(xm * ym)
            r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
            
            # Avoid division by zero
            r_den = torch.clamp(r_den, min=1e-8)
            
            r = r_num / r_den
            
            # Correlation loss (aim for correlation close to 1)
            corr_loss += (1.0 - r)
            
        corr_loss = corr_loss / batch_size
        
        # Combine losses
        total_loss = self.mse_weight * mse_loss + self.corr_weight * corr_loss
        
        return total_loss

def main():
    parser = argparse.ArgumentParser(description="Train Convolutional Autoencoder Denoising Model")
    parser.add_argument("--data_dir", type=str, default="../data", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--save_dir", type=str, default="AE/models", help="Model save directory")
    parser.add_argument("--results_dir", type=str, default="AE/results", help="Results save directory")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size proportion")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--signal_loss", action="store_true", help="Use signal-specific loss function")
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Prepare data file paths
    clean_file = os.path.join(args.data_dir, "clean", "clean_microseismic.csv")
    noisy_file = os.path.join(args.data_dir, "mixed_dataset", "-10_10_combined.csv")
    
    # Load and split data
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_data(
        clean_file, noisy_file, 
        test_size=args.test_size, 
        random_state=args.random_seed
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model
    input_length = X_train.shape[1]
    print(f"Input feature dimension: {input_length}")
    model = ConvAutoencoder(input_length=input_length).to(args.device)
    
    # Define loss function and optimizer
    if args.signal_loss:
        print("Using signal-specific loss function")
        criterion = SignalLoss(mse_weight=0.7, corr_weight=0.3)
    else:
        print("Using MSE loss function")
        criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    print("Starting model training...")
    model, history = train_model(
        model, train_loader, val_loader, 
        criterion, optimizer, 
        num_epochs=args.epochs, 
        device=args.device,
        save_path=args.save_dir,
        patience=args.patience
    )
    
    # Plot training history
    plot_training_history(history, save_path=args.results_dir)
    
    print("Training completed!")
    print(f"Best model saved at: {args.save_dir}/conv_autoencoder.pt")
    print(f"Training history plot saved at: {args.results_dir}/training_history.png")
    print(f"Training time: {history['training_time']:.2f} seconds")

if __name__ == "__main__":
    main() 