import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadConv1D(nn.Module):
    """MultiHead Conv1D layer implemented in PyTorch."""
    def __init__(self, in_channels, out_channels, kernel_size=255, layer_num=5):
        super(MultiHeadConv1D, self).__init__()
        self.layer_num = layer_num
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=kernel_size, 
                      padding='same') 
            for _ in range(layer_num)
        ])
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, in_channels]
        # Convert to PyTorch Conv1d input format
        x = x.transpose(1, 2)  # [batch_size, in_channels, seq_len]
        
        outputs = []
        for i in range(self.layer_num):
            output = self.relu(self.convs[i](x))
            outputs.append(output)
            
        # Concatenate outputs from all heads
        combined = torch.cat(outputs, dim=1)  # Concatenate along channel dimension
        # Convert back to original format
        combined = combined.transpose(1, 2)  # [batch_size, seq_len, out_channels*layer_num]
        return combined

class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder model."""
    def __init__(self, input_length=4000, in_channels=1):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            MultiHeadConv1D(in_channels=in_channels, 
                            out_channels=2,
                            layer_num=5),
            nn.Flatten()
        )
        
        # Decoder (Fully connected layer outputs signal)
        self.decoder = nn.Sequential(
            nn.Linear(input_length * 2 * 5, input_length),  # 5 heads, each with 2 filters
            nn.Tanh()
        )
        
    def forward(self, x):
        # Ensure input shape is correct [batch_size, seq_len, channels]
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Add channel dimension
            
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def denoise(self, x):
        """Denoise for inference."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

def normalize_signal(signal):
    """Normalize signal."""
    max_val = max(np.max(signal), abs(np.min(signal)))
    if max_val > 0:
        return signal / max_val
    return signal

def interpolate_signal(signal, target_length):
    """Interpolate signal to target length."""
    original_length = len(signal)
    indices = np.linspace(0, original_length - 1, target_length)
    indices = np.round(indices).astype(int)
    return signal[indices]

# Function to load model
def load_model(model_path, device='cuda'):
    """Load trained model."""
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
