"""
Speech Embedding Models for FS-KWS.

Contains 2 architectures for comparison:
1. SpeechC64: Baseline 4-block 2D CNN.
2. TCResNet8: Temporal Convolution ResNet using 1D dilated convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the original ProtoNet from model.py
from model import ProtoNetEncoder

# ==================================================================
# 1. BASELINE: ProtoNet 4-Layer CNN (Traditional 2D Convolution)
# ==================================================================
class SpeechC64(nn.Module):
    """
    The baseline ProtoNet 4-Layer CNN.
    
    The encoder treats the [B, 1, 40, 51] MFCC tensor like a 1-channel grayscale 
    image. It uses standard 3x3 2D convolutions to look for visual signals in 
    the spectrogram.
    """
    def __init__(self, in_channels=1, hidden_dim=64, dropblock_size=0, dropblock_prob=0.1):
        super().__init__()
        self.encoder = ProtoNetEncoder(
            in_channels=in_channels, 
            hidden_dim=hidden_dim, 
            dropblock_size=dropblock_size, 
            dropblock_prob=dropblock_prob
        )

    def forward(self, x):
        """
        Args:
            x: Spectrograms of shape (B, 1, 40, 51) -> [Batch, Channel, Freq, Time]
        Returns: 
            Embeddings of shape (B, D)
        """
        return self.encoder(x)

# ==================================================================
# 2. EXTENSION: TC-ResNet-8 (1D Temporal Convolution)
# ==================================================================
class TCResidualBlock(nn.Module):
    """
    A single Residual Block using 1D Dilated Convolutions.
    
    Instead of a 2D 3x3 kernel, this uses a 1D kernel of size 3 that spans 
    across the time axis. The 'dilation' parameter forces the kernel to skip 
    time steps to capture long-range phonetics without adding extra parameters.
    """
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        
        # Calculate padding to keep sequence length consistent when stride=1
        # padding = (kernel_size - 1) * dilation // 2
        padding = dilation  

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, 
            stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection to bypass the conv block
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # If the dimensions change, use a 1x1 Conv1D to align the shortcut dimensions
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # Save a copy of input data x
        residual = self.shortcut(x)
        
        # Feed input into first 1D Conv, normalize w/ ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # Feed data into second 1D Conv and normalize w/o ReLU 
        out = self.bn2(self.conv2(out))
        
        # Add output from 2 Conv with input x
        out += residual
        # Apply ReLU after 2 streams of data are merged
        return F.relu(out)

class TCResNet8(nn.Module):
    """
    Temporal Convolution ResNet-8 for FS-KWS.
    
    Architecture Design (8 parameterized layers):
    - 1 Initial Conv1D layer
    - 3 Residual Blocks x 2 Conv1D layers each = 6 layers
    - 1 Final Fully Connected / Embedding layer
    """
    def __init__(self, embedding_dim=64):
        super().__init__()
        
        # Initial Temporal Convolution
        # Take 40 raw MFCC frequencies & Extract 16 basic acoustic features (e.g. volume spikes, tones)
        # Input shape: (B, 40, 51) -> [Batch, Freq_Channels, Time]
        self.conv1 = nn.Conv1d(in_channels=40, out_channels=16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        
        # Dilated Residual Blocks
        # The dilation parameter increases exponentially to capture a wider span in the 1-second audio
        # stride=2 to skip every other step, i.e. cutting time in half (for translation invariance)
        # E.g: The label is still X whether or not X is is said at 0 seconds or 0.5 seconds.
        self.block1 = TCResidualBlock(in_channels=16, out_channels=24, stride=2, dilation=1)
        self.block2 = TCResidualBlock(in_channels=24, out_channels=32, stride=2, dilation=2)
        self.block3 = TCResidualBlock(in_channels=32, out_channels=48, stride=2, dilation=4)
        
        # Global Average Pooling flattens the remaining time steps into a single scalar per channel
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Final linear projection to match the C64 embedding dimension
        self.fc = nn.Linear(48, embedding_dim)

    def forward(self, x):
        """
        Args:
            x: Spectrograms of shape (B, 1, 40, 51) -> [Batch, 1_Channel, MFCC_Bins, Time]
        """
        # The 1D Shift: (B, 1, 40, 51) -> (B, 40, 51)
        x = x.squeeze(1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.avgpool(x) # Shape: (B, 48, 1) to flatten vector by averaging time axis using AdaptiveAvgPool1d
        x = x.view(x.size(0), -1) # Shape: (B, 48)
        
        embedding = self.fc(x) # Shape: (B, embedding_dim)
        
        return embedding