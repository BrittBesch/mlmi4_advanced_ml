"""
Prototypical Networks for Few-shot Learning
Model implementation based on paper reviewed.

The embedding architecture is four convolutional blocks, each comprising:
  - 64-filter 3x3 convolution
  - Batch normalization
  - ReLU nonlinearity
  - 2x2 max-pooling

For 28x28 Omniglot images  -> 64-dim output space
For 84x84 miniImageNet images -> 1600-dim output space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

#####################################################################################################################
## Extension: Introducing Cutout and DropBlock techniques for Extension pt 1
#####################################################################################################################

class DropBlock(nn.Module):
    """
    DropBlock: drops contiguous rectangular regions from feature maps
    instead of individual pixels.
    Args:
        block_size: Size of the square region to drop (default: 5  - can change this to whatever we fancy, could do a variety of combos and report in a table??)
        drop_prob: Probability of dropping a block (default: 0.1  same as above^)
    """

    def __init__(self, block_size=5, drop_prob=0.1):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        _, _, h, w = x.shape
        valid_h = h - self.block_size + 1
        valid_w = w - self.block_size + 1

        if valid_h <= 0 or valid_w <= 0:
            return x

        gamma = (self.drop_prob / (self.block_size ** 2)) * \
                ((h * w) / (valid_h * valid_w))

        mask = torch.zeros_like(x)
        mask[:, :, :valid_h, :valid_w] = (torch.rand(
            x.shape[0], x.shape[1], valid_h, valid_w, device=x.device
        ) < gamma).float()

        block_mask = F.max_pool2d(
            mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2,
        )
        block_mask = block_mask[:, :, :h, :w]
        keep_mask = 1.0 - block_mask

        count_ones = keep_mask.sum()
        if count_ones == 0:
            return torch.zeros_like(x)

        # Cap rescaling to prevent explosion on small feature maps
        scale = min((keep_mask.numel() / count_ones).item(), 10.0)
        return x * keep_mask * scale

########################################################################################## end of extension code add-ins

class ConvBlock(nn.Module):
    """A single convolutional block: Conv2d -> BatchNorm -> ReLU -> MaxPool -> (DropBlock)."""

    def __init__(self, in_channels, out_channels, drop_block=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop_block = drop_block

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if self.drop_block is not None:
            x = self.drop_block(x)
        return x

### BRINGING IN THE RESNET ENCODER FOR EXTENSION   

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        if in_channels != 3:
            raise ValueError("Resnet18 requires 3 input channels")
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove final so we are just doing feature extraction
        self.embed_dim=512
        
    def forward(self,x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
    
    

class ProtoNetEncoder(nn.Module):
    """
    The embedding function f_phi from the paper.
    Four convolutional blocks with 64 filters each.

    Args:
        in_channels: Number of input channels (1 for Omniglot, 3 for miniImageNet)
        hidden_dim: Number of filters per conv block (default 64)
    """
    def __init__(self, in_channels, hidden_dim=64,
                    dropblock_size=0, dropblock_prob=0.1, backbone='conv4'):
            super().__init__()
            
            if backbone=='resnet18':
                self.encoder = ResNetEncoder(in_channels=in_channels)
                self.embed_dim=512
                
            else:
                # DropBlock on conv blocks 3 & 4 only (leaving early layers alone atm)
                db = DropBlock(block_size=dropblock_size, drop_prob=dropblock_prob) \
                    if dropblock_size > 0 else None

                self.encoder = nn.Sequential(
                    ConvBlock(in_channels, hidden_dim),   # 42x42 ← safe
                    ConvBlock(hidden_dim, hidden_dim),    # 21x21 ← safe
                    ConvBlock(hidden_dim, hidden_dim, drop_block=db),                   # 10x10
                    ConvBlock(hidden_dim, hidden_dim, drop_block=db),                   #  5x5
                )
                self.embed_dim=None
                
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

         
def forward(self, x):
        """
        Args: x: Images of shape (B, C, H, W)
        Returns: Embeddings of shape (B, D)
        """
        x = self.encoder(x)
        return x.view(x.size(0), -1)