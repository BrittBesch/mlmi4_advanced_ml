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

from loss import euclidean_dist ### linking with Britt's loss.py code


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

        count = keep_mask.numel()
        count_ones = keep_mask.sum()
        if count_ones == 0:
            return torch.zeros_like(x)

        return x * keep_mask * (count / count_ones)


class Cutout(nn.Module):
    """
    Cutout: masks out a random square patch on input images.

    Args:
        n_holes: Number of patches to cut out (default: 1)
        length: Side length of each square patch (default: 16)
    """

    def __init__(self, n_holes=1, length=16):
        super().__init__()
        self.n_holes = n_holes
        self.length = length

    def forward(self, x):
        if not self.training:
            return x

        _, _, h, w = x.shape
        mask = torch.ones_like(x)

        for _ in range(self.n_holes):
            cy = torch.randint(0, h, (x.shape[0],))
            cx = torch.randint(0, w, (x.shape[0],))

            for i in range(x.shape[0]):
                y1 = max(0, cy[i] - self.length // 2)
                y2 = min(h, cy[i] + self.length // 2)
                x1 = max(0, cx[i] - self.length // 2)
                x2 = min(w, cx[i] + self.length // 2)
                mask[i, :, y1:y2, x1:x2] = 0.0

        return x * mask
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


class ProtoNetEncoder(nn.Module):
    """
    The embedding function f_phi from the paper.
    Four convolutional blocks with 64 filters each.

    Args:
        in_channels: Number of input channels (1 for Omniglot, 3 for miniImageNet)
        hidden_dim: Number of filters per conv block (default 64)
    """

    def __init__(self, in_channels, hidden_dim=64,
                    dropblock_size=0, dropblock_prob=0.1, cutout_length=0):
            super().__init__()

            # Optional Cutout on input images
            self.cutout = Cutout(n_holes=1, length=cutout_length) \
                if cutout_length > 0 else None

            # DropBlock on conv blocks 3 & 4 only (leaving early layers alone atm)
            db = DropBlock(block_size=dropblock_size, drop_prob=dropblock_prob) \
                if dropblock_size > 0 else None

            self.encoder = nn.Sequential(
                ConvBlock(in_channels, hidden_dim),
                ConvBlock(hidden_dim, hidden_dim),
                ConvBlock(hidden_dim, hidden_dim, drop_block=db),
                ConvBlock(hidden_dim, hidden_dim, drop_block=db),
            )

    def forward(self, x):
        if self.cutout is not None:
            x = self.cutout(x)
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class PrototypicalNetwork(nn.Module):
    """
    Full Prototypical Network.

    Given a support set and query set, computes class prototypes as the mean
    of embedded support examples, then returns distances from queries
    to prototypes. 

    Args:
        in_channels: Number of input channels (1 for Omniglot, 3 for miniImageNet)
        hidden_dim: Number of filters per conv block (default 64)
        distance: Distance function
        embed_dim: Required for mahalanobis (64 for Omniglot, 1600 for miniImageNet)
    """

    def __init__(self, in_channels=1, hidden_dim=64, distance='euclidean',
                 embed_dim=None, dropblock_size=0, dropblock_prob=0.1,
                 cutout_length=0):
        super().__init__()
        self.encoder = ProtoNetEncoder(
            in_channels, hidden_dim,
            dropblock_size=dropblock_size,
            dropblock_prob=dropblock_prob,
            cutout_length=cutout_length,
        )

        if distance == 'euclidean':                   ##### here i have made it dependent on the type of distance function we want to use. We can import the relevant one from another file. Currently routes to euclid dist 
            self.distance_fn = euclidean_dist         ##### matchup with the function defined in loss.py
        elif distance == 'cosine':
            self.distance_fn = cosine_distance
        elif distance == 'mahalanobis':
            if embed_dim is None:
                raise ValueError(
                    "embed_dim is required for Mahalanobis distance. "
                    "Use 64 for Omniglot (28x28), 1600 for miniImageNet (84x84)."
                )
            self.mahalanobis = MahalanobisDistance(embed_dim)
            self.distance_fn = self.mahalanobis
        else:
            raise ValueError(f"Unknown distance function: {distance}")

    def compute_prototypes(self, support_embeddings, n_way, n_support):
        """
        Compute class prototypes as the mean of support embeddings per class as done in paper.

        Args:
            support_embeddings: Shape (n_way * n_support, embed_dim)
            n_way: Number of classes
            n_support: Number of support examples per class

        Returns:
            Prototypes of shape (n_way, embed_dim)
        """
        return support_embeddings.view(n_way, n_support, -1).mean(dim=1)

    def forward(self, support, query, n_way, n_support, n_query):
        """
        Forward pass for an episode.

        Args:
            support: Support set images, shape (n_way * n_support, C, H, W)
            query: Query set images, shape (n_way * n_query, C, H, W)
            n_way: Number of classes in the episode
            n_support: Number of support examples per class
            n_query: Number of query examples per class

        Returns:
            prototypes: Shape (n_way, embed_dim)
            query_embeddings: Shape (n_way * n_query, embed_dim)
            dists: Shape (n_way * n_query, n_way) - distances from queries to prototypes
        """
        support_embeddings = self.encoder(support)
        query_embeddings = self.encoder(query)

        # Compute prototypes: c_k = mean of embeddings for class k (Eq. 1)
        prototypes = self.compute_prototypes(support_embeddings, n_way, n_support)

        # Compute distances from each query to each prototype
        dists = self.distance_fn(query_embeddings, prototypes)

        return prototypes, query_embeddings, dists