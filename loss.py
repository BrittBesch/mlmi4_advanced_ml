"""
Prototypical Networks loss (Snell et al., 2017).

Implements J(phi): negative log-probability of the correct class when
using Euclidean distance to class prototypes. Extensible for Mahalanobis
distance (extension: distance metric limitation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Squared Euclidean distance between two sets of vectors.

    Args:
        x: (N, D) query embeddings
        y: (M, D) prototype embeddings

    Returns:
        (N, M) squared distances
    """
    n, d = x.size(0), x.size(1)
    m = y.size(0)
    if y.size(1) != d:
        raise ValueError("x and y must have same embedding dim")
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


# ============================================================
# EXTENSION: Learnable Mahalanobis distance variants
# ============================================================
#
# Background
# ----------
# Euclidean distance treats every embedding dimension equally:
#     d²(x, y) = ||x - y||²  =  (x-y)ᵀ I (x-y)
#
# Mahalanobis distance replaces the identity with a learned PSD matrix M:
#     d²(x, y) = (x-y)ᵀ M (x-y),  M = LᵀL ≥ 0
#
# For zero-shot learning this is especially useful: prototypes are built
# from *attribute* vectors, not visual examples, so the natural geometry
# of the embedding space may not align with Euclidean distance. Letting
# the model learn a better metric directly compensates for that gap.
#
# Experiments
# -----------
#   Experiment 1 (baseline)  : --distance euclidean  → standard Euclidean (0 extra params)
#   Experiment 2 (EXTENSION) : --distance diagonal   → diagonal M        (d extra params)
#   Experiment 3 (EXTENSION) : --distance lowrank    → low-rank M        (d×r extra params)


class DiagonalMahalanobisDistance(nn.Module):
    """
    EXTENSION – Experiment 2: Diagonal Mahalanobis distance.

    M = diag(s²),  so  d²(x, y) = Σᵢ sᵢ² (xᵢ - yᵢ)²  =  ||s ⊙ (x-y)||²

    This is identical to Euclidean distance after a learned per-dimension
    rescaling s = exp(log_s).  Because each dimension gets its own scale
    factor, the model can up-weight informative embedding dimensions and
    down-weight noisy ones.

    Parameters: d  (one log-scale per embedding dimension)
    Initialisation: log_s = 0 → s = 1 → identical to Euclidean at the
    start of training, so switching from Euclidean baseline is safe.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Stored in log-space to keep s strictly positive during optimisation.
        self.log_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, D) query embeddings
            y: (M, D) prototype embeddings
        Returns:
            (N, M) diagonal Mahalanobis distances
        """
        s = self.log_scale.exp()          # (D,) per-dimension positive scale
        xs = x * s                         # (N, D) scaled queries
        ys = y * s                         # (M, D) scaled prototypes
        n, d = xs.size(0), xs.size(1)
        m = ys.size(0)
        xs = xs.unsqueeze(1).expand(n, m, d)
        ys = ys.unsqueeze(0).expand(n, m, d)
        return torch.pow(xs - ys, 2).sum(2)


class LowRankMahalanobisDistance(nn.Module):
    """
    EXTENSION – Experiment 3: Low-rank Mahalanobis distance.

    M = LᵀL where L has shape (rank, dim), so:
        d²(x, y) = (x-y)ᵀ LᵀL (x-y)  =  ||L(x-y)||²

    Unlike the diagonal variant this can capture *correlations* between
    embedding dimensions.  Using a low rank (r ≪ d) keeps the parameter
    count manageable and acts as implicit regularisation:
        r=64, d=1024  →  65,536 extra parameters (vs 1,048,576 for full M)

    Parameters: rank × dim
    Initialisation: small random weights scaled by 1/√d so that the
    projected distances start in a similar range to Euclidean distances.
    """

    def __init__(self, dim: int, rank: int = 64):
        super().__init__()
        # L: (rank, dim) projection matrix.  LᵀL is the metric tensor.
        self.L = nn.Parameter(torch.randn(rank, dim) * (1.0 / dim ** 0.5))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, D) query embeddings
            y: (M, D) prototype embeddings
        Returns:
            (N, M) low-rank Mahalanobis distances
        """
        # Project both sets into rank-dimensional space via L.
        xL = x @ self.L.t()               # (N, rank)
        yL = y @ self.L.t()               # (M, rank)
        n, r = xL.size(0), xL.size(1)
        m = yL.size(0)
        xL = xL.unsqueeze(1).expand(n, m, r)
        yL = yL.unsqueeze(0).expand(n, m, r)
        return torch.pow(xL - yL, 2).sum(2)


def prototypical_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    n_support: int,
    distance_fn=euclidean_dist,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Few-shot prototypical loss: prototypes = mean of support embeddings per class;
    loss = NLL over distances from query embeddings to prototypes.

    Args:
        embeddings: (B, D) model embeddings for a batch (support + query)
        labels: (B,) class index for each sample; first n_support per class are support
        n_support: number of support samples per class
        distance_fn: function (x, y) -> (N, M) distances; default euclidean_dist

    Returns:
        loss: scalar NLL
        acc: scalar accuracy on query set
    """
    device = embeddings.device
    labels_cpu = labels.cpu()
    embeddings_cpu = embeddings.cpu()

    classes = torch.unique(labels_cpu, sorted=True)
    n_classes = len(classes)

    def support_idxs(c):
        return labels_cpu.eq(c).nonzero(as_tuple=True)[0][:n_support]

    support_idxs_list = [support_idxs(c) for c in classes]
    prototypes = torch.stack([embeddings_cpu[idxs].mean(0) for idxs in support_idxs_list])

    n_query = labels_cpu.eq(classes[0].item()).sum().item() - n_support
    query_idxs = torch.stack(
        [labels_cpu.eq(c).nonzero(as_tuple=True)[0][n_support:] for c in classes]
    ).view(-1)

    query_emb = embeddings.index_select(0, query_idxs.to(device))
    prototypes_dev = prototypes.to(device)
    dists = distance_fn(query_emb, prototypes_dev)

    log_p_y = F.log_softmax(-dists, dim=1)
    log_p_y = log_p_y.view(n_classes, n_query, -1)

    target_inds = (
        torch.arange(n_classes, device=device, dtype=torch.long)
        .view(n_classes, 1, 1)
        .expand(n_classes, n_query, 1)
    )
    loss = -log_p_y.gather(2, target_inds).squeeze(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc = y_hat.eq(target_inds.squeeze(-1)).float().mean()

    return loss, acc


def prototypical_loss_from_prototypes(
    query_embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    distance_fn=euclidean_dist,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loss when prototypes are given (e.g. zero-shot: prototypes from attributes).
    Used for CUB zero-shot: prototypes = (n_classes, D), labels = class indices in [0, n_classes-1].

    Args:
        query_embeddings: (N, D)
        prototypes: (C, D)
        labels: (N,) in [0, C-1]
        distance_fn: (x, y) -> distances

    Returns:
        loss: scalar NLL
        acc: scalar accuracy
    """
    dists = distance_fn(query_embeddings, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1)
    loss = F.nll_loss(log_p_y, labels)
    _, y_hat = log_p_y.max(1)
    acc = y_hat.eq(labels).float().mean()
    return loss, acc


class PrototypicalLoss(nn.Module):
    """Module wrapper for few-shot prototypical loss (support + query in one batch)."""

    def __init__(self, n_support: int, distance_fn=euclidean_dist):
        super().__init__()
        self.n_support = n_support
        self.distance_fn = distance_fn

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return prototypical_loss(
            embeddings, labels, self.n_support, distance_fn=self.distance_fn
        )
