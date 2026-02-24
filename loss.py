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
