"""
Zero-shot training and evaluation for CUB using precomputed features (Table 3 replication).

Implements Snell et al. (2017) zero-shot CUB experiment with the precomputed
GoogLeNet image features and 312-dim CUB attribute vectors.

Architecture (paper):
  f: linear(1024 → z_dim)   image encoder
  g: linear(312  → z_dim)   class attribute encoder; prototypes L2-normalised

Image features: data/cvpr2016_cub/images/*.t7  (precomputed GoogLeNet, Reed et al. 2016)
  Train: average of all 10 crops per image
  Test:  middle crop of original image only (paper: "At test time we use only the middle crop")

Attributes: data/CUB_200_2011/attributes/class_attribute_labels_continuous.txt
  312-dim continuous attribute vectors, one per class

Training (two-phase as in paper):
  Phase 1: episodic 50-way, 10-query on train classes; early-stop on val loss
  Phase 2: retrain on train+val for best_epoch epochs; evaluate on test

Usage:
  python src/training/train_zeroshot_precomputed.py \
      --data_root data/cvpr2016_cub \
      --cub_root data/CUB_200_2011
"""

import argparse
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from loss import (
    prototypical_loss_from_prototypes,
    euclidean_dist,
    build_distance,
)
from src.data_loader.dataloader_cub import (
    CUBPrecomputedDataset, IMAGE_DIM,
)
from src.utils.device import get_device
from src.utils.seed import set_seed


class LinearImageEncoder(nn.Module):
    """f_phi: linear map from precomputed image features to embedding space."""

    def __init__(self, in_dim: int = IMAGE_DIM, z_dim: int = 1024):
        super().__init__()
        self.fc = nn.Linear(in_dim, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class LinearAuxEncoder(nn.Module):
    """g_phi: linear map from class auxiliary features to embedding space."""

    def __init__(self, aux_dim: int, z_dim: int = 1024):
        super().__init__()
        self.fc = nn.Linear(aux_dim, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def run_episodes(
    model_img,
    model_aux,
    aux_features_split: np.ndarray,
    optimizer,
    dataset: CUBPrecomputedDataset,
    device: torch.device,
    distance_fn,
    n_episodes: int,
    n_way: int,
    n_query: int,
    do_backward: bool,
) -> tuple:
    """
    Run episodic training or validation for one epoch.

    Each episode:
      1. Sample n_way classes from the split
      2. Sample n_query images per class as queries (zero-shot: no support set)
      3. Prototypes = g(aux_features[episode_classes])
      4. Loss = prototypical NLL over query embeddings vs prototypes
    """
    model_img.train(do_backward)
    model_aux.train(do_backward)
    if isinstance(distance_fn, nn.Module):
        distance_fn.train(do_backward)

    aux_tensor = torch.tensor(aux_features_split, dtype=torch.float32, device=device)

    # Build per-class index lists once
    indices_per_class: dict = defaultdict(list)
    for idx in range(len(dataset)):
        indices_per_class[int(dataset.labels[idx])].append(idx)
    all_classes = [c for c in indices_per_class if len(indices_per_class[c]) > 0]

    total_loss, total_acc = 0.0, 0.0

    for _ in tqdm(range(n_episodes), desc="episodes", leave=False):
        n_avail = len(all_classes)
        this_way = min(n_way, n_avail)
        episode_classes = random.sample(all_classes, this_way)

        # Collect query indices and local labels
        episode_indices, episode_labels = [], []
        for local_idx, c in enumerate(episode_classes):
            pool = indices_per_class[c]
            chosen = random.sample(pool, n_query) if len(pool) >= n_query \
                else [random.choice(pool) for _ in range(n_query)]
            episode_indices.extend(chosen)
            episode_labels.extend([local_idx] * n_query)

        # Build tensors — use dataset[i] so __getitem__ handles crop selection
        feats = torch.stack([
            dataset[i][0] for i in episode_indices
        ]).to(device)
        labels = torch.tensor(episode_labels, dtype=torch.long, device=device)
        cls_idx = torch.tensor(episode_classes, dtype=torch.long, device=device)

        # Forward
        if do_backward:
            prototypes = _l2_normalize(model_aux(aux_tensor[cls_idx]))
            z = model_img(feats)
            loss, acc = prototypical_loss_from_prototypes(z, prototypes, labels, distance_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                prototypes = _l2_normalize(model_aux(aux_tensor[cls_idx]))
                z = model_img(feats)
                loss, acc = prototypical_loss_from_prototypes(z, prototypes, labels, distance_fn)

        total_loss += loss.item()
        total_acc += acc.item()

    return total_loss / max(n_episodes, 1), total_acc / max(n_episodes, 1)


@torch.no_grad()
def evaluate(
    model_img,
    model_aux,
    aux_features_test: np.ndarray,
    dataset: CUBPrecomputedDataset,
    device: torch.device,
    distance_fn,
) -> float:
    """Zero-shot accuracy: prototypes from aux features, queries from test images."""
    model_img.eval()
    model_aux.eval()
    if isinstance(distance_fn, nn.Module):
        distance_fn.eval()

    aux_tensor = torch.tensor(aux_features_test, dtype=torch.float32, device=device)
    prototypes = _l2_normalize(model_aux(aux_tensor))   # (n_test_classes, z_dim)

    # dataset is test_time=True so __getitem__ returns fixed middle crop
    all_feats = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
    all_labels = torch.from_numpy(dataset.labels)

    z = model_img(all_feats)   # (N, z_dim)
    dists = distance_fn(z, prototypes)   # (N, n_test_classes)
    pred = dists.argmin(1).cpu()
    acc = (pred == all_labels).float().mean().item()
    
    
    return acc


@torch.no_grad()
def evaluate_episodic(
    model_img,
    model_aux,
    aux_features_test: np.ndarray,
    dataset: CUBPrecomputedDataset,
    device: torch.device,
    distance_fn,
    n_episodes: int,
    n_way: int,
    n_query: int,
) -> tuple[float, float]:
    """
    Episodic zero-shot evaluation with a 95% confidence interval.

    Treats each randomly sampled test episode as one sample, computes the
    mean episode accuracy and a 95% CI over episodes using:
        CI = 1.96 * std(accs) / sqrt(len(accs))
    """
    model_img.eval()
    model_aux.eval()
    if isinstance(distance_fn, nn.Module):
        distance_fn.eval()

    aux_tensor = torch.tensor(aux_features_test, dtype=torch.float32, device=device)

    # Build per-class index lists once (on the fixed test split)
    indices_per_class: dict = defaultdict(list)
    for idx in range(len(dataset)):
        indices_per_class[int(dataset.labels[idx])].append(idx)
    all_classes = [c for c in indices_per_class if len(indices_per_class[c]) > 0]

    accs = []

    for _ in range(n_episodes):
        n_avail = len(all_classes)
        this_way = min(n_way, n_avail)
        episode_classes = random.sample(all_classes, this_way)

        # Sample query indices and local labels for this episode
        episode_indices, episode_labels = [], []
        for local_idx, c in enumerate(episode_classes):
            pool = indices_per_class[c]
            chosen = random.sample(pool, n_query) if len(pool) >= n_query \
                else [random.choice(pool) for _ in range(n_query)]
            episode_indices.extend(chosen)
            episode_labels.extend([local_idx] * n_query)

        feats = torch.stack([dataset[i][0] for i in episode_indices]).to(device)
        labels = torch.tensor(episode_labels, dtype=torch.long, device=device)
        cls_idx = torch.tensor(episode_classes, dtype=torch.long, device=device)

        prototypes = _l2_normalize(model_aux(aux_tensor[cls_idx]))
        z = model_img(feats)
        _, acc = prototypical_loss_from_prototypes(z, prototypes, labels, distance_fn)

        accs.append(acc.item())

    mean_acc = float(np.mean(accs))
    # 95% CI radius assuming approximate normality of episode accuracies
    ci95 = 1.96 * float(np.std(accs, ddof=1)) / np.sqrt(len(accs))
    return mean_acc, ci95


def build_models(aux_dim: int, z_dim: int, device: torch.device, dist_cfg: dict):
    model_img = LinearImageEncoder(in_dim=IMAGE_DIM, z_dim=z_dim).to(device)
    model_aux = LinearAuxEncoder(aux_dim=aux_dim, z_dim=z_dim).to(device)
    distance_fn = build_distance(dist_cfg, z_dim, device)
    return model_img, model_aux, distance_fn


def build_optimizer(model_img, model_aux, distance_fn, lr: float, weight_decay: float):
    dist_params = list(distance_fn.parameters()) if isinstance(distance_fn, nn.Module) else []
    return torch.optim.Adam(
        list(model_img.parameters()) + list(model_aux.parameters()) + dist_params,
        lr=lr,
        weight_decay=weight_decay,
    )


def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot CUB with precomputed features (Table 3)"
    )
    parser.add_argument("--config", type=str, default="",
                        help="Path to cub_config.yaml (optional)")
    parser.add_argument("--data_root", type=str, default="data/cvpr2016_cub")
    parser.add_argument("--cub_root", type=str, default="data/CUB_200_2011",
                        help="Path to CUB_200_2011 (attributes only)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--z_dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_way", type=int, default=50,
                        help="Episode n-way (paper: 50)")
    parser.add_argument("--n_query", type=int, default=10,
                        help="Queries per class per episode (paper: 10)")
    parser.add_argument("--n_episodes", type=int, default=100,
                        help="Training episodes per epoch (not specified in paper)")
    parser.add_argument("--val_episodes", type=int, default=20,
                        help="Validation episodes per epoch (not specified in paper)")
    parser.add_argument("--test_episodes", type=int, default=1000,
                        help="Number of test episodes for episodic CI evaluation")
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--exp_name", type=str, default="cub_zeroshot_precomputed")
    parser.add_argument("--distance", type=str, default="euclidean",
                        choices=["euclidean", "diagonal", "lowrank"])
    parser.add_argument("--lowrank_r", type=int, default=64)
    args = parser.parse_args()

    cfg = load_config(args.config or os.path.join(PROJECT_ROOT, "configs",
                                                   "cub_config.yaml"))
    if cfg:
        d = cfg.get("data", {})
        t = cfg.get("training", {})
        e = cfg.get("experiment", {})
        args.data_root = d.get("data_root", args.data_root)
        args.cub_root = d.get("cub_root", args.cub_root)
        args.epochs = t.get("epochs", args.epochs)
        args.lr = t.get("lr", args.lr)
        args.n_way = t.get("n_way", args.n_way)
        args.n_query = t.get("n_query", args.n_query)
        args.n_episodes = t.get("n_episodes", args.n_episodes)
        args.val_episodes = t.get("val_episodes", args.val_episodes)
        es = t.get("early_stopping", {}) or {}
        args.early_stopping_patience = es.get("patience", args.early_stopping_patience)
        args.early_stopping_min_delta = es.get("min_delta", args.early_stopping_min_delta)
        args.seed = e.get("seed", args.seed)
        args.save_dir = e.get("save_dir", args.save_dir)
        args.exp_name = e.get("exp_name", args.exp_name)
        if "z_dim" in cfg.get("model", {}):
            args.z_dim = cfg["model"]["z_dim"]

    set_seed(args.seed)
    device = get_device()

    aux_dim = 312
    dist_cfg = {"distance": args.distance, "lowrank_r": args.lowrank_r}

    print(f"Loading CUB precomputed features from {args.data_root}")
    print(f"Auxiliary modality: attributes ({aux_dim}-dim)")
    print(f"Distance metric: {args.distance}")

    cub_root = args.cub_root

    # Train: average all 10 crops. Val/test: middle crop only (paper).
    train_ds = CUBPrecomputedDataset(
        args.data_root, split="train",
        cub_root=cub_root, test_time=False,
    )
    val_ds = CUBPrecomputedDataset(
        args.data_root, split="val",
        cub_root=cub_root, test_time=True,
    )
    test_ds = CUBPrecomputedDataset(
        args.data_root, split="test",
        cub_root=cub_root, test_time=True,
    )

    print(f"Train: {len(train_ds.class_names)} classes, {len(train_ds)} images")
    print(f"Val:   {len(val_ds.class_names)} classes, {len(val_ds)} images")
    print(f"Test:  {len(test_ds.class_names)} classes, {len(test_ds)} images")

    # Phase 1: train on train classes, early-stop on val
    model_img, model_aux, distance_fn = build_models(aux_dim, args.z_dim, device, dist_cfg)
    optimizer = build_optimizer(model_img, model_aux, distance_fn, args.lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = run_episodes(
            model_img, model_aux, train_ds.aux_features,
            optimizer, train_ds, device, distance_fn,
            n_episodes=args.n_episodes, n_way=args.n_way,
            n_query=args.n_query, do_backward=True,
        )
        val_loss, val_acc = run_episodes(
            model_img, model_aux, val_ds.aux_features,
            optimizer, val_ds, device, distance_fn,
            n_episodes=args.val_episodes, n_way=min(50, len(val_ds.class_names)),
            n_query=args.n_query, do_backward=False,
        )

        print(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        improved = val_loss < best_val_loss - float(args.early_stopping_min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if args.early_stopping_patience and epochs_no_improve >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch+1} "
                f"(best epoch={best_epoch}, val_loss={best_val_loss:.4f})"
            )
            break

    if best_epoch <= 0:
        best_epoch = 1

    # Phase 2: retrain on train+val for best_epoch epochs, evaluate on test
    print(f"\nPhase 2: retraining on train+val for {best_epoch} epoch(s)...")
    trainval_ds = CUBPrecomputedDataset(
        args.data_root, split="trainval",
        cub_root=cub_root, test_time=False,
    )
    print(f"Train+val: {len(trainval_ds.class_names)} classes, {len(trainval_ds)} images")

    set_seed(args.seed)
    model_img, model_aux, distance_fn = build_models(aux_dim, args.z_dim, device, dist_cfg)
    optimizer = build_optimizer(model_img, model_aux, distance_fn, args.lr, weight_decay=1e-5)

    for epoch in range(best_epoch):
        train_loss, train_acc = run_episodes(
            model_img, model_aux, trainval_ds.aux_features,
            optimizer, trainval_ds, device, distance_fn,
            n_episodes=args.n_episodes, n_way=args.n_way,
            n_query=args.n_query, do_backward=True,
        )
        print(f"[Retrain {epoch+1}/{best_epoch}] loss={train_loss:.4f}  acc={train_acc:.4f}")

    test_mean, test_ci = evaluate_episodic(
        model_img,
        model_aux,
        test_ds.aux_features,
        test_ds,
        device,
        distance_fn,
        n_episodes=args.test_episodes,
        n_way=len(test_ds.class_names),
        n_query=args.n_query,
    )
    print(
        f"\nFinal zero-shot test accuracy ({len(test_ds.class_names)}-way): "
        f"{test_mean:.4f} ± {test_ci:.4f} (95% CI over {args.test_episodes} episodes)"
    )

    # Save checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = {
        "best_epoch": best_epoch,
        "aux_type": "attributes",
        "distance": args.distance,
        "z_dim": args.z_dim,
        "img_encoder": model_img.state_dict(),
        "aux_encoder": model_aux.state_dict(),
        "test_acc": test_mean,
        "test_ci95": test_ci,
    }
    if isinstance(distance_fn, nn.Module):
        ckpt["distance_fn"] = distance_fn.state_dict()

    path = os.path.join(
        args.save_dir,
        f"{args.exp_name}_attributes_{args.distance}_best.pt"
    )
    torch.save(ckpt, path)
    print(f"Checkpoint saved to {path}")


if __name__ == "__main__":
    main()
