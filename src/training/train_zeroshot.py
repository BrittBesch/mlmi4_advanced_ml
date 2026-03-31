"""
Zero-shot training and evaluation for CUB using precomputed features (Table 3 replication).

Implements Snell et al. (2017) zero-shot CUB experiment with the precomputed
GoogLeNet image features and 312-dim CUB attribute vectors.

Architecture (paper):
  f: linear(1024 -> z_dim)   image encoder
  g: linear(312  -> z_dim)   class attribute encoder; prototypes L2-normalised

Training (two-phase as in paper):
  Phase 1: episodic 50-way, 10-query on train classes; early-stop on val loss
  Phase 2: retrain on train+val for best_epoch epochs; evaluate on test
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from loss import prototypical_loss_from_prototypes, build_distance
from src.data_loader.dataloader_cub import CUBPrecomputedDataset, IMAGE_DIM
from src.utils.device import get_device
from src.utils.seed import set_seed

DistanceFn = Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]

_Z_SCORE_95 = 1.96


class DistanceType(Enum):
    EUCLIDEAN = "euclidean"
    DIAGONAL = "diagonal"
    LOWRANK = "lowrank"


@dataclass(frozen=True)
class TrainConfig:
    data_root: Path
    cub_root: Path
    epochs: int
    lr: float
    weight_decay: float
    z_dim: int
    seed: int
    n_way: int
    n_query: int
    n_episodes: int
    val_episodes: int
    test_episodes: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    save_dir: Path
    exp_name: str
    distance: DistanceType
    lowrank_r: int
    phase2_epochs_scale: float

    @classmethod
    def from_args_and_yaml(
        cls, args: argparse.Namespace, yaml_path: Union[Path, str],
    ) -> "TrainConfig":
        """Build training config from CLI args overlaid with YAML values."""
        cfg = _load_yaml(yaml_path)
        d = cfg.get("data", {})
        t = cfg.get("training", {})
        e = cfg.get("experiment", {})
        m = cfg.get("model", {})
        es = t.get("early_stopping") or {}

        return cls(
            data_root=Path(d.get("data_root", args.data_root)),
            cub_root=Path(d.get("cub_root", args.cub_root)),
            epochs=t.get("epochs", args.epochs),
            lr=t.get("lr", args.lr),
            weight_decay=t.get("weight_decay", args.weight_decay),
            z_dim=m.get("z_dim", args.z_dim),
            seed=e.get("seed", args.seed),
            n_way=t.get("n_way", args.n_way),
            n_query=t.get("n_query", args.n_query),
            n_episodes=t.get("n_episodes", args.n_episodes),
            val_episodes=t.get("val_episodes", args.val_episodes),
            test_episodes=t.get("test_episodes", args.test_episodes),
            early_stopping_patience=es.get("patience", args.early_stopping_patience),
            early_stopping_min_delta=es.get("min_delta", args.early_stopping_min_delta),
            save_dir=Path(e.get("save_dir", args.save_dir)),
            exp_name=e.get("exp_name", args.exp_name),
            distance=DistanceType(args.distance),
            lowrank_r=args.lowrank_r,
            phase2_epochs_scale=args.phase2_epochs_scale,
        )


@dataclass(frozen=True)
class ModelCheckpoint:
    img_encoder: Dict[str, torch.Tensor]
    aux_encoder: Dict[str, torch.Tensor]
    distance_fn: Optional[Dict[str, torch.Tensor]] = None


@dataclass(frozen=True)
class Episode:
    features: torch.Tensor
    labels: torch.Tensor
    class_indices: torch.Tensor


class LinearImageEncoder(nn.Module):
    def __init__(self, in_dim: int = IMAGE_DIM, z_dim: int = 1024) -> None:
        """Initialise linear image encoder."""
        super().__init__()
        self.fc = nn.Linear(in_dim, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project image features into embedding space."""
        return self.fc(x)


class LinearAuxEncoder(nn.Module):
    def __init__(self, aux_dim: int, z_dim: int = 1024) -> None:
        """Initialise linear auxiliary-feature encoder."""
        super().__init__()
        self.fc = nn.Linear(aux_dim, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project class auxiliary features into embedding space."""
        return self.fc(x)


def _l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Return L2-normalised tensor along the specified dimension."""
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def _load_yaml(path: Union[Path, str]) -> dict:
    """Load YAML from disk, returning an empty dict if missing."""
    path = Path(path)
    if not path.is_file():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_class_index(dataset: CUBPrecomputedDataset) -> Dict[int, List[int]]:
    """Map class id to dataset indices belonging to that class."""
    index: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        index[int(dataset.labels[idx])].append(idx)
    return {c: idxs for c, idxs in index.items() if idxs}


def _sample_episode(
    all_classes: List[int],
    indices_per_class: Dict[int, List[int]],
    dataset: CUBPrecomputedDataset,
    n_way: int,
    n_query: int,
    device: torch.device,
) -> Episode:
    """Sample one N-way, N-query episode and return tensors on device."""
    if not all_classes:
        raise ValueError("Cannot sample episodes from an empty class set.")
    this_way = min(n_way, len(all_classes))
    episode_classes = random.sample(all_classes, this_way)

    episode_indices: List[int] = []
    episode_labels: List[int] = []
    for local_idx, c in enumerate(episode_classes):
        pool = indices_per_class[c]
        chosen = (
            random.sample(pool, n_query)
            if len(pool) >= n_query
            else [random.choice(pool) for _ in range(n_query)]
        )
        episode_indices.extend(chosen)
        episode_labels.extend([local_idx] * n_query)

    return Episode(
        features=torch.stack([dataset[i][0] for i in episode_indices]).to(device),
        labels=torch.tensor(episode_labels, dtype=torch.long, device=device),
        class_indices=torch.tensor(episode_classes, dtype=torch.long, device=device),
    )


def _snapshot_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone a model state dict for safe checkpointing."""
    return {k: v.clone() for k, v in model.state_dict().items()}


def _snapshot_checkpoint(
    model_img: nn.Module,
    model_aux: nn.Module,
    distance_fn: DistanceFn,
) -> ModelCheckpoint:
    """Capture current model and distance states as a checkpoint."""
    return ModelCheckpoint(
        img_encoder=_snapshot_state(model_img),
        aux_encoder=_snapshot_state(model_aux),
        distance_fn=_snapshot_state(distance_fn) if isinstance(distance_fn, nn.Module) else None,
    )


def _restore_checkpoint(
    checkpoint: ModelCheckpoint,
    model_img: nn.Module,
    model_aux: nn.Module,
    distance_fn: DistanceFn,
) -> None:
    """Restore model and optional distance state from checkpoint."""
    model_img.load_state_dict(checkpoint.img_encoder)
    model_aux.load_state_dict(checkpoint.aux_encoder)
    if isinstance(distance_fn, nn.Module) and checkpoint.distance_fn is not None:
        distance_fn.load_state_dict(checkpoint.distance_fn)


def run_episodes(
    model_img: nn.Module,
    model_aux: nn.Module,
    aux_features_split: np.ndarray,
    optimizer: torch.optim.Optimizer,
    dataset: CUBPrecomputedDataset,
    device: torch.device,
    distance_fn: DistanceFn,
    n_episodes: int,
    n_way: int,
    n_query: int,
    do_backward: bool,
) -> tuple[float, float]:
    """Run episodic train/eval passes and return mean loss and accuracy."""
    model_img.train(do_backward)
    model_aux.train(do_backward)
    if isinstance(distance_fn, nn.Module):
        distance_fn.train(do_backward)

    aux_tensor = torch.tensor(aux_features_split, dtype=torch.float32, device=device)
    indices_per_class = _build_class_index(dataset)
    all_classes = list(indices_per_class.keys())

    total_loss, total_acc = 0.0, 0.0

    for _ in tqdm(range(n_episodes), desc="episodes", leave=False):
        ep = _sample_episode(all_classes, indices_per_class, dataset, n_way, n_query, device)

        if do_backward:
            prototypes = _l2_normalize(model_aux(aux_tensor[ep.class_indices]))
            z = model_img(ep.features)
            loss, acc = prototypical_loss_from_prototypes(z, prototypes, ep.labels, distance_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                prototypes = _l2_normalize(model_aux(aux_tensor[ep.class_indices]))
                z = model_img(ep.features)
                loss, acc = prototypical_loss_from_prototypes(z, prototypes, ep.labels, distance_fn)

        total_loss += loss.item()
        total_acc += acc.item()

    return total_loss / max(n_episodes, 1), total_acc / max(n_episodes, 1)


@torch.no_grad()
def evaluate_episodic(
    model_img: nn.Module,
    model_aux: nn.Module,
    aux_features_test: np.ndarray,
    dataset: CUBPrecomputedDataset,
    device: torch.device,
    distance_fn: DistanceFn,
    n_episodes: int,
    n_way: int,
    n_query: int,
) -> Tuple[float, float]:
    """Returns (mean_accuracy, 95% CI radius) over episodes."""
    model_img.eval()
    model_aux.eval()
    if isinstance(distance_fn, nn.Module):
        distance_fn.eval()

    aux_tensor = torch.tensor(aux_features_test, dtype=torch.float32, device=device)
    indices_per_class = _build_class_index(dataset)
    all_classes = list(indices_per_class.keys())

    accs: List[float] = []

    for _ in range(n_episodes):
        ep = _sample_episode(all_classes, indices_per_class, dataset, n_way, n_query, device)
        prototypes = _l2_normalize(model_aux(aux_tensor[ep.class_indices]))
        z = model_img(ep.features)
        _, acc = prototypical_loss_from_prototypes(z, prototypes, ep.labels, distance_fn)
        accs.append(acc.item())

    mean_acc = float(np.mean(accs))
    if len(accs) < 2:
        ci95 = 0.0
    else:
        ci95 = _Z_SCORE_95 * float(np.std(accs, ddof=1)) / np.sqrt(len(accs))
    return mean_acc, ci95


def build_models(
    aux_dim: int,
    z_dim: int,
    device: torch.device,
    distance: DistanceType,
    lowrank_r: int,
) -> Tuple[nn.Module, nn.Module, DistanceFn]:
    """Construct encoders and configured distance function on device."""
    model_img = LinearImageEncoder(in_dim=IMAGE_DIM, z_dim=z_dim).to(device)
    model_aux = LinearAuxEncoder(aux_dim=aux_dim, z_dim=z_dim).to(device)
    distance_fn = build_distance({"distance": distance.value, "lowrank_r": lowrank_r}, z_dim, device)
    return model_img, model_aux, distance_fn


def build_optimizer(
    model_img: nn.Module,
    model_aux: nn.Module,
    distance_fn: DistanceFn,
    lr: float,
    weight_decay: float,
) -> torch.optim.Adam:
    """Create Adam optimizer over model and optional distance parameters."""
    dist_params = list(distance_fn.parameters()) if isinstance(distance_fn, nn.Module) else []
    return torch.optim.Adam(
        list(model_img.parameters()) + list(model_aux.parameters()) + dist_params,
        lr=lr,
        weight_decay=weight_decay,
    )


def main() -> None:
    """Parse configuration, train the model, evaluate, and save checkpoint."""
    parser = argparse.ArgumentParser(
        description="Zero-shot CUB with precomputed features (Table 3)"
    )
    parser.add_argument("--config", type=str, default="",
                        help="Path to cub_config.yaml (optional)")
    parser.add_argument("--data_root", type=str, default="data/cvpr2016_cub")
    parser.add_argument("--cub_root", type=str, default="data/CUB_200_2011")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--z_dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_way", type=int, default=50,
                        help="Episode n-way (paper: 50)")
    parser.add_argument("--n_query", type=int, default=10,
                        help="Queries per class per episode (paper: 10)")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--val_episodes", type=int, default=20)
    parser.add_argument("--test_episodes", type=int, default=1000)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--exp_name", type=str, default="cub_zeroshot_precomputed")
    parser.add_argument("--distance", type=str, default="euclidean",
                        choices=[d.value for d in DistanceType])
    parser.add_argument("--lowrank_r", type=int, default=64)
    parser.add_argument("--phase2_epochs_scale", type=float, default=1.0,
                        help="Multiply best_epoch by this for Phase 2 (0 to skip)")
    args = parser.parse_args()

    yaml_path = args.config or os.path.join(PROJECT_ROOT, "configs", "cub_config.yaml")
    config = TrainConfig.from_args_and_yaml(args, yaml_path)

    set_seed(config.seed)
    device = get_device()

    train_ds = CUBPrecomputedDataset(
        str(config.data_root), split="train",
        cub_root=str(config.cub_root), test_time=False,
    )
    val_ds = CUBPrecomputedDataset(
        str(config.data_root), split="val",
        cub_root=str(config.cub_root), test_time=True,
    )
    test_ds = CUBPrecomputedDataset(
        str(config.data_root), split="test",
        cub_root=str(config.cub_root), test_time=True,
    )

    aux_dim = train_ds.aux_features.shape[1]

    print(f"Loading CUB precomputed features from {config.data_root}")
    print(f"Auxiliary modality: attributes ({aux_dim}-dim)")
    print(f"Distance metric: {config.distance.value}")
    print(f"Train: {len(train_ds.class_names)} classes, {len(train_ds)} images")
    print(f"Val:   {len(val_ds.class_names)} classes, {len(val_ds)} images")
    print(f"Test:  {len(test_ds.class_names)} classes, {len(test_ds)} images")

    # Phase 1: train on train classes, early-stop on val
    model_img, model_aux, distance_fn = build_models(
        aux_dim, config.z_dim, device, config.distance, config.lowrank_r,
    )
    optimizer = build_optimizer(model_img, model_aux, distance_fn, config.lr, config.weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_checkpoint = _snapshot_checkpoint(model_img, model_aux, distance_fn)

    for epoch in range(config.epochs):
        train_loss, train_acc = run_episodes(
            model_img, model_aux, train_ds.aux_features,
            optimizer, train_ds, device, distance_fn,
            n_episodes=config.n_episodes, n_way=config.n_way,
            n_query=config.n_query, do_backward=True,
        )
        val_loss, val_acc = run_episodes(
            model_img, model_aux, val_ds.aux_features,
            optimizer, val_ds, device, distance_fn,
            n_episodes=config.val_episodes, n_way=min(50, len(val_ds.class_names)),
            n_query=config.n_query, do_backward=False,
        )

        print(
            f"Epoch {epoch+1}/{config.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss - config.early_stopping_min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_checkpoint = _snapshot_checkpoint(model_img, model_aux, distance_fn)
        else:
            epochs_no_improve += 1

        if config.early_stopping_patience and epochs_no_improve >= config.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch+1} "
                f"(best epoch={best_epoch}, val_loss={best_val_loss:.4f})"
            )
            break

    if best_epoch <= 0:
        best_epoch = 1

    _restore_checkpoint(best_checkpoint, model_img, model_aux, distance_fn)

    p1_mean, p1_ci = evaluate_episodic(
        model_img, model_aux, test_ds.aux_features, test_ds, device, distance_fn,
        n_episodes=config.test_episodes, n_way=len(test_ds.class_names),
        n_query=config.n_query,
    )
    print(
        f"\nPhase-1 best checkpoint test accuracy ({len(test_ds.class_names)}-way): "
        f"{p1_mean:.4f} \u00b1 {p1_ci:.4f} (95% CI over {config.test_episodes} episodes)"
    )

    # Phase 2: retrain on train+val
    phase2_epochs = max(1, round(best_epoch * config.phase2_epochs_scale))
    skip_phase2 = config.phase2_epochs_scale == 0

    if skip_phase2:
        print("\nPhase 2 skipped (--phase2_epochs_scale 0)")
        test_mean, test_ci = p1_mean, p1_ci
    else:
        print(
            f"\nPhase 2: retraining on train+val for {phase2_epochs} epoch(s) "
            f"(best_epoch={best_epoch} x scale={config.phase2_epochs_scale:.2f})..."
        )
        trainval_ds = CUBPrecomputedDataset(
            str(config.data_root), split="trainval",
            cub_root=str(config.cub_root), test_time=False,
        )
        print(f"Train+val: {len(trainval_ds.class_names)} classes, {len(trainval_ds)} images")

        set_seed(config.seed)
        model_img, model_aux, distance_fn = build_models(
            aux_dim, config.z_dim, device, config.distance, config.lowrank_r,
        )
        optimizer = build_optimizer(model_img, model_aux, distance_fn, config.lr, config.weight_decay)

        for epoch in range(phase2_epochs):
            train_loss, train_acc = run_episodes(
                model_img, model_aux, trainval_ds.aux_features,
                optimizer, trainval_ds, device, distance_fn,
                n_episodes=config.n_episodes, n_way=config.n_way,
                n_query=config.n_query, do_backward=True,
            )
            print(f"[Retrain {epoch+1}/{phase2_epochs}] loss={train_loss:.4f}  acc={train_acc:.4f}")

        test_mean, test_ci = evaluate_episodic(
            model_img, model_aux, test_ds.aux_features, test_ds, device, distance_fn,
            n_episodes=config.test_episodes, n_way=len(test_ds.class_names),
            n_query=config.n_query,
        )

    print(
        f"\nFinal zero-shot test accuracy ({len(test_ds.class_names)}-way): "
        f"{test_mean:.4f} \u00b1 {test_ci:.4f} (95% CI over {config.test_episodes} episodes)"
    )

    os.makedirs(config.save_dir, exist_ok=True)
    ckpt = {
        "best_epoch": best_epoch,
        "phase2_epochs": 0 if skip_phase2 else phase2_epochs,
        "phase2_epochs_scale": config.phase2_epochs_scale,
        "aux_type": "attributes",
        "distance": config.distance.value,
        "z_dim": config.z_dim,
        "img_encoder": model_img.state_dict(),
        "aux_encoder": model_aux.state_dict(),
        "test_acc": test_mean,
        "test_ci95": test_ci,
        "phase1_test_acc": p1_mean,
        "phase1_test_ci95": p1_ci,
    }
    if isinstance(distance_fn, nn.Module):
        ckpt["distance_fn"] = distance_fn.state_dict()

    path = os.path.join(
        str(config.save_dir),
        f"{config.exp_name}_attributes_{config.distance.value}_best.pt",
    )
    torch.save(ckpt, path)
    print(f"Checkpoint saved to {path}")


if __name__ == "__main__":
    main()
