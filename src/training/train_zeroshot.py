"""
Zero-shot training and evaluation for CUB (Table 3 replication).

Train on seen classes: image encoder + attribute→embedding map; prototypes for
seen classes = attribute_embed(attributes). At test time, prototypes for unseen
classes = attribute_embed(unseen_attributes); classify by nearest prototype.
"""

import argparse
import os
import sys
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models, transforms

# Add project root so we can import from mlmi4_advanced_ml
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from loss import (
    prototypical_loss_from_prototypes,
    euclidean_dist,
    build_distance,
)
from src.data_loader.dataloader_cub import get_dataloader, load_class_attributes
from src.utils.device import get_device
from src.utils.seed import set_seed


# NOTE (Britt / models.py):
# The following two classes define the *model* components used in the CUB
# zero-shot experiment and could be moved into `model.py` to be shared by
# multiple training scripts. They are kept here for now for clarity.

class CUBImageEncoder(nn.Module):
    """
    Image embedding f_φ for CUB.

    In the paper, 1,024-dim GoogLeNet features are used and a linear map is
    learned on top. Here we follow that setup by using a pretrained GoogLeNet
    backbone frozen on ImageNet and learning only a linear head.
    """

    def __init__(self, z_dim: int = 1024):
        super().__init__()
        # Pretrained GoogLeNet on ImageNet. We keep the default aux_logits=True
        # required by the pretrained weights, and later ignore aux outputs.
        # The final classifier is removed so the network outputs 1,024-dim features.
        self.backbone = models.googlenet(
            weights=models.GoogLeNet_Weights.IMAGENET1K_V1
        )
        self.backbone.fc = nn.Identity()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.z_dim = z_dim
        self.fc = nn.Linear(1024, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x can be either:
        # - (B, 3, H, W): single crop per image (val/test)
        # - (B, Ncrops, 3, H, W): multiple crops per image (train with TenCrop)
        if x.dim() == 5:
            b, n_crops, c, h, w = x.shape
            x = x.view(b * n_crops, c, h, w)
            with torch.no_grad():
                feats = self.backbone(x)
            feats = feats.view(b, n_crops, -1).mean(1)
        else:
            with torch.no_grad():
                feats = self.backbone(x)
        return self.fc(feats)


class AttributeEmbedding(nn.Module):
    """
    Map class attributes (312-dim) to embedding space (z_dim).

    In the paper this is a simple linear map 312 → 1024, with unit-length
    prototypes after embedding.
    """

    def __init__(self, attr_dim: int = 312, z_dim: int = 1024):
        super().__init__()
        self.fc = nn.Linear(attr_dim, z_dim)
        self.z_dim = z_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ------ Training and evaluation ------
def _l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def train_epoch(model_enc, model_attr, attributes_split, optimizer, loader, device,
                distance_fn=None):
    """
    Run one training epoch.

    distance_fn controls which distance metric is used to compare image
    embeddings to class prototypes:
      - None / euclidean_dist  →  Experiment 1: paper baseline
      - DiagonalMahalanobisDistance instance  →  Experiment 2: EXTENSION
      - LowRankMahalanobisDistance  instance  →  Experiment 3: EXTENSION

    Note: prototypes are recomputed inside every batch loop so that
    model_attr receives gradients and actually learns.  (Computing them
    once outside the loop with torch.no_grad() would silently block all
    gradients from reaching model_attr.)
    """
    if distance_fn is None:
        distance_fn = euclidean_dist

    model_enc.train()
    model_attr.train()
    # If distance_fn is a learnable module, set it to train mode too.
    if isinstance(distance_fn, nn.Module):
        distance_fn.train()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    # Convert fixed class attributes to a tensor once (they don't change).
    att = torch.tensor(attributes_split, dtype=torch.float32, device=device)

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Recompute prototypes each step so gradients flow back through
        # model_attr.  L2-normalise as in the paper (unit-length prototypes).
        prototypes = _l2_normalize(model_attr(att))   # (n_classes, z_dim)

        z = model_enc(images)                          # (B, z_dim)
        loss, acc = prototypical_loss_from_prototypes(
            z, prototypes, labels, distance_fn=distance_fn
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc.item()
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def train_epoch_episodic(
    model_enc,
    model_attr,
    attributes_split,
    optimizer,
    dataset,
    device,
    distance_fn=None,
    n_episodes: int = 50,
    n_way: int = 50,
    n_query: int = 10,
    do_backward: bool = True,
):
    """
    Phase 1 training using episodic 50-way, 10-query episodes as in Snell et al.
    """
    if distance_fn is None:
        distance_fn = euclidean_dist

    model_enc.train()
    model_attr.train()
    if isinstance(distance_fn, nn.Module):
        distance_fn.train()

    # Fixed class attributes for this split.
    att_full = torch.tensor(attributes_split, dtype=torch.float32, device=device)

    # Build index lists per class label once per epoch.
    indices_per_class: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        _, lbl = dataset[idx]
        indices_per_class[int(lbl)] += [idx]

    all_classes = [c for c in indices_per_class.keys() if len(indices_per_class[c]) > 0]

    total_loss = 0.0
    total_acc = 0.0

    for _ in tqdm(range(n_episodes), desc="Train (episodic)", leave=False):
        if len(all_classes) < n_way:
            episode_classes = all_classes
            this_way = len(all_classes)
        else:
            episode_classes = random.sample(all_classes, n_way)
            this_way = n_way

        episode_indices = []
        episode_labels = []
        for local_idx, c in enumerate(episode_classes):
            idxs = indices_per_class[c]
            if len(idxs) >= n_query:
                chosen = random.sample(idxs, n_query)
            else:
                chosen = [random.choice(idxs) for _ in range(n_query)]
            episode_indices.extend(chosen)
            episode_labels.extend([local_idx] * n_query)

        images = []
        for idx in episode_indices:
            img, _ = dataset[idx]
            images.append(img)
        images = torch.stack(images, dim=0).to(device)
        labels = torch.tensor(episode_labels, dtype=torch.long, device=device)

        cls_idx_tensor = torch.tensor(episode_classes, dtype=torch.long, device=device)
        prototypes = _l2_normalize(model_attr(att_full[cls_idx_tensor[:this_way]]))

        z = model_enc(images)
        loss, acc = prototypical_loss_from_prototypes(
            z, prototypes, labels, distance_fn=distance_fn
        )
        if do_backward:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()

    return total_loss / max(n_episodes, 1), total_acc / max(n_episodes, 1)


@torch.no_grad()
def evaluate(model_enc, model_attr, attributes_unseen, loader, device,
             distance_fn=None):
    """
    Evaluate zero-shot accuracy on unseen classes.

    Pass the *same* distance_fn used during training so the learned metric
    is applied at inference time.  Defaults to euclidean_dist.
    """
    if distance_fn is None:
        distance_fn = euclidean_dist

    model_enc.eval()
    model_attr.eval()
    if isinstance(distance_fn, nn.Module):
        distance_fn.eval()

    att = torch.tensor(attributes_unseen, dtype=torch.float32, device=device)
    prototypes = _l2_normalize(model_attr(att))   # (n_unseen, z_dim)

    total, correct = 0, 0
    for images, labels in loader:
        images = images.to(device)
        z = model_enc(images)                      # (B, z_dim)
        dists = distance_fn(z, prototypes)         # (B, n_unseen)
        pred = dists.argmin(1)
        correct += (pred.cpu() == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def load_config(config_path: str) -> dict:
    if not os.path.isfile(config_path):
        return {}
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Zero-shot CUB (Table 3)")
    parser.add_argument("--config", type=str, default="", help="Path to cub_config.yaml (optional)")
    parser.add_argument("--data_root", type=str, default="data/CUB_200_2011", help="CUB dataset root")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--z_dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--exp_name", type=str, default="cub_zeroshot")

    # EXTENSION: distance metric selection
    # Experiment 1: --distance euclidean  (paper baseline, no extra parameters)
    # Experiment 2: --distance diagonal   (diagonal Mahalanobis, d extra params)
    # Experiment 3: --distance lowrank    (low-rank Mahalanobis, d×r extra params)
    parser.add_argument(
        "--distance", type=str, default="euclidean",
        choices=["euclidean", "diagonal", "lowrank"],
        help="Distance metric to use.  'euclidean' reproduces the paper; "
             "'diagonal' and 'lowrank' are EXTENSION experiments.",
    )
    parser.add_argument(
        "--lowrank_r", type=int, default=64,
        help="EXTENSION: rank r for low-rank Mahalanobis (only used when "
             "--distance=lowrank).  r=64 gives 64×1024=65 536 extra params.",
    )

    args = parser.parse_args()

    cfg = load_config(args.config or os.path.join(PROJECT_ROOT, "configs", "cub_config.yaml"))
    if cfg:
        data_cfg = cfg.get("data", {})
        train_cfg = cfg.get("training", {})
        exp_cfg = cfg.get("experiment", {})
        args.data_root = data_cfg.get("data_dir", args.data_root)
        args.image_size = data_cfg.get("image_size", args.image_size)
        args.batch_size = data_cfg.get("batch_size", args.batch_size)
        args.epochs = train_cfg.get("epochs", args.epochs)
        args.lr = train_cfg.get("lr", args.lr)
        args.seed = exp_cfg.get("seed", args.seed)
        args.save_dir = exp_cfg.get("save_dir", args.save_dir)
        args.exp_name = exp_cfg.get("exp_name", args.exp_name)
        if "z_dim" in cfg.get("model", {}):
            args.z_dim = cfg["model"]["z_dim"]

    set_seed(args.seed)
    device = get_device()

    data_cfg = {
        "data_dir": args.data_root,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": 0,
    }
    # For CUB zero-shot, use episodic 50-way / 10-query training batches to
    # more closely follow the setup described in Snell et al. (2017).
    train_loader = get_dataloader(data_cfg, "train", episodic=True, n_way=50, n_query=10)
    val_loader = get_dataloader(data_cfg, "val")
    test_loader = get_dataloader(data_cfg, "test")
    attributes = load_class_attributes(args.data_root)

    # 100 train, 50 val, 50 test classes (rows 0..99, 100..149, 150..199)
    attrs_train = attributes[:100]
    attrs_val = attributes[100:150]
    attrs_test = attributes[150:]

    model_enc = CUBImageEncoder(z_dim=args.z_dim).to(device)
    model_attr = AttributeEmbedding(attr_dim=312, z_dim=args.z_dim).to(device)

    # ---- EXTENSION: distance metric ------------------------------------------
    # Experiment 1 (baseline) : --distance euclidean  → no extra module/params
    # Experiment 2 (EXTENSION): --distance diagonal   → DiagonalMahalanobisDistance
    # Experiment 3 (EXTENSION): --distance lowrank    → LowRankMahalanobisDistance
    dist_cfg = {"distance": args.distance, "lowrank_r": args.lowrank_r}
    distance_fn = build_distance(dist_cfg, args.z_dim, device)

    # Collect learnable parameters from the distance module (empty list for Euclidean).
    dist_params = list(distance_fn.parameters()) if isinstance(distance_fn, nn.Module) else []
    print(f"Distance metric : {args.distance}"
          + (f"  (rank={args.lowrank_r})" if args.distance == "lowrank" else ""))
    print(f"Distance params : {sum(p.numel() for p in dist_params)}")
    # --------------------------------------------------------------------------

    optimizer = torch.optim.Adam(
        list(model_enc.parameters()) + list(model_attr.parameters()) + dist_params,
        lr=args.lr,
        weight_decay=1e-5,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0

    # Phase 1: train on train classes with episodic 50-way, 10-query episodes,
    # then early-stop on val loss (as in paper).
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch_episodic(
            model_enc,
            model_attr,
            attrs_train,
            optimizer,
            train_loader.dataset,
            device,
            distance_fn=distance_fn,
            n_episodes=50,
            n_way=50,
            n_query=10,
            do_backward=True,
        )

        # Validation: approximate paper by computing loss on episodic val episodes.
        model_enc.eval()
        model_attr.eval()
        if isinstance(distance_fn, nn.Module):
            distance_fn.eval()
        with torch.no_grad():
            # Reuse episodic routine but on val split (no optimiser step).
            val_loss, _ = train_epoch_episodic(
                model_enc,
                model_attr,
                attrs_val,
                optimizer,            # not used for updates (no backward in eval)
                val_loader.dataset,
                device,
                distance_fn=distance_fn,
                n_episodes=20,
                n_way=50,
                n_query=10,
                do_backward=False,
            )

        print(
            f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"train_acc={train_acc:.4f}  val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

    # Phase 2: retrain on train+val for best_epoch epochs, then evaluate on test
    from torch.utils.data import ConcatDataset, DataLoader
    from src.data_loader.dataloader_cub import CUBDataset

    # For Phase 2 we want the same multi-crop behaviour as in Phase 1 training,
    # so we mirror the TenCrop-based transform used in get_dataloader(..., "train").
    phase2_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.TenCrop(args.image_size),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        )(transforms.ToTensor()(c))
                        for c in crops
                    ]
                )
            ),
        ]
    )
    train_ds_phase2 = CUBDataset(args.data_root, split="train", transform=phase2_transform)
    val_ds_phase2 = CUBDataset(args.data_root, split="val", transform=phase2_transform)
    combined_ds = ConcatDataset([train_ds_phase2, val_ds_phase2])
    combined_loader = DataLoader(
        combined_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    attrs_train_val = attributes[:150]  # 100 train + 50 val

    # Reinitialise models, distance module, and optimizer from scratch.
    # Phase 2 must start from the same initial conditions as Phase 1 so
    # that training for best_epoch steps is a faithful replay.
    set_seed(args.seed)
    model_enc = CUBImageEncoder(z_dim=args.z_dim).to(device)
    model_attr = AttributeEmbedding(attr_dim=312, z_dim=args.z_dim).to(device)
    distance_fn = build_distance(dist_cfg, args.z_dim, device)   # fresh distance module
    dist_params = list(distance_fn.parameters()) if isinstance(distance_fn, nn.Module) else []
    optimizer = torch.optim.Adam(
        list(model_enc.parameters()) + list(model_attr.parameters()) + dist_params,
        lr=args.lr,
        weight_decay=1e-5,
    )

    for epoch in range(best_epoch):
        train_loss, train_acc = train_epoch(
            model_enc, model_attr, attrs_train_val, optimizer, combined_loader, device,
            distance_fn=distance_fn,   # same metric as Phase 1
        )
        print(
            f"[Retrain {epoch+1}/{best_epoch}] loss={train_loss:.4f} acc={train_acc:.4f}"
        )

    test_acc = evaluate(
        model_enc, model_attr, attrs_test, test_loader, device,
        distance_fn=distance_fn,   # use learned metric at inference time
    )

    # Save checkpoint; include distance module state for EXTENSION experiments.
    ckpt = {
        "best_epoch": best_epoch,
        "distance": args.distance,
        "encoder": model_enc.state_dict(),
        "attr_embed": model_attr.state_dict(),
        "test_acc": test_acc,
    }
    if isinstance(distance_fn, nn.Module):
        # EXTENSION: persist learned metric parameters alongside the model.
        ckpt["distance_fn"] = distance_fn.state_dict()

    path = os.path.join(args.save_dir, f"{args.exp_name}_{args.distance}_best.pt")
    torch.save(ckpt, path)
    print(f"Final zero-shot test accuracy (50-way): {test_acc:.4f}")
    print(f"  -> saved checkpoint to {path}")


if __name__ == "__main__":
    main()
