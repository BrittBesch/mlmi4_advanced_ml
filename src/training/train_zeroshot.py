"""
Zero-shot training and evaluation for CUB (Table 3 replication).

Train on seen classes: image encoder + attribute→embedding map; prototypes for
seen classes = attribute_embed(attributes). At test time, prototypes for unseen
classes = attribute_embed(unseen_attributes); classify by nearest prototype.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models

# Add project root so we can import from mlmi4_advanced_ml
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from loss import prototypical_loss_from_prototypes, euclidean_dist
from src.data_loader.dataloader_cub import get_cub_dataloaders
from src.utils.device import get_device
from src.utils.seed import set_seed


class CUBImageEncoder(nn.Module):
    """
    Image embedding f_φ for CUB.

    In the paper, 1,024-dim GoogLeNet features are used and a linear map is
    learned on top. Here we follow that setup by using a pretrained GoogLeNet
    backbone frozen on ImageNet and learning only a linear head.
    """

    def __init__(self, z_dim: int = 1024):
        super().__init__()
        # Pretrained GoogLeNet on ImageNet. We remove the final classifier so that
        # the network outputs 1,024-dim pooled features.
        self.backbone = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1, aux_logits=False)
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


def train_epoch(model_enc, model_attr, attributes_split, optimizer, loader, device):
    model_enc.train()
    model_attr.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    # Prototypes for current split: normalize to unit length as in the paper
    with torch.no_grad():
        att = torch.tensor(attributes_split, dtype=torch.float32, device=device)
        prototypes = _l2_normalize(model_attr(att))
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        z = model_enc(images)
        loss, acc = prototypical_loss_from_prototypes(z, prototypes, labels, distance_fn=euclidean_dist)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc.item()
        n_batches += 1
    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


@torch.no_grad()
def evaluate(model_enc, model_attr, attributes_unseen, loader, device):
    model_enc.eval()
    model_attr.eval()
    att = torch.tensor(attributes_unseen, dtype=torch.float32, device=device)
    prototypes = _l2_normalize(model_attr(att))
    total, correct = 0, 0
    for images, labels in loader:
        images = images.to(device)
        z = model_enc(images)
        dists = euclidean_dist(z, prototypes)
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
    args = parser.parse_args()

    cfg = load_config(args.config or os.path.join(PROJECT_ROOT, "configs", "cub_config.yaml"))
    if cfg:
        data_cfg = cfg.get("data", {})
        train_cfg = cfg.get("training", {})
        exp_cfg = cfg.get("experiment", {})
        args.data_root = data_cfg.get("root", args.data_root)
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

    data = get_cub_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    attributes = data["attributes"]

    # 100 train, 50 val, 50 test classes (rows 0..99, 100..149, 150..199)
    attrs_train = attributes[:100]
    attrs_val = attributes[100:150]
    attrs_test = attributes[150:]

    model_enc = CUBImageEncoder(z_dim=args.z_dim).to(device)
    model_attr = AttributeEmbedding(attr_dim=312, z_dim=args.z_dim).to(device)
    optimizer = torch.optim.Adam(
        list(model_enc.parameters()) + list(model_attr.parameters()),
        lr=args.lr,
        weight_decay=1e-5,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0

    # Phase 1: train on train classes, early-stop on val loss (as in paper)
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model_enc, model_attr, attrs_train, optimizer, train_loader, device
        )

        # Validation loss on val classes
        model_enc.eval()
        model_attr.eval()
        with torch.no_grad():
            att_val = torch.tensor(attrs_val, dtype=torch.float32, device=device)
            prototypes_val = _l2_normalize(model_attr(att_val))
            val_loss = 0.0
            n_batches = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                z = model_enc(images)
                loss, _ = prototypical_loss_from_prototypes(
                    z, prototypes_val, labels, distance_fn=euclidean_dist
                )
                val_loss += loss.item()
                n_batches += 1
            val_loss /= max(n_batches, 1)

        print(
            f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"train_acc={train_acc:.4f}  val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

    # Phase 2: retrain on train+val for best_epoch epochs, then evaluate on test
    from torch.utils.data import ConcatDataset, DataLoader

    combined_ds = ConcatDataset([data["train_dataset"], data["val_dataset"]])
    combined_loader = DataLoader(
        combined_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    attrs_train_val = attributes[:150]  # 100 train + 50 val

    # Reinitialize models and optimizer
    model_enc = CUBImageEncoder(z_dim=args.z_dim).to(device)
    model_attr = AttributeEmbedding(attr_dim=312, z_dim=args.z_dim).to(device)
    optimizer = torch.optim.Adam(
        list(model_enc.parameters()) + list(model_attr.parameters()),
        lr=args.lr,
        weight_decay=1e-5,
    )

    for epoch in range(best_epoch):
        train_loss, train_acc = train_epoch(
            model_enc, model_attr, attrs_train_val, optimizer, combined_loader, device
        )
        print(
            f"[Retrain {epoch+1}/{best_epoch}] loss={train_loss:.4f} acc={train_acc:.4f}"
        )

    test_acc = evaluate(model_enc, model_attr, attrs_test, test_loader, device)
    ckpt = {
        "best_epoch": best_epoch,
        "encoder": model_enc.state_dict(),
        "attr_embed": model_attr.state_dict(),
        "test_acc": test_acc,
    }
    path = os.path.join(args.save_dir, f"{args.exp_name}_best.pt")
    torch.save(ckpt, path)
    print(f"Final zero-shot test accuracy (50-way): {test_acc:.4f}")
    print(f"  -> saved best to {path}")


if __name__ == "__main__":
    main()
