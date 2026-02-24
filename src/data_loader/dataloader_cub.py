"""
CUB-200-2011 dataloader for zero-shot learning (Table 3 replication).

Loads images and class-level attributes. Supports the 100/50/50 class split
(train/val/test) as in Snell et al. (2017), following Reed et al. (2016):
100 training classes, 50 validation classes, 50 test classes.
Dataset: https://data.caltech.edu/records/65de6-vp158
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# CUB has 200 classes. For zero-shot per Snell et al.:
# 100 train, 50 val, 50 test classes.
N_TOTAL_CLASSES = 200
N_TRAIN_CLASSES = 100
N_VAL_CLASSES = 50
N_TEST_CLASSES = 50
N_ATTRIBUTES = 312


def _read_lines(path: str) -> list[list[str]]:
    with open(path) as f:
        return [line.strip().split() for line in f if line.strip()]


def load_class_attributes(root: str) -> np.ndarray:
    """
    Load class-level attributes (200 x 312).
    Expects attributes in root/attributes/class_attribute_labels_continuous.txt
    (one row per class, 312 space-separated values). If not found, tries
    root/class_attribute_labels_continuous.txt.
    """
    for subpath in [
        "attributes/class_attribute_labels_continuous.txt",
        "class_attribute_labels_continuous.txt",
    ]:
        p = Path(root) / subpath
        if p.exists():
            rows = _read_lines(str(p))
            arr = np.array([[float(x) for x in row] for row in rows], dtype=np.float32)
            assert arr.shape[1] == N_ATTRIBUTES, f"expected {N_ATTRIBUTES} attributes"
            return arr
    raise FileNotFoundError(
        f"No class attributes file found under {root}. "
    )


def build_cub_index(root: str):
    """
    Parse CUB_200_2011 images.txt and image_class_labels.txt.
    Returns list of (image_path, class_id_1based), and number of classes.
    """
    root = Path(root)
    images_file = root / "images.txt"
    labels_file = root / "image_class_labels.txt"
    if not images_file.exists() or not labels_file.exists():
        raise FileNotFoundError(
            f"CUB metadata not found under {root}. "
            "Ensure images.txt and image_class_labels.txt exist (extract CUB_200_2011.tgz)."
        )

    # image_id -> path (e.g. "001.Black_footed_Albatross/Black_Footed_Albatross_0001.jpg")
    id_to_path = {}
    for parts in _read_lines(str(images_file)):
        img_id, path = int(parts[0]), parts[1]
        id_to_path[img_id] = path

    # image_id -> class_id (1..200)
    id_to_cls = {}
    for parts in _read_lines(str(labels_file)):
        img_id, cls = int(parts[0]), int(parts[1])
        id_to_cls[img_id] = cls

    img_dir = root / "images"
    samples = []
    for img_id, path in id_to_path.items():
        if img_id not in id_to_cls:
            continue
        full_path = img_dir / path
        if full_path.exists():
            samples.append((str(full_path), id_to_cls[img_id]))
    return samples


class CUBDataset(Dataset):
    """
    CUB-200-2011 for zero-shot: images + labels. Class attributes loaded separately
    via load_class_attributes() for building prototypes.

    Splits are by class:
        - train: classes 1..100
        - val:   classes 101..150
        - test:  classes 151..200
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        """
        Args:
            root: path to CUB_200_2011 (containing images/, images.txt, etc.)
            split: "train", "val", or "test" (class-based split)
            transform: optional transform applied to PIL image
        """
        self.root = Path(root)
        self.transform = transform

        samples = build_cub_index(str(self.root))
        # class_id in CUB is 1..200
        if split == "train":
            cls_set = set(range(1, N_TRAIN_CLASSES + 1))
        elif split == "val":
            cls_set = set(range(N_TRAIN_CLASSES + 1, N_TRAIN_CLASSES + N_VAL_CLASSES + 1))
        elif split == "test":
            cls_set = set(
                range(
                    N_TRAIN_CLASSES + N_VAL_CLASSES + 1,
                    N_TRAIN_CLASSES + N_VAL_CLASSES + N_TEST_CLASSES + 1,
                )
            )
        else:
            raise ValueError(f"Unknown split: {split}")

        self.classes = sorted(cls_set)
        self.samples = [(p, c) for p, c in samples if c in cls_set]

        # Map class id (1..200) to index in this split's classes (0..len-1)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[class_id]
        return img, label


def get_cub_dataloaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
):
    """
    Build train/val/test DataLoaders for CUB zero-shot following the paper:
    100 train classes, 50 val classes, 50 test classes.
    Also returns class attributes array (200, 312).
    """
    from torchvision import transforms

    # Normalization used with ImageNet-pretrained GoogLeNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Training: follow the paper by using multiple crops from the original and
    # horizontally-flipped image. torchvision.transforms.TenCrop implements
    # (top-left, top-right, bottom-left, bottom-right, center) + their mirrors.
    transform_train = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.TenCrop(image_size),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [normalize(transforms.ToTensor()(c)) for c in crops]
                )
            ),
        ]
    )

    # Validation / test: use only the center crop of the original image.
    transform_eval = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_ds = CUBDataset(root, split="train", transform=transform_train)
    val_ds = CUBDataset(root, split="val", transform=transform_eval)
    test_ds = CUBDataset(root, split="test", transform=transform_eval)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Class attributes: (200, 312). Rows 0..99 = train, 100..149 = val, 150..199 = test (0-indexed)
    attributes = load_class_attributes(root)
    assert attributes.shape[0] == 200 and attributes.shape[1] == N_ATTRIBUTES

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "attributes": attributes,
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "train_dataset": train_ds,
        "test_dataset": test_ds,
    }
