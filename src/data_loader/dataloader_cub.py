"""
CUB dataloader using precomputed features from cvpr2016_cub (Reed et al., 2016).

Implements the data loading for Snell et al. (2017) Table 3 zero-shot CUB:
  - Image features: precomputed 1024-dim GoogLeNet features (.t7 files)
      10 crops per image: middle, upper-left, upper-right, lower-left, lower-right
      of the original image and its horizontal flip (crop index 0 = middle original).
      Train: one random crop per __getitem__ (uniform over the 10); stored arrays are (n_imgs, 1024, 10).
      Val/test (test_time=True): middle crop only — paper: "middle crop of original".
  - Auxiliary class features: 312-dim continuous CUB attributes (paper) loaded
      from CUB_200_2011/attributes/class_attribute_labels_continuous.txt.
  - Class splits from the provided split text files (100/50/50 train/val/test)

Data directory: data/cvpr2016_cub/
  images/   *.t7  shape (n_imgs, 1024, 10) - precomputed GoogLeNet features
  trainclasses.txt / valclasses.txt / testclasses.txt / trainvalclasses.txt

Attributes (paper): data/CUB_200_2011/attributes/class_attribute_labels_continuous.txt
"""

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import torchfile
except ImportError as e:
    raise ImportError("torchfile is required to read .t7 files: pip install torchfile") from e

# Crop index 0 is the middle crop of the original image (paper test-time crop).
MIDDLE_CROP_IDX = 0

IMAGE_DIM = 1024


def _load_split_names(split_file: Path) -> List[str]:
    """Return list of class folder names (e.g. '001.Black_footed_Albatross') from a split file."""
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


def _class_number(cls_name: str) -> int:
    """Extract 1-based class number from folder name e.g. '001.Black_footed_Albatross' → 1."""
    return int(cls_name.split(".")[0])


def _load_class_image_features(class_file: Path) -> np.ndarray:
    """
    Load precomputed GoogLeNet features for one class.

    Returns full (n_imgs, 1024, 10) array. Crop selection happens at
    __getitem__ time: random crop per access during training (matching
    Reed et al.'s torch.randperm sampling), fixed MIDDLE_CROP_IDX at test time.
    """
    return np.array(torchfile.load(str(class_file)), dtype=np.float32)  # (n_imgs, 1024, 10)


def load_cub_attributes(cub_root: str, class_names: List[str]) -> np.ndarray:
    """
    Load 312-dim continuous CUB attributes for the given classes (paper aux modality).

    Reads class_attribute_labels_continuous.txt (200 rows, one per class in order 1..200)
    and returns rows corresponding to the requested class_names.

    Args:
        cub_root: path to CUB_200_2011 directory
        class_names: list of class folder names e.g. ['001.Black_footed_Albatross', ...]

    Returns:
        (n_classes, 312) float32 attribute array
    """
    cub_root = Path(cub_root)
    for subpath in [
        "attributes/class_attribute_labels_continuous.txt",
        "class_attribute_labels_continuous.txt",
    ]:
        attr_file = cub_root / subpath
        if attr_file.exists():
            rows = []
            with open(attr_file) as f:
                all_rows = [
                    [float(v) for v in line.strip().split()]
                    for line in f if line.strip()
                ]
            for cls_name in class_names:
                cls_idx = _class_number(cls_name) - 1   # 0-based row index
                rows.append(all_rows[cls_idx])
            return np.array(rows, dtype=np.float32)   # (n_classes, 312)
    raise FileNotFoundError(
        f"CUB attributes file not found under {cub_root}. "
        "Download CUB_200_2011.tgz from https://data.caltech.edu/records/65de6-vp158 "
        "and extract to data/CUB_200_2011/."
    )


def load_split_data(
    data_root: str,
    split: str,
    cub_root: Optional[str] = None,
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """
    Load all image features and class-level auxiliary features for a split.

    Args:
        data_root: path to cvpr2016_cub directory
        split: one of "train", "val", "test", "trainval"
        cub_root: path to CUB_200_2011 directory

    Returns:
        image_features_by_class: list of (n_imgs, 1024, 10) arrays, one per class
        aux_features: (n_classes, aux_dim) class-level auxiliary embeddings
        class_names: list of class folder names in split order
    """
    root = Path(data_root)
    split_file_map = {
        "train": root / "trainclasses.txt",
        "val": root / "valclasses.txt",
        "test": root / "testclasses.txt",
        "trainval": root / "trainvalclasses.txt",
    }
    if split not in split_file_map:
        raise ValueError(f"Unknown split '{split}'. Choose from: {list(split_file_map)}")

    class_names = _load_split_names(split_file_map[split])

    # Load image features
    image_features_by_class = []
    for cls_name in class_names:
        img_file = root / "images" / f"{cls_name}.t7"
        if not img_file.exists():
            raise FileNotFoundError(f"Image feature file not found: {img_file}")
        image_features_by_class.append(_load_class_image_features(img_file))

    # Load auxiliary features (attributes only)
    if cub_root is None:
        raise ValueError(
            "cub_root must be provided. Set --cub_root data/CUB_200_2011"
        )
    aux_features = load_cub_attributes(cub_root, class_names)

    return image_features_by_class, aux_features, class_names


class CUBPrecomputedDataset(Dataset):
    """
    Dataset of precomputed GoogLeNet image features for CUB zero-shot learning.

    Each item is (feature_vector, class_label) where:
      - feature_vector: (1024,) float32 GoogLeNet feature
      - class_label: int in [0, n_classes_in_split - 1]

    Class-level auxiliary features (for building prototypes) are available via
    the .aux_features attribute: (n_classes, aux_dim) float32 array.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        cub_root: Optional[str] = None,
        test_time: bool = False,
    ):
        """
        Args:
            data_root: path to cvpr2016_cub directory
            split: "train", "val", "test", or "trainval"
            cub_root: path to CUB_200_2011
            test_time: use middle crop only for image features (paper test protocol)
        """
        image_features_by_class, aux_features, class_names = load_split_data(
            data_root, split, cub_root=cub_root
        )
        self.class_names = class_names
        self.aux_features = aux_features   # (n_classes, aux_dim)
        self.test_time = test_time

        all_feats, all_labels = [], []
        for cls_idx, feats in enumerate(image_features_by_class):
            all_feats.append(feats)
            all_labels.extend([cls_idx] * feats.shape[0])

        # (N, 1024, 10) — all crops retained; selection happens in __getitem__
        self.features = np.concatenate(all_feats, axis=0)
        self.labels = np.array(all_labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        crops = self.features[idx]   # (1024, 10)
        if self.test_time:
            feat = crops[:, MIDDLE_CROP_IDX]
        else:
            # Random crop per access, matching Reed et al.'s torch.randperm sampling
            crop_idx = np.random.randint(0, crops.shape[1])
            feat = crops[:, crop_idx]
        return torch.from_numpy(feat.copy()), int(self.labels[idx])
