"""
Verify which .t7 crop index corresponds to the middle crop of the original image.

Approach: take a known image, extract GoogLeNet features for each crop type using
torchvision, then compare cosine similarity against each of the 10 stored crop
vectors in the .t7 file. The highest-similarity index = that crop type.

Usage (run from repo root on HPC):
  python scripts/verify_crop_index.py \
      --cub_root data/CUB_200_2011 \
      --t7_root  data/cvpr2016_cub \
      --n_images 5
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from PIL import Image

try:
    import torchfile
except ImportError:
    sys.exit("pip install torchfile")


def get_googlenet_feature(img_tensor: torch.Tensor, model) -> np.ndarray:
    """Run GoogLeNet (no final FC) on a (1,3,H,W) tensor, return (1024,) numpy array."""
    with torch.no_grad():
        feat = model(img_tensor.unsqueeze(0))
    return feat.squeeze(0).cpu().numpy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub_root", default="data/CUB_200_2011")
    parser.add_argument("--t7_root",  default="data/cvpr2016_cub")
    parser.add_argument("--n_images", type=int, default=5,
                        help="Number of images to test per class (averaged for robustness)")
    parser.add_argument("--n_classes", type=int, default=3,
                        help="Number of classes to test")
    args = parser.parse_args()

    # Load GoogLeNet with final FC removed
    model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = T.ToTensor()

    # The 10 crops: 5 of original + 5 of horizontally-flipped
    # Crop types in order we'll test: middle(center), UL, UR, LL, LR
    def make_crops(img: Image.Image):
        crops = []
        for flip in [False, True]:
            im = T.functional.hflip(img) if flip else img
            im_resized = T.Resize(256)(im)
            crops.append(T.CenterCrop(224)(im_resized))          # middle
            crops.append(T.functional.crop(im_resized, 0,   0,   224, 224))   # upper-left
            crops.append(T.functional.crop(im_resized, 0,   32,  224, 224))   # upper-right
            crops.append(T.functional.crop(im_resized, 32,  0,   224, 224))   # lower-left
            crops.append(T.functional.crop(im_resized, 32,  32,  224, 224))   # lower-right
        return crops  # 10 crops in order: orig_middle, orig_UL, orig_UR, orig_LL, orig_LR,
                      #                    flip_middle, flip_UL, flip_UR, flip_LL, flip_LR

    crop_names = [
        "orig_middle", "orig_UL", "orig_UR", "orig_LL", "orig_LR",
        "flip_middle", "flip_UL", "flip_UR", "flip_LL", "flip_LR",
    ]

    # Read image list from CUB
    images_txt = os.path.join(args.cub_root, "images.txt")
    labels_txt  = os.path.join(args.cub_root, "image_class_labels.txt")

    id_to_path, id_to_cls = {}, {}
    with open(images_txt) as f:
        for line in f:
            iid, path = line.strip().split()
            id_to_path[int(iid)] = path
    with open(labels_txt) as f:
        for line in f:
            iid, cls = line.strip().split()
            id_to_cls[int(iid)] = int(cls)

    # Group by class
    cls_to_ids: dict = {}
    for iid, cls in id_to_cls.items():
        cls_to_ids.setdefault(cls, []).append(iid)

    # For accumulating similarity scores across all tested images
    total_sims = np.zeros((10, 10))  # [t7_crop_idx, our_crop_idx]
    n_tested = 0

    # Read class list from t7_root
    t7_classes_file = os.path.join(args.t7_root, "allclasses.txt")
    with open(t7_classes_file) as f:
        all_class_names = [l.strip() for l in f if l.strip()]

    for cls_name in all_class_names[:args.n_classes]:
        cls_num = int(cls_name.split(".")[0])   # 1-based
        t7_file = os.path.join(args.t7_root, "images", f"{cls_name}.t7")
        if not os.path.exists(t7_file):
            print(f"Skipping {cls_name} - t7 missing")
            continue

        arr = np.array(torchfile.load(t7_file), dtype=np.float32)
        # arr: (n_imgs, 1024, 10)

        img_ids = cls_to_ids.get(cls_num, [])[:args.n_images]
        if not img_ids:
            continue

        print(f"\nClass {cls_name} ({len(img_ids)} images)")

        for img_rank, iid in enumerate(img_ids):
            img_path = os.path.join(args.cub_root, "images", id_to_path[iid])
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert("RGB")
            our_crops = make_crops(img)

            # Extract GoogLeNet features for each of our 10 crops
            our_feats = []
            for crop in our_crops:
                t = normalize(to_tensor(crop))
                our_feats.append(get_googlenet_feature(t, model))

            # Compare each t7 crop index against each of our crop types
            t7_img_feats = arr[img_rank]   # (1024, 10)

            sims = np.zeros((10, 10))
            for t7_idx in range(10):
                for our_idx in range(10):
                    sims[t7_idx, our_idx] = cosine_sim(
                        t7_img_feats[:, t7_idx], our_feats[our_idx]
                    )

            total_sims += sims
            n_tested += 1

            # Per-image: best match for each t7 index
            best = sims.argmax(axis=1)
            print(f"  img {img_rank}: t7[0]→{crop_names[best[0]]}  "
                  f"t7[1]→{crop_names[best[1]]}  t7[2]→{crop_names[best[2]]}  ...")

    if n_tested == 0:
        print("No images tested.")
        return

    avg_sims = total_sims / n_tested
    print(f"\n{'='*60}")
    print(f"Average cosine similarity over {n_tested} images")
    print(f"Rows = t7 crop index (0..9), Cols = our crop type")
    print(f"{'':12s}", "  ".join(f"{n[:8]:8s}" for n in crop_names))
    for t7_idx in range(10):
        best = avg_sims[t7_idx].argmax()
        row = "  ".join(
            f"{'>>':>8s}" if i == best else f"{avg_sims[t7_idx,i]:8.4f}"
            for i in range(10)
        )
        print(f"t7[{t7_idx}]      {row}  ← best: {crop_names[best]}")

    print(f"\nConclusion:")
    for t7_idx in range(10):
        best = avg_sims[t7_idx].argmax()
        print(f"  t7 crop index {t7_idx} = {crop_names[best]}")


if __name__ == "__main__":
    main()
