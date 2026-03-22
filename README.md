# Prototypical Networks – Project Overview

This repository contains code and configuration for experiments based on **Prototypical Networks** for few-shot (and related) learning across several datasets (Omniglot, miniImageNet, CUB, Speech Commands).

**Anchor paper:**
> Snell, J., Swersky, K., & Zemel, R. S. (2017). Prototypical networks for few-shot learning. *Advances in Neural Information Processing Systems*, 30. [https://arxiv.org/abs/1703.05175](https://arxiv.org/abs/1703.05175)

## Project structure

```
mlmi4_advanced_ml/
├── README.md
├── requirements.txt           # Python dependencies
├── loss.py                    # Loss / objective and distance metric helpers
├── model.py                   # Prototypical network encoder (vision)
├── model_speech.py            # Prototypical network encoders (speech)
├── protonet_sampler.py        # Episode / batch sampling utilities
├── configs/
│   ├── omniglot_config.yaml   # Omniglot few-shot experiment config
│   ├── miniimagenet_config.yaml # miniImageNet few-shot experiment config
│   ├── cub_config.yaml        # CUB zero-shot experiment config
│   └── speech_config.yaml     # Speech Commands few-shot experiment config
└── src/
    ├── __init__.py
    ├── data_loader/
    │   ├── __init__.py
    │   ├── dataloader_omniglot.py     # Omniglot few-shot loader
    │   ├── dataloader_miniImageNet.py # miniImageNet few-shot loader
    │   ├── dataloader_cub.py          # CUB dataset loader
    │   └── dataloader_speech.py       # Speech Commands loader
    ├── training/
    │   ├── train_fewshot.py           # Few-shot training (Omniglot, miniImageNet)
    │   ├── train_fewshot_speech.py    # Few-shot training (Speech Commands)
    │   └── train_zeroshot.py          # Zero-shot training (CUB, Table 3 replication)
    └── utils/
        ├── __init__.py
        ├── device.py                  # Device (CPU / GPU) utilities
        └── seed.py                    # Reproducibility / seeding utilities
```

## Installation

From the project root:

```bash
cd mlmi4_advanced_ml
python -m venv mlmi4_env
source mlmi4_env/bin/activate  # On Windows: mlmi4_env\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Configs live under `configs/`. Each config controls:

- **Data**: dataset choice, paths, N-way / K-shot, number of episodes
- **Model**: embedding dimension, backbone details, etc.
- **Training**: learning rate, number of epochs, batch / episode size
- **Experiment**: checkpoint, log, and results directories

## Running baseline replications

### Omniglot few-shot (Table 4 replication)

1. From the **project root** run:

   ```bash
   PYTHONPATH=. python src/training/train_fewshot.py configs/omniglot_config.yaml
   ```

2. The Omniglot dataset is automatically downloaded to `data/omniglot` via `torchvision` on first run.
3. The best model is saved to `results/omniglot_baseline/best_model.pt`. Final test accuracy is the few-shot result to report.

### miniImageNet few-shot (Table 2 replication)

1. Download [miniImageNet](https://drive.google.com/file/d/1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk/view) and extract to `data/miniimagenet/` so that the split CSV files and image directories are present.
2. From the **project root** run:

   ```bash
   PYTHONPATH=. python src/training/train_fewshot.py configs/miniimagenet_config.yaml
   ```

3. Results are saved under `results/table2_1shot/`.

### CUB zero-shot (Table 3 replication)

1. Download [CUB-200-2011](https://data.caltech.edu/records/65de6-vp158) and extract to `data/CUB_200_2011/` so that `data/CUB_200_2011/images.txt` and `data/CUB_200_2011/images/` exist.
2. Place the 312-dim continuous class attribute file at `data/CUB_200_2011/attributes/class_attribute_labels_continuous.txt`.
3. Download the precomputed GoogLeNet features (`cvpr2016_cub`, Reed et al., 2016) and place them at `data/cvpr2016_cub/` so that `data/cvpr2016_cub/images/` exists.
4. From the **project root** run:

   ```bash
   python src/training/train_zeroshot.py --data_root data/cvpr2016_cub --cub_root data/CUB_200_2011
   ```

   Or using the config file:

   ```bash
   python src/training/train_zeroshot.py --config configs/cub_config.yaml
   ```

   The best checkpoint is saved to `experiments/checkpoints/cub_zeroshot_precomputed_euclidean_best.pt` by default.

### Speech Commands few-shot keyword spotting

1. From the **project root** run:

   ```bash
   PYTHONPATH=. python src/training/train_fewshot_speech.py configs/speech_config.yaml
   ```

2. The Speech Commands v0.02 dataset is automatically downloaded to `data/speech` via `torchaudio` on first run.
3. The best model is saved to `results/speech_baseline/best_model.pt`. Final test accuracy is the few-shot result to report.

## Running extensions

Extensions are controlled via config values or command-line flags — no code changes needed.

### DropBlock regularisation (vision)

Enable DropBlock in the vision encoder by setting `dropblock_size` to a non-zero value (e.g. `5`) in any vision config:

```yaml
model_params:
  dropblock_size: 5      # 0 = disabled (baseline); 5 = extension
  dropblock_prob: 0.1
```

### Alternative distance metrics

The `distance` field in any config accepts:

| Value | Description |
|---|---|
| `euclidean` | Standard Euclidean distance (paper baseline) |
| `diagonal` | Learned diagonal (per-dimension) scaling |
| `lowrank` | Learned low-rank Mahalanobis approximation |

```yaml
distance: "diagonal"
```

### ResNet-18 backbone (miniImageNet)

Switch from the default 4-layer conv backbone to ResNet-18:

```yaml
model_params:
  backbone: "resnet18"   # default: "conv4"
```

### Data augmentation (miniImageNet)

Enable cutout or colour jitter augmentation:

```yaml
data:
  use_cutout: true
  cutout_prob: 0.5
  use_colour_jitter: true
```

### Speech model architecture

Switch between the C64 convolutional baseline and TC-ResNet8:

```yaml
model_params:
  model_type: "tc_resnet8"   # default: "c64"
```
