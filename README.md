# Prototypical Networks – Project Overview

This repository contains code and configuration for experiments based on **Prototypical Networks** for few‑shot (and related) learning across several datasets (Omniglot, miniImageNet, CUB).

> Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. *Advances in Neural Information Processing Systems*, 30.

## Project structure

```
mlmi4_advanced_ml/
├── README.md
├── requirements.txt           # Python dependencies
├── loss.py                    # Loss / objective helpers
├── model.py                   # Model definitions (Prototypical network etc.)
├── protonet_sampler.py        # Episode / batch sampling utilities
├── configs/
│   ├── omniglot_config.yaml   # Omniglot experiment config
│   ├── miniimagenet_config.yaml # MimiImageNet config
│   └── cub_config.yaml        # CUB / fine‑grained image experiments
├── experiments/
│   ├── checkpoints/           # Saved model checkpoints
│   ├── logs/                  # Training logs
│   └── results/               # Evaluation results / metrics
└── src/
    ├── __init__.py
    ├── data_loader/
    │   ├── __init__.py
    │   ├── dataloader_omniplot.py     # Omniglot‑style characters
    │   ├── dataloader_miniImageNet.py # miniImageNet few‑shot loader
    │   ├── dataloader_cub.py          # CUB dataset loader
    │   └── dataloader_speech.py       # Speech loader
    ├── training/
    │   ├── train_fewshot.py           # Few‑shot training script
    │   └── train_zeroshot.py          # Zero‑shot training script
    └── utils/
        ├── __init__.py
        ├── device.py                  # Device (CPU / GPU) utilities
        ├── metrics.py                 # Metrics and evaluation helpers
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

Configs live under `configs/`:

- **`omniglot_config.yaml`**: settings for Omniglot experiments  
- **`miniimagenet_config.yaml`**: settings for miniImageNet experiments  
- **`cub_config.yaml`**: settings for CUB / fine‑grained classification

Each config typically controls:

- **Data**: dataset choice, paths, N‑way / K‑shot, number of episodes  
- **Model**: embedding dimension, backbone details, etc.  
- **Training**: learning rate, number of epochs, batch / episode size  
- **Experiment**: checkpoint, log, and results directories  

## Running experiments (high level)

The typical workflow could be:

1. **Choose / edit a config** in `configs/` for your dataset.
2. **Run a training script**, e.g. a few‑shot experiment via `src/training/train_fewshot.py`.
3. **Inspect outputs** under `experiments/checkpoints`, `experiments/logs`, and `experiments/results`.

### Omniglot Few-shot Experiment (Thao)

1. From the **project root** `mlmi4_advanced_ml` run:

   ```bash
   PYTHONPATH=. python src/training/train_fewshot.py configs/omniglot_config.yaml
2. The Omniglot dataset will be automatically downloaded to `data/omniglot` via `torchvision` upon the first run.
3. Best model is saved to `results/omniglot_baseline/best_model.pt`; final test accuracy is the few-shot result to report.
