"""
Training script for Prototypical Networks - Few-shot Learning

Paper hyperparameters:
  - Optimizer: Adam, lr=1e-3, halved every 2000 episodes
  - No regularization other than batch normalization
  - miniImageNet: 30-way (1-shot) or 20-way (5-shot), 15 queries/class
  - Omniglot: 60-way, 5 queries/class, match train/test shot

"""

import os
import pickle
import random
import yaml
import sys
import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---- Team module imports and dependencies ----
from model import ProtoNetEncoder  #from jack's model.py
from loss import prototypical_loss, build_distance  #name matching with britt's loss.py code

from src.data_loader.dataloader_miniImageNet import get_dataloader as get_miniimagenet_loader
from src.data_loader.dataloader_omniglot import get_dataloader as get_omniglot_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")  # NEW

def train_episode(model, x, y, n_support, optimizer, distance_fn):
    """
    This function runs a single training episode.

    Args:
        model: ProtoNetEncoder - embeds images into flat vectors of shape (B, D)
        x: All images (support + query) for this episode, shape (B, C, H, W)
        y: Integer class labels for each image, shape (B,)
        n_support: Number of support examples per class (N_S)
        optimizer: Optimizer (Adam, lr=1e-3 per paper)
        distance_fn: Distance function

    Returns:
        loss (float), accuracy (float)
    """
    model.train()

    # Model outputs flat embeddings of shape (B, D)
    embeddings = model(x)

    # 2. Loss function handles prototype slicing and distance math
    loss, acc = prototypical_loss(
            embeddings=embeddings, 
            labels=y, 
            n_support=n_support, 
            distance_fn=distance_fn
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), acc.item()


def evaluate(model, val_loader, n_support, distance_fn, device):
    '''
    Args:
    
    model: ProtoNetEncoder - embeds images into flat vectors of shape (B, D)
    val_loader: DataLoader yielding episodic batches 
    n_support: Number of support examples per class (N_S)
    distance_fn: Distance function 
    device: torch device (cuda or cpu)
    '''
    model.eval()
    losses = []
    accs = []

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            embeddings = model(x)
            loss, acc = prototypical_loss(
                            embeddings=embeddings, 
                            labels=y, 
                            n_support=n_support, 
                            distance_fn=distance_fn
            )

            losses.append(loss.item())
            accs.append(acc.item())

    mean_acc = np.mean(accs)
    mean_loss = np.mean(losses)
    ci95 = 1.96 * np.std(accs) / np.sqrt(len(accs))   ### gives us our CI for accuracy

    return mean_acc, mean_loss, ci95


def train(config):      #### will introduce configs with YAML to match britt and thao
    """Main training loop."""

    data_config = config['data']
    dataset_name = data_config.get('dataset', 'miniimagenet')

    # ---- 1. Data Setup ----
    if dataset_name == 'miniimagenet':
        train_loader = get_miniimagenet_loader(data_config, split='train')
        val_loader = get_miniimagenet_loader(data_config, split='val')
        in_channels = 3
    elif dataset_name == 'omniglot':
        train_loader = get_omniglot_loader(data_config, split='train')
        val_loader = get_omniglot_loader(data_config, split='val')
        in_channels = 1
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_n_shot = data_config['train_params']['n_shot']
    val_n_shot = data_config.get('val_params', data_config['test_params'])['n_shot']
    test_n_shot = data_config['test_params']['n_shot']
    n_episodes = data_config['train_params']['n_episodes']

    # ---- 2. Model Setup ----
    # embed_dim depends on input size: 64 for 28x28 (Omniglot), 1600 for 84x84 (miniImageNet)


    # Extract model parameters from config
    model_config = config.get('model_params', {})
    hidden_dim = model_config.get('hidden_dim', 64)
    dropblock_size = model_config.get('dropblock_size', 0)
    dropblock_prob = model_config.get('dropblock_prob', 0.1)

    model = ProtoNetEncoder(
        in_channels=in_channels,
        **model_config                                                                                                                             ######## CHANGED HERE TOO 
        ).to(device)
    
    embed_dim = model.embed_dim if model.embed_dim else (64 if dataset_name == 'omniglot' else 1600)
    distance_fn = build_distance(config, z_dim=embed_dim, device=device)

    lr = config.get('lr', 1e-3)
    # ---- 3. Optimizer Setup ----
    # Include learnable distance parameters if using diagonal or lowrank metric
    if hasattr(distance_fn, 'parameters'):
        optimizer = optim.Adam(
            list(model.parameters()) + list(distance_fn.parameters()),
            lr=lr
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.get('lr_step', 2000), gamma=0.5)
    
    # ---- Training loop ----
    best_val_acc = float('-inf')
    saved_best = False
    train_losses = []
    train_accs = []
    log_interval = config.get('log_interval', 100)
    val_interval = config.get('val_interval', 1000)
    output_dir = config.get('output_dir', 'results')

    for episode, batch in enumerate(train_loader, 1):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        # Reshape batch into support/query split
        # Sampler gives us (n_way, n_shot + n_query) samples grouped by class
        loss, acc = train_episode(
                    model, x, y, train_n_shot, optimizer, distance_fn
                )
        scheduler.step()

        train_losses.append(loss)
        train_accs.append(acc)

        # ---- Logging ----
        if episode % log_interval == 0:
            avg_loss = np.mean(train_losses[-log_interval:])
            avg_acc = np.mean(train_accs[-log_interval:])
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Episode {episode}/{n_episodes} | "
                f"Loss: {avg_loss:.4f} | "
                f"Acc: {avg_acc*100:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

        # ---- Validation ----
        if episode % val_interval == 0:
            val_acc, val_loss, val_ci = evaluate(
                            model, val_loader, val_n_shot, distance_fn, device
                        )
            print(
                f"  --> Validation | "
                f"Loss: {val_loss:.4f} | "
                f"Acc: {val_acc*100:.2f}% ± {val_ci*100:.2f}%"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(output_dir, 'best_model.pt')
                torch.save({
                    'episode': int(episode),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': float(val_acc)
                }, save_path)
                saved_best = True
                print(f"  --> New best model saved (acc={val_acc*100:.2f}%)")

    # ---- Final test evaluation (1000 episodes, as in paper) ----
    print("\n" + "=" * 60)
    print("Final evaluation (1000 episodes):")
    print("=" * 60)

    checkpoint_path = os.path.join(output_dir, 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        print("No valid best checkpoint found during validation; saving final model as fallback.")
        torch.save({
            'episode': int(n_episodes),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': float('nan')
        }, checkpoint_path)

    # Load best model
    checkpoint = torch.load(
        os.path.join(output_dir, 'best_model.pt'),
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test dataset
    if dataset_name == 'miniimagenet':
        test_loader = get_miniimagenet_loader(data_config, split='test')
    elif dataset_name == 'omniglot':
        test_loader = get_omniglot_loader(data_config, split='test')

    test_acc, test_loss, test_ci = evaluate(
        model, test_loader, test_n_shot, distance_fn, device
        )
    print(f"Test Accuracy: {test_acc*100:.2f}% ± {test_ci*100:.2f}% ")

if __name__ == '__main__':

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/miniimagenet_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Moved the print statement down here
    print("\n" + "="*40)
    print("RUNNING EXPERIMENT WITH CONFIG:")
    print("="*40)
    print(yaml.dump(config, default_flow_style=False))
    print("="*40 + "\n")

    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create a unique timestamped folder
    base_dir = config.get('output_dir', 'results')
    dataset = config['data'].get('dataset', 'unknown')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Creates e.g., results/omniglot_baseline/omniglot_20260310_101129
    unique_run_dir = os.path.join(base_dir, f"{dataset}_{timestamp}")
    os.makedirs(unique_run_dir, exist_ok=True)
    
    config['output_dir'] = unique_run_dir
    
    train(config)