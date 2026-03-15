"""
Training script for Prototypical Networks - Few-shot KWS Learning

"""


import os
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

# Import the Speech dataloader
from src.data_loader.dataloader_speech import get_dataloader as get_speech_loader

# Import both Speech architectures
from model_speech import SpeechC64, TCResNet8


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
    train_loader = get_speech_loader(data_config, split='train')
    val_loader = get_speech_loader(data_config, split='val')

    train_n_shot = data_config['train_params']['n_shot']
    val_n_shot = data_config.get('val_params', data_config['test_params'])['n_shot']
    test_n_shot = data_config['test_params']['n_shot']
    n_episodes = data_config['train_params']['n_episodes']

    # ---- 2. Model Setup ----
    # Both speech models output a 64-dimensional embedding
    embed_dim = 64

    # Extract model parameters from config
    model_config = config.get('model_params', {})
    model_type = model_config.get('model_type', 'c64')

    if model_type == 'tc_resnet8':
        print("--> Initializing TC-ResNet-8")
        model = TCResNet8(embedding_dim=embed_dim).to(device)
    else:
        print("--> Initializing Baseline SpeechC64")
        model = SpeechC64(
            hidden_dim=model_config.get('hidden_dim', 64),
            dropblock_size=model_config.get('dropblock_size', 0),
            dropblock_prob=model_config.get('dropblock_prob', 0.1)
        ).to(device)

    # ---- 3. Distance Setup ----
    distance_fn = build_distance(config, z_dim=embed_dim, device=device)

    lr = config.get('lr', 1e-3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.get('lr_step', 2000), gamma=0.5)
    
    # ---- Training loop ----
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    log_interval = config.get('log_interval', 100)
    val_interval = config.get('val_interval', 1000)
    output_dir = config.get('output_dir', 'results')

    for episode, batch in enumerate(train_loader, 1):
        x, y = batch
        x = x.to(device)

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
                    'episode': episode,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc
                }, save_path)
                print(f"  --> New best model saved (acc={val_acc*100:.2f}%)")

    # ---- Final test evaluation (1000 episodes, as in paper) ----
    print("\n" + "=" * 60)
    print("Final evaluation (1000 episodes):")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(
        os.path.join(output_dir, 'best_model.pt'),
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test dataset
    test_loader = get_speech_loader(data_config, split='test')

    test_acc, test_loss, test_ci = evaluate(
        model, test_loader, test_n_shot, distance_fn, device
        )
    print(f"Test Accuracy: {test_acc*100:.2f}% ± {test_ci*100:.2f}% ")

if __name__ == '__main__':
    # Default to the speech config
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/speech_config.yaml"

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