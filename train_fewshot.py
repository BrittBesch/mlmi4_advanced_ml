"""
Training script for Prototypical Networks - Few-shot Learning

Paper hyperparameters:
  - Optimizer: Adam, lr=1e-3, halved every 2000 episodes
  - No regularization other than batch normalization
  - miniImageNet: 30-way (1-shot) or 20-way (5-shot), 15 queries/class
  - Omniglot: 60-way, 5 queries/class, match train/test shot

"""

import os
import argparse
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# ---- Team module imports and dependencies ----
from model import PrototypicalNetwork  #from jack's model.py
from loss import prototypical_loss  #name matching with britt's loss.py code
from protonet_sampler import PrototypicalBatchSampler ##from Britt
from dataloader_miniImageNet import MiniImageNet ## from Thao
from dataloader_omniplot import OmniglotRotated ## from Thao


def get_dataloader(dataset, sampler):
    """
    Wrap a dataset with the episodic batch sampler into a DataLoader.

    Args:
        dataset: Dataset object (MiniImageNetDataset or OmniglotDataset)
        sampler: PrototypicalBatchSampler instance

    Returns:
        DataLoader yielding episodic batches
    """
    return DataLoader(dataset, batch_sampler=sampler)


def setup_dataloaders(args):
    """
    Create train and validation datasets + samplers based on args.

    Returns:
        train_loader, val_loader, in_channels
    """
    if args.dataset == 'miniimagenet':
        train_dataset = MiniImageNet(root=args.data_root, split='train')
        val_dataset = MiniImageNet(root=args.data_root, split='val')
        in_channels = 3

        # Paper: 30-way for 1-shot, 20-way for 5-shot
        if args.train_way is None:
            train_n_way = 30 if args.n_shot == 1 else 20
        else:
            train_n_way = args.train_way
        n_query = args.n_query or 15

    elif args.dataset == 'omniglot':
        train_dataset = OmniglotRotated(root=args.data_root, split='train')
        val_dataset = OmniglotRotated(root=args.data_root, split='test')
        in_channels = 1

        # Paper: 60-way, 5 queries/class
        train_n_way = args.train_way or 60
        n_query = args.n_query or 5

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Create episodic samplers
    train_sampler = PrototypicalBatchSampler(
        labels=train_dataset.labels,
        classes_per_episode=train_n_way,
        samples_per_class=args.n_shot + n_query,
        episodes=args.n_episodes,
    )

    val_sampler = PrototypicalBatchSampler(
        labels=val_dataset.labels,
        classes_per_episode=args.test_way,
        samples_per_class=args.n_shot + n_query,
        episodes=args.val_episodes,
    )

    train_loader = get_dataloader(train_dataset, train_sampler)
    val_loader = get_dataloader(val_dataset, val_sampler)

    return train_loader, val_loader, in_channels, train_n_way, n_query


def train_episode(model, support, query, n_way, n_shot, n_query, optimizer):
    """
    This function runs a single training episode.

    Args:
        model: PrototypicalNetwork
        support: Support images, shape (n_way * n_shot, C, H, W)
        query: Query images, shape (n_way * n_query, C, H, W)
        n_way: Number of classes
        n_shot: Support examples per class
        n_query: Query examples per class
        optimizer: Optimizer (we go for Adam with lr=1e-3 as per initial paper)

    Returns:
        loss (float), accuracy (float)
    """
    model.train()

    # Forward pass
    _, _, dists = model(support, query, n_way, n_shot, n_query)

    # Loss from loss.py
    loss, acc = prototypical_loss(dists, n_way=n_way, n_query=n_query)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), acc.item()


def evaluate(model, val_loader, n_way, n_shot, n_query, device):
    """
    Evaluate the model over all episodes in val_loader.

    Args:
        model: PrototypicalNetwork
        val_loader: DataLoader with episodic sampler
        n_way: Number of classes per episode
        n_shot: Support examples per class
        n_query: Query examples per class
        device: torch device

    Returns:
        mean_acc, mean_loss, ci95 (95% confidence interval on accuracy)
    """
    model.eval()
    losses = []
    accs = []

    with torch.no_grad():
        for batch in val_loader:
            # Split batch into support and query
            # batch shape: (n_way * (n_shot + n_query), C, H, W)
            x, y = batch
            x = x.to(device)

            # Reshape: (n_way, n_shot + n_query, C, H, W)
            x = x.view(n_way, n_shot + n_query, *x.shape[1:])
            support = x[:, :n_shot].contiguous().view(-1, *x.shape[2:])
            query = x[:, n_shot:].contiguous().view(-1, *x.shape[2:])

            _, _, dists = model(support, query, n_way, n_shot, n_query)
            loss, acc = prototypical_loss(dists, n_way=n_way, n_query=n_query)

            losses.append(loss.item())
            accs.append(acc.item())

    mean_acc = np.mean(accs)
    mean_loss = np.mean(losses)
    ci95 = 1.96 * np.std(accs) / np.sqrt(len(accs))   ### gives us our CI for accuracy

    return mean_acc, mean_loss, ci95


def train(args):
    """Main training loop."""

    # ---- Device ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---- Data ----
    train_loader, val_loader, in_channels, train_n_way, n_query = \
        setup_dataloaders(args)

    # ---- Model ----
    # embed_dim depends on input size: 64 for 28x28 (Omniglot), 1600 for 84x84 (miniImageNet)
    embed_dim = 64 if args.dataset == 'omniglot' else 1600

    model = PrototypicalNetwork(
        in_channels=in_channels,
        hidden_dim=64,
        distance=args.distance,
        embed_dim=embed_dim if args.distance == 'mahalanobis' else None,
    ).to(device)

    # ---- Optimizer: Adam, lr=1e-3 ----
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # LR schedule: halve every 2000 episodes
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=0.5
    )

'''
### can add this in if we are tracking our training to ensure everything is set up correctly. We can also add in the distance function we want to use here as well.
    print(f"\nTraining config:")
    print(f"  Dataset:        {args.dataset}")
    print(f"  Distance:       {args.distance}")
    print(f"  Train way:      {train_n_way}")
    print(f"  Test way:       {args.test_way}")
    print(f"  Shot:           {args.n_shot}")
    print(f"  Query/class:    {n_query}")
    print(f"  Episodes:       {args.n_episodes}")
    print(f"  LR:             {args.lr}")
    print(f"  LR step:        {args.lr_step}")
    print()
'''
    # ---- Training loop ----
    best_val_acc = 0.0
    train_losses = []
    train_accs = []

    for episode, batch in enumerate(train_loader, 1):
        x, y = batch
        x = x.to(device)

        # Reshape batch into support/query split
        # Sampler gives us (n_way, n_shot + n_query) samples grouped by class
        x = x.view(train_n_way, args.n_shot + n_query, *x.shape[1:])
        support = x[:, :args.n_shot].contiguous().view(-1, *x.shape[2:])
        query = x[:, args.n_shot:].contiguous().view(-1, *x.shape[2:])

        loss, acc = train_episode(
            model, support, query,
            train_n_way, args.n_shot, n_query, optimizer
        )
        scheduler.step()

        train_losses.append(loss)
        train_accs.append(acc)

        # ---- Logging ----
        if episode % args.log_interval == 0:
            avg_loss = np.mean(train_losses[-args.log_interval:])
            avg_acc = np.mean(train_accs[-args.log_interval:])
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Episode {episode}/{args.n_episodes} | "
                f"Loss: {avg_loss:.4f} | "
                f"Acc: {avg_acc*100:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

        # ---- Validation ----
        if episode % args.val_interval == 0:
            val_acc, val_loss, val_ci = evaluate(
                model, val_loader, args.test_way, args.n_shot,
                n_query, device
            )
            print(
                f"  --> Validation | "
                f"Loss: {val_loss:.4f} | "
                f"Acc: {val_acc*100:.2f}% ± {val_ci*100:.2f}%"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(args.output_dir, 'best_model.pt')
                torch.save({
                    'episode': episode,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'args': vars(args),
                }, save_path)
                print(f"  --> New best model saved (acc={val_acc*100:.2f}%)")

    # ---- Final test evaluation (1000 episodes, as in paper) ----
    print("\n" + "=" * 60)
    print("Final evaluation (1000 episodes):")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(
        os.path.join(args.output_dir, 'best_model.pt'),
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test dataset
    if args.dataset == 'miniimagenet':
        test_dataset = MiniImageNet(root=args.data_root, split='test')
    elif args.dataset == 'omniglot':
        test_dataset = OmniglotRotated(root=args.data_root, split='test')

    test_sampler = PrototypicalBatchSampler(
        labels=test_dataset.labels,
        classes_per_episode=args.test_way,
        samples_per_class=args.n_shot + n_query,
        episodes=1000,
    )
    test_loader = get_dataloader(test_dataset, test_sampler)

    test_acc, test_loss, test_ci = evaluate(
        model, test_loader, args.test_way, args.n_shot, n_query, device
    )
    print(
        f"Test Accuracy: {test_acc*100:.2f}% ± {test_ci*100:.2f}% "
        f"({args.test_way}-way {args.n_shot}-shot, Euclidean distance)"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prototypical Networks - Few-shot Training'
    )

    # Dataset
    parser.add_argument('--dataset', type=str, default='miniimagenet',
                        choices=['omniglot', 'miniimagenet'],
                        help='Dataset to use (default: miniimagenet)')
    parser.add_argument('--data-root', type=str, default='data/miniImageNet',
                        help='Path to dataset root directory')

    # Episode configuration
    parser.add_argument('--n-shot', type=int, default=5,
                        help='Number of support examples per class (N_S)')
    parser.add_argument('--train-way', type=int, default=None,
                        help='Classes per training episode (N_C). '
                             'Default: 60 for Omniglot, 30/20 for miniImageNet')
    parser.add_argument('--test-way', type=int, default=5,
                        help='Classes per test episode (default: 5)')
    parser.add_argument('--n-query', type=int, default=None,
                        help='Query examples per class (N_Q). '
                             'Default: 5 for Omniglot, 15 for miniImageNet')

    # Model
    parser.add_argument('--distance', type=str, default='euclidean',
                        choices=['euclidean', 'cosine', 'mahalanobis'],
                        help='Distance function (paper: euclidean)')

    # Training
    parser.add_argument('--n-episodes', type=int, default=20000,
                        help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (paper: 1e-3)')
    parser.add_argument('--lr-step', type=int, default=2000,
                        help='Halve LR every N episodes (paper: 2000)')

    # Logging & saving
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Log training stats every N episodes')
    parser.add_argument('--val-interval', type=int, default=1000,
                        help='Run validation every N episodes')
    parser.add_argument('--val-episodes', type=int, default=600,
                        help='Number of validation episodes (paper: 600)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save model checkpoints')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)