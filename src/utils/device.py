# Always get device through this script; use GPU if available

import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [x.to(device) for x in batch]
    return batch.to(device)
