import os
import sys
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
import torchvision.transforms.functional as F

# Note: Ensure project root is in PYTHONPATH (e.g., run with PYTHONPATH=.)
# E.g.: PYTHONPATH=. python src/training/train_fewshot.py
from protonet_sampler import PrototypicalBatchSampler


'''
dataloader_omniglot.py logic

Original (Snell): https://github.com/jakesnell/prototypical-networks/blob/master/protonets/data/omniglot.py

'''

class OmniglotRotated(Dataset):
  '''
  Omniglot Dataset
  Custom Omniglot dataset that multiplies classes by 4x via 90-degree rotations.
  Implements the exact 1200 (train) / 423 (test) split from Vinyals et al.

  Goal: 
  Load base dataset and perform data augmentation (rotation)
  If the original dataset has N total classes (0 to N-1), and
  an image originally belong to class c:
  1. 0 rotation: Class c
  2. 90 rotation: Class N + c
  3. 180 rotation: Class 2N + c
  4. 270 rotation: Class 3N + c

  Rationale: 
  For the Omniglot experiment, each rotation is treated as a
  completely different character. This allows few-shot learning to learn
  as many distinct classes as possible to learn a good embedding space

  '''
  # ==================================================================
  # STEP 1: DOWNLOAD / LOAD DATA & AUGMENTATION INSTRUCTIONS
  # ==================================================================
  def __init__(self, root_dir, split, transform=None, download=True):
    '''
    Generate 2 empty lists to reflect augmented inputs

    self.labels     : Flat list storing new, augmented class ID for sampler
    self.flat_items : List of tuples containing (base_index, rotation_angle)
    '''
    super().__init__()
    self.root_dir = root_dir
    self.split = split
    self.transform = transform

    # 1. Load base dataset
    # 1.1 Download both PyTorch subsets to get all 1623 classes
    # Accessing private attributes of torchvision.datasets.Omniglot (tested with torchvision >=0.15).
    # If this breaks on upgrade, iterate dataset manually instead.
    bg_dataset = datasets.Omniglot(root_dir, background=True, download=download)
    ev_dataset = datasets.Omniglot(root_dir, background=False, download=download)
    

    # 1.2 Map every item to (dataset_obj, original_idx, continuous_label)
    all_items = []
    
    # Background dataset (Labels 0 to 963)
    for original_idx in range(len(bg_dataset._flat_character_images)):
        _, label = bg_dataset._flat_character_images[original_idx]
        all_items.append((bg_dataset, original_idx, label))
        
    # Evaluation dataset (Labels 964 to 1622)
    bg_num_classes = len(bg_dataset._characters)
    for original_idx in range(len(ev_dataset._flat_character_images)):
        _, label = ev_dataset._flat_character_images[original_idx]
        all_items.append((ev_dataset, original_idx, label + bg_num_classes))

    # 1.3 Apply the 1200 / 423 split as the paper
    if split == 'train':
      self.base_items = [(ds, idx, lbl) for ds, idx, lbl in all_items if lbl < 1200]
      self.num_base_classes = 1200
    else: # test
      self.base_items = [(ds, idx, lbl - 1200) for ds, idx, lbl in all_items if lbl >= 1200]
      self.num_base_classes = 423

    # 3. Initialize label and image list
    self.labels = []
    self.flat_items = []

    # 4. Perform augmentation on base dataset
    for i in range(len(self.base_items)):
      _, _, original_class = self.base_items[i]
      
      for rotation in [0, 90, 180, 270]:
        rotation_multiplier = rotation // 90
        new_class = original_class + (rotation_multiplier * self.num_base_classes)
        
        self.labels.append(new_class)
        self.flat_items.append((i, rotation))

  # ==================================================================
  # STEP 2: __len__
  # ==================================================================
  def __len__(self):
    '''
    Return the total number of augmented images (N * 4)
    '''
    return len(self.labels)
  
  # ==================================================================
  # STEP 3: LOOK UP IMAGES & PERFORM AUGMENTATION
  # ==================================================================
  def __getitem__(self, index):

    base_idx, angle = self.flat_items[index]
    dataset_obj, original_idx, _ = self.base_items[base_idx]

    # Let PyTorch handle the paths! It returns (PIL Image, label)
    image, _ = dataset_obj[original_idx] 

    # Rotate image (PyTorch loads Omniglot in 'L' grayscale mode already)
    if angle > 0:
        image = F.rotate(image, angle)
    
    # Apply the transform (Resize to 28x28 and convert ToTensor)
    if self.transform is not None:
      image = self.transform(image)
    
    # Return the tensor and new augmented label
    return image, self.labels[index]

# ==================================================================
# STEP 4: INSTANTIATE DATALOADER
# ==================================================================
def get_dataloader(config, split):
  '''
  Create the Omniglot DataLoader.
  OG paper resize grayscale images to 28 x 28
  However, Omniglot images are natively grayscale.
    
  Args:
    config (dict): The loaded YAML configuration dictionary
    split (str): 'train' or 'test' (or 'val').
  '''
  # 1. Define the transform
  omniglot_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
  
  # 2. Instantiate augmented dataset
  root_dir = config.get('data_dir', './data/omniglot') # In case root_dir is not defined in yaml
  dataset = OmniglotRotated(
        root_dir=root_dir, 
        split=split, 
        transform=omniglot_transform
    )
  
  # 3. Handle episode counts based on split (100 for train, 1000 for test)
  if split == 'train':
      params = config['train_params']
  elif split == 'val':
      params = config.get('val_params', config['test_params'])
  else:
      params = config['test_params']
  
  # 4. Instantiate custom sampler
  sampler = PrototypicalBatchSampler(
        labels=dataset.labels,
        n_way=params['n_way'],
        n_shot=params['n_shot'],
        n_query=params['n_query'],
        n_episodes=params['n_episodes']
    )
  
  # 5. Instantiate DataLoader
  dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=config.get('num_workers', 4), 
        pin_memory=True                          
    )

  return dataloader