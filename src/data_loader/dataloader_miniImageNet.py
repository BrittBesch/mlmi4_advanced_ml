import os
import sys
import csv
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Note: Ensure project root is in PYTHONPATH (e.g., run with PYTHONPATH=.)
# E.g.: PYTHONPATH=. python src/training/train_fewshot.py
from protonet_sampler import PrototypicalBatchSampler

class MiniImageNet(Dataset):
  '''
  miniImageNet Dataset
  Reads from train.csv, val.csv, or test.csv provided by Ravi & Larochelle.
  https://github.com/jakesnell/prototypical-networks/tree/master/data/miniImagenet/splits/ravi
  
  1. Set up paths to CSV split and image folders
  2. Parse CSV and Map labels to integers, as PrototypicalBatchSampler
      requires a flat list of integer labels (e.g., [0, 0, 1, 1, 2...])
  3. Store absolute path and new integer label
  '''
  # ==================================================================
  # STEP 1: LOAD CSV & MAP STRING LABELS TO INTEGERS
  # ==================================================================
  def __init__(self, root_dir, split, transform=None):
    super().__init__()
    self.root_dir = root_dir
    self.split = split
    self.transform = transform

    # 1. Define paths to CSV split and images folder
    csv_path = os.path.join(root_dir, f'{split}.csv')
    images_dir = os.path.join(root_dir, 'images')

    # Inialize empty lists for image paths and integer labels
    self.image_paths = []
    self.labels = []

    label_map = {} # Dictionary to map string labels to integer IDs
    current_id = 0

    # 2. Parse the CSV file
    with open(csv_path, 'r') as f:
      reader = csv.reader(f)
      next(reader) # Skip header
      
      for row in reader:
        filename, string_label = row[0], row[1]
        
        # If we haven't seen this class string before, assign it a new integer ID
        if string_label not in label_map:
          label_map[string_label] = current_id
          current_id += 1
          
        # 3. Store the absolute path and the new integer label
        self.image_paths.append(os.path.join(images_dir, filename))
        self.labels.append(label_map[string_label])

  # ==================================================================
  # STEP 2: __len__
  # ==================================================================
  def __len__(self):
    '''
    Return the total number of images in that specific CSV split.
    '''
    return len(self.labels)

  # ==================================================================
  # STEP 3: LOOK UP IMAGES & PERFORM TRANSFORMATION
  # ==================================================================
  def __getitem__(self, index):
    # 3.1 Fetch image path & integer label
    image_path = self.image_paths[index]
    label = self.labels[index]

    # 3.2 Open images
    image = Image.open(image_path).convert('RGB')

    # 3.3 Apply transformation if not none
    if self.transform is not None:
      image = self.transform(image)
    
    # 3.4 Return images and labels
    return image, label

# ==================================================================
# STEP 4: INSTANTIATE DATALOADER & PERFORM CUTOUT
# ==================================================================
def get_dataloader(config, split):
  
  # 4.1 Define the transformations
  transform_list = [
        transforms.Resize((84,84)),
        transforms.ToTensor(),
        # Data normalization speeds up training & ResNet extension is optimized for normalized inputs
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

  # 4.2 Apply cutout on training set if cutout and its probability are specified in params
  # RandomErasing: https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandomErasing.html
  if split == 'train' and config.get('use_cutout', False):
    cutout_prob = config.get('cutout_prob', 0.5)
    transform_list.append(transforms.RandomErasing(p=cutout_prob))
  
  transform = transforms.Compose(transform_list)
    
  # 4.3 Instantiate data
  root_dir = config.get('data_dir', './data/miniimagenet') # In case root_dir is not defined in yaml
  dataset = MiniImageNet(
      root_dir=root_dir,
      split=split,
      transform=transform
    ) 

  if split == 'train':
    n_episodes = config['train_params']['n_episodes']
  elif split == 'val':
    n_episodes = config['val_params']['n_episodes']
  else:
    n_episodes = config['test_params']['n_episodes']

  # 4.4 Instantiate sampler
  sampler = PrototypicalBatchSampler(
      labels=dataset.labels,
      n_way=config['n_way'],
      n_shot=config['n_shot'],
      n_query=config['n_query'],
      n_episodes=n_episodes
    )

  # 4.5 Instantiate DataLoader
  dataloader = DataLoader(
      dataset=dataset,
      batch_sampler=sampler,
      num_workers=config.get('num_workers', 4),
      pin_memory=True
  )

  return dataloader