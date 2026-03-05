import numpy as np
import torch
from torch.utils.data import Sampler

'''
protonet_sampler.py logic

Original (Snell): https://github.com/jakesnell/prototypical-networks/blob/master/protonets/data/base.py
Inpiration (Orobix):  https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/prototypical_batch_sampler.py

Key Modifications vs. Snell vs. Orobix
(1) Orobix vs. Snell: 
  - Snell's method puts the Sampler inside the Dataset class. Thus, the Dataset has to scan through labels every single step: O(N)
  - Orobix's method decouples the Sampler from the Dataset: The Dataset loads images. The Sampler uses a pre-calculated matrix of indices to tell Dataset
  which index unmbers to load: O(1).

(2) Our Approach vs. Orobix: 
  - Keep Orobix's matrix efficiency but change input arguments to be more intuitive
  n_way, n_shot, n_query, and n_episodes instead of classes_per_it and num_samples (= n_shot + n_query).
  - Replace 2D matrix lookup table of shape [num_classes, max_images] with dictionary lookup to
  [a] Eliminate padding problems for classes with different image counts.
  [b] Handle non-sequential class IDS (e.g. 10,20,30 instead of 0,1,2)
  - Add safety check to ensure the num_samples cannot be larger than number of images available
  per class.

'''

class PrototypicalBatchSampler(Sampler):
  '''
  PrototypicalBatchSampler: yield a batch of image indices per episode (iteration).

  Args:
    labels          : list of integers, representing class_id for every image
    n_way           : number of classes to sample per batch
    n_shot          : number of images to sample per class
    n_query         : number of images to use for query set
    num_episodes    : number of episodes to train on

  Yields:
    batch           : 1D tensor of length n_way * (n_shot + n_query)
  '''

  # ==================================================================
  # STEP 1: __init__
  # ==================================================================
  def __init__(self, labels, n_way, n_shot, n_query, n_episodes):
    '''
    Build a dictionary to look up image indices with key=class_id 
    and value=list of image indices
    '''
    super().__init__(None)
    self.labels = labels
    self.n_way = n_way
    self.n_shot = n_shot
    self.n_query = n_query
    self.n_episodes = n_episodes

    # 1. Total number of images needed per class
    self.n_samples = self.n_shot + self.n_query

    # 2. Find unique classes and how many times they appear
    self.classes, self.count = np.unique(self.labels, return_counts=True)
    self.classes = torch.tensor(self.classes)
    
    # 3. Safety check to ensure each class has enough images
    if min(self.count) < self.n_samples:
      raise ValueError(f'Not enough images per class! The smallest class only has {min(self.count)} images.')

    # 4. Build the dictionary lookup table
    labels_tensor = torch.tensor(self.labels)
    self.class_indices = {} # Initialize empty dictionary

    for c in self.classes:
      # Get index position where label == c
      self.class_indices[c.item()] = torch.where(labels_tensor == c)[0]

  # ==================================================================
  # STEP 2: __len__
  # ==================================================================
  def __len__(self):
    '''
    Return the number of episodes / iterations per epoch
    '''
    return self.n_episodes

  # ==================================================================
  # STEP 3: __iter__
  # ==================================================================
  def __iter__(self):
    for ep in range(self.n_episodes):
      batch = []

      # 1. Randomly select N-way classes from list of classes
      class_idx = torch.randperm(len(self.classes))[:self.n_way]
      chosen_classes = self.classes[class_idx]

      # 2. For each class, randomly pick n_samples images
      for c in chosen_classes:
        # Look up image idx in the dictionary
        img_idx = torch.randperm(len(self.class_indices[c.item()]))[:self.n_samples]
        selected_images = self.class_indices[c.item()][img_idx]
        # Append to the batch
        batch.append(selected_images)
      
      # Combine list of small tensors -> one long tensor
      yield torch.cat(batch)