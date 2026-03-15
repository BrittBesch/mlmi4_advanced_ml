import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MFCC

# Note: Ensure project root is in PYTHONPATH (e.g., run with PYTHONPATH=.)
from protonet_sampler import PrototypicalBatchSampler

'''
dataloader_speech.py logic

Replication of FS-KWS data pipeline (Parnami et al.).
1. Feature extraction:  40 MFCC features with a 40ms window and 20ms stride.
2. Grouping:            Isolates the 30 "Core" keywords.
3. Filtering:           Removes any audio clip shorter than 1 second (16,000 frames).
4. Balancing:           Caps every class at exactly 1062 samples to maintain class balance.
5. Splitting:           Disjoint split of the 30 classes into 20 (Train), 5 (Val), and 5 (Test).

'''

class SpeechCommandsFewShot(Dataset):
    # ==================================================================
    # STEP 1: DOWNLOAD / LOAD DATA & APPLY KWS PRE-PROCESSING
    # ==================================================================
    def __init__(self, root_dir, split='train', transform=None, download=True, seed=42):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # 1. Download raw Google Speech command dataset
        self.base_dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=root_dir,
            url='speech_commands_v0.02',
            folder_in_archive='SpeechCommands',
            download=download
        )
        
        # 2. Grouping: Define the 30 Core Keywords
        core_keywords = [
            'down', 'zero', 'seven', 'nine', 'five', 'yes', 'four', 'left', 'stop', 'six', 
            'right', 'on', 'three', 'off', 'dog', 'marvin', 'one', 'go', 'no', 'two', 
            'eight', 'house', 'wow', 'happy', 'bird', 'cat', 'up', 'sheila', 'bed', 'tree'
        ]
        
        # 3. Splitting: Sort and shuffle to ensure consistent, disjoint splits
        # In FS-learning, performance is highly sensitive to specific classes in the test set
        # Shuffling seed allow for running N experiments with N different keyword combinations for robust eval.
        core_keywords.sort()
        rng = random.Random(seed)
        rng.shuffle(core_keywords)
        
        if split == 'train':
            self.classes = core_keywords[:20]      # 20 classes for training
        elif split == 'val':
            self.classes = core_keywords[20:25]    # 5 classes for validation
        else: # test
            self.classes = core_keywords[25:]      # 5 classes for testing
        
        # Dictionary to temporarily hold valid index mappings for our specific split
        class_samples = {cls: [] for cls in self.classes}
        
        print(f"[{split.upper()}] Filtering Speech Commands...")
        
        # 4. Filtering: Scan the dataset to find valid 1-second clips
        for i in range(len(self.base_dataset)):
            # torchaudio returns: (waveform, sample_rate, label, speaker_id, utterance_number)
            waveform, sample_rate, label, _, _ = self.base_dataset[i]
            
            if label in self.class_to_idx:
                # Filter out utterances which are less than 1 second (i.e. 16000 frames at 16kHz)
                if waveform.shape[1] >= 16000:
                    class_samples[label].append(i)
        
        self.data_indices = []
        self.labels = []
        
        # 5. Balancing: Enforce exactly 1062 samples per core keyword
        target_samples = 1062
        
        # Safety check to ensure sufficient desired samples
        for cls in self.classes:
            indices = class_samples[cls]
            if len(indices) < target_samples:
                raise ValueError(f"Class {cls} only has {len(indices)} valid samples, need {target_samples}!")
            
            # Shuffle the available indices to randomize speakers, then cap at 1062
            rng.shuffle(indices)
            selected_indices = indices[:target_samples]
            
            self.data_indices.extend(selected_indices)
            # Map the string label to a continuous integer (0 to N-1) for the sampler
            self.labels.extend([self.class_to_idx[cls]] * target_samples)
    
    # ==================================================================
    # STEP 2: __len__
    # ==================================================================
    def __len__(self):
        """
        Returns the total balanced number of items per split
        """
        return len(self.data_indices)
    
    # ==================================================================
    # STEP 3: AUDIO FEATURE EXTRACTION
    # ==================================================================
    def __getitem__(self, idx):
        # Retrieve original torchaudio idx
        dataset_idx = self.data_indices[idx]
        waveform, _, _, _, _ = self.base_dataset[dataset_idx]
        
        # Truncate to exactly 16000 frames (in case some clips are slightly longer)
        waveform = waveform[:, :16000]
        
        # Apply the MFCC transform (Converting 1D Audio to 2D Tensor)
        if self.transform is not None:
            features = self.transform(waveform)
        else:
            features = waveform
            
        return features, self.labels[idx]
    
# ==================================================================
# STEP 4: INSTANTIATE DATALOADER
# ==================================================================
def get_dataloader(config, split):
    '''
    Create the Speech Commands DataLoader.
    
    Output shape: [channel, mfcc_bins, time_frames] -> [1,40,51] since time_frames = 16000/320+1
    
    '''
    # 1. Feature Extraction definition
    # 16kHz sample rate with a 40ms frame length and 20ms stride
    # Ref: https://github.com/ArchitParnami/Few-Shot-KWS/blob/master/protonets/data/FewShotSpeechData.py
    # 40ms * 16000 = 640 frame size (n_fft)
    # 20ms * 16000 = 320 stride size (hop_length)
    mfcc_transform = MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={
            'n_fft': 640,
            'hop_length': 320,
            'n_mels': 40 # number of frequency bands (n_mels must be >= n_mfcc)
        }
    )
    
    # 2. Instantiate filtered dataset
    root_dir = config.get('data_dir', './data/speech')
    dataset = SpeechCommandsFewShot(
        root_dir=root_dir, 
        split=split, 
        transform=mfcc_transform
    )
    
    # 3. Handle episode counts based on split
    if split == 'train':
        params = config['train_params']
    elif split == 'val':
        params = config.get('val_params', config['test_params'])
    else:
        params = config['test_params']
        
    # 4. Instantiate sampler
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