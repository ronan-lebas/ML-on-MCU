import os
from typing import Tuple, List

import torch
from torch.utils.data import Dataset
import torchaudio


class SpeechCommandsDataset(Dataset):
    def __init__(self, root_dir: str, split="TRAIN", sampling_rate=16000, transform=None, max_sample_per_class=-1, only_classes=None):
        """
        Args:
            root_dir (str): Root directory of dataset, structured as root/class/*.wav
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.sampling_rate = sampling_rate
        self.max_duration = 1.0

        with open(os.path.join(root_dir, "testing_list.txt"), "r") as f:
            test_samples = set(line.strip() for line in f)
        with open(os.path.join(root_dir, "validation_list.txt"), "r") as f:
            val_samples = set(line.strip() for line in f)

        # List all .wav files and associated class labels
        self.samples = []
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls in self.classes:
            if only_classes is not None and cls not in only_classes:
                continue
            cls_dir = os.path.join(root_dir, cls)
            sample_counter = 0
            for filename in os.listdir(cls_dir):
                if not filename.endswith('.wav'):
                    continue
                filepath = os.path.join(cls, filename)
                if (self.split == "TEST" and filepath in test_samples) or \
                        (self.split == "VAL" and filepath in val_samples) or \
                        (self.split == "TRAIN" and filepath not in test_samples and filepath not in val_samples):
                    self.samples.append((os.path.join(root_dir, filepath), self.class_to_idx[cls]))
                    sample_counter += 1
                    if sample_counter >= max_sample_per_class > 0:
                        break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.samples[idx]

        waveform, sample_rate = torchaudio.load(filepath)
        if sample_rate != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sampling_rate)(waveform)
        # Pad or truncate waveform to have a fixed length
        target_length = int(self.max_duration * self.sampling_rate)
        if waveform.size(1) < target_length:
            padding = target_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :target_length]

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label
