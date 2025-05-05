import os
from typing import Tuple, List

import torch
from torch.utils.data import Dataset
import torchaudio


class SpeechCommandsDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir (str): Root directory of dataset, structured as root/class/*.wav
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.sampling_rate = 16000
        self.max_duration = 1.0

        # List all .wav files and associated class labels
        self.samples = []
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for filename in os.listdir(cls_dir):
                if filename.endswith('.wav'):
                    filepath = os.path.join(cls_dir, filename)
                    self.samples.append((filepath, self.class_to_idx[cls]))

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
