import torch
from torch.utils.data import DataLoader
from dataloader import SpeechCommandsDataset
from model import AudioCNN
from training import train_model
from dotenv import load_dotenv
import os

load_dotenv()
dataset_path = os.getenv("DATASET_PATH")
dataset = SpeechCommandsDataset(dataset_path)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Iterate through one batch
for batch_waveforms, batch_labels in dataloader:
    print(batch_waveforms.shape)  # [B, 1, N]
    print(batch_labels)
    break

model = AudioCNN(num_classes=len(dataset.classes), in_channels=1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_model(
    model=model,
    dataloader=dataloader,
    device=device
)