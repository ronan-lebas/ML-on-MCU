import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio.transforms as T
from dataloader import SpeechCommandsDataset
from model import AudioCNN
from training import train_model
from evaluating import eval_model
from dotenv import load_dotenv
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt


load_dotenv()
dataset_path = os.getenv("DATASET_PATH")

config = {
    'num_epochs': 50,
    'batch_size': 64,
    'max_sample_per_class': 1000,
    'only_classes': ["yes", "no", "on", "off"],
    'sampling_rate': 8000
}

using_wandb = True
if using_wandb:
    import wandb
    wandb.init(
        project="ML-on-MCU",
        entity="ronan-lebas-projects",
        config=config,
    )

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

transform = T.MFCC(
    sample_rate=config['sampling_rate'],
    n_mfcc=40,
    melkwargs={
        "n_fft": 400,
        "hop_length": 160,
        "n_mels": 64
    }
)
print("Loading Train dataset...")
train_loader = DataLoader(
    SpeechCommandsDataset(
        dataset_path,
        "TRAIN",
        sampling_rate=config['sampling_rate'],
        transform=transform,
        max_sample_per_class=config['max_sample_per_class'],
        only_classes=config['only_classes']
    ),
    batch_size=config['batch_size'],
    shuffle=True
)
print("Done loading Train dataset")
print("Loading Validation dataset...")

val_loader = DataLoader(
    SpeechCommandsDataset(
        dataset_path,
        "VAL",
        sampling_rate=config['sampling_rate'],
        transform=transform,
        max_sample_per_class=config['max_sample_per_class'],
        only_classes=config['only_classes']
    ),
    batch_size=config['batch_size'],
    shuffle=False
)
print("Done loading Validation dataset")
print("Loading Test dataset...")

test_loader = DataLoader(
    SpeechCommandsDataset(
        dataset_path,
        "TEST",
        sampling_rate=config['sampling_rate'],
        transform=transform,
        max_sample_per_class=config['max_sample_per_class'],
        only_classes=config['only_classes']
    ),
    batch_size=config['batch_size'],
    shuffle=False
)
print("Done loading Test dataset")
print("Loading Model...")

model = AudioCNN(num_classes=len(train_loader.dataset.classes), in_channels=1)
print("Done loading Model")
print("Training Model...")

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    num_epochs=config['num_epochs'],
    
)
print("Done Training Model")

print("Evaluating Model...")
_, _, class_report, conf_matrix = eval_model(
    model=model,
    test_loader=test_loader,
    device=device
)
print("Done Evaluating Model")

print("Saving Model and Reports...")
os.makedirs("reports", exist_ok=True)
timestamp = time.strftime("%Y%m%d-%H%M%S")
os.makedirs(f"reports/{timestamp}", exist_ok=True)
with open(f"reports/{timestamp}/class_report.txt", "w") as f:
    f.write(class_report)
model_path = f"reports/{timestamp}/model.pth"
torch.save(model.state_dict(), model_path)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=test_loader.dataset.classes, yticklabels=test_loader.dataset.classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"reports/{timestamp}/conf_matrix.png")
plt.close()
if using_wandb:
    wandb.log({"conf_matrix": wandb.Image(f"reports/{timestamp}/conf_matrix.png")})
    wandb.save(model_path)
    wandb.finish()
print("Done Saving Model and Reports")
print("All done!")