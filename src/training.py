import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

def train_model(model, train_loader, **kwargs):
    args = {
        'num_epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'criterion': nn.CrossEntropyLoss(),
        'optimizer': torch.optim.Adam(model.parameters(), lr=0.001),
        'val_loader': None,
    }
    args = {**args, **kwargs}
    device = args['device']
    num_epochs = args['num_epochs']
    criterion = args['criterion']
    optimizer = args['optimizer']
    val_loader = args['val_loader']    
    model.to(device)
    if val_loader is not None:
        best_val_accuracy = 0.0
        best_model = None
        best_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_waveforms, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
            batch_waveforms, batch_labels = batch_waveforms.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_waveforms)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        log_dict = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy
        }
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss on train set: {train_loss:.4f}, Accuracy on train set: {100 * train_accuracy:.2f}%')
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_waveforms, batch_labels in tqdm(val_loader, desc="Validation", unit="batch"):
                    batch_waveforms, batch_labels = batch_waveforms.to(device), batch_labels.to(device)
                    
                    outputs = model(batch_waveforms)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            val_loss /= len(val_loader)
            val_accuracy = val_correct / val_total
            log_dict.update({
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = model.state_dict().copy()
                best_epoch = epoch + 1
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss on validation set: {val_loss:.4f}, Accuracy on validation set: {100 * val_accuracy:.2f}%')
        wandb.log(log_dict)
    if val_loader is not None:
        model.load_state_dict(best_model)
        print(f"Training complete, best model found at epoch {best_epoch} with validation accuracy: {100 * best_val_accuracy:.2f}%")