import torch
import torch.nn as nn

def train_model(model, dataloader, **kwargs):
    args = {
        'num_epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'criterion': nn.CrossEntropyLoss(),
        'optimizer': torch.optim.Adam(model.parameters(), lr=0.001),
    }
    args = {**args, **kwargs}
    device = args['device']
    num_epochs = args['num_epochs']
    criterion = args['criterion']
    optimizer = args['optimizer']
    
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_waveforms, batch_labels in dataloader:
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
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%')