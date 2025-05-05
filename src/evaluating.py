import torch
import torch.nn as nn
from tqdm import tqdm

def eval_model(model, test_loader, **kwargs):
    args = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'criterion': nn.CrossEntropyLoss(),
    }
    args = {**args, **kwargs}
    device = args['device']
    criterion = args['criterion']
    model.to(device)

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_waveforms, batch_labels in tqdm(test_loader, desc="Predicting", unit="batch"):
            batch_waveforms, batch_labels = batch_waveforms.to(device), batch_labels.to(device)
            
            outputs = model(batch_waveforms)
            loss = criterion(outputs, batch_labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()
    
    print(f'Loss on testing set: {test_loss/len(test_loader):.4f}, Accuracy on testing set: {100 * test_correct / test_total:.2f}%')
    return test_loss / len(test_loader), test_correct / test_total