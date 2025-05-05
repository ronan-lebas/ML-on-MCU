import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

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
    
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for batch_waveforms, batch_labels in tqdm(test_loader, desc="Predicting", unit="batch"):
            batch_waveforms, batch_labels = batch_waveforms.to(device), batch_labels.to(device)
            
            outputs = model(batch_waveforms)
            loss = criterion(outputs, batch_labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()
            all_labels.extend(batch_labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    print(f'Loss on testing set: {test_loss/len(test_loader):.4f}, Accuracy on testing set: {100 * test_correct / test_total:.2f}%')
    class_report = classification_report(all_labels, all_predictions, target_names=test_loader.dataset.classes)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    return test_loss / len(test_loader), test_correct / test_total, class_report, conf_matrix