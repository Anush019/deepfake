import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data_loader import get_dataloader
from model import DeepFakeResNetLSTM
from sklearn.metrics import accuracy_score, f1_score
import json

with open("config.json", "r") as f:
    config = json.load(f)

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=5):
    writer = SummaryWriter("runs/deepfake_detection")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)            
            optimizer.zero_grad()            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        writer.add_scalar("training_loss", running_loss / len(train_loader), epoch)
        
        val_accuracy, val_f1 = evaluate_model(val_loader, model)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

    writer.close()
    return model

def evaluate_model(data_loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return accuracy, f1

def save_model(model, path="model.pth"):
    """Save the model to the specified path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

train_loader = get_dataloader(config["train_data_path"])
val_loader = get_dataloader(config["val_data_path"])
model = DeepFakeResNetLSTM()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

trained_model = train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=config["epochs"])

save_model(trained_model, path="pytorch_model.bin")
