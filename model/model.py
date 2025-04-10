import torch
import torch.nn as nn
from torchvision import models

class DeepFakeResNetLSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden_size=128):
        super(DeepFakeResNetLSTM, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity() 
        
        self.lstm = nn.LSTM(input_size=2048, hidden_size=lstm_hidden_size, batch_first=True)        
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
    
    def forward(self, x):
        batch_size, _, _, _ = x.size()        
        features = self.resnet(x)        
        features = features.unsqueeze(1)        
        lstm_out, _ = self.lstm(features)        
        out = self.fc(lstm_out[:, -1, :]) 
        return out
