import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pathlib

class DeepFakeDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.image_paths = list(pathlib.Path(dataset_path).glob('*/*/*.*'))
        self.transform = transform
        self.labels = [0 if 'real' in str(path.parent) else 1 for path in self.image_paths]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataloader(dataset_path, batch_size=32, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = DeepFakeDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader
