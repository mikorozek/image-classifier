import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, ImageFolder
import torchvision.transforms as transforms
from model import CustomCNN
import csv

def save_prediction(model_path='models/best_model.pth', test_dir='data/test_all', output_file='pred.csv'):
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = CustomCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(test_loader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                start_idx = i * batch_size
                for i, pred in enumerate(preds):
                    if start_idx + i < len(test_dataset.samples):
                        file_path = test_dataset.samples[start_idx + i][0]
                        file_name = os.path.basename(file_path)
                        class_idx = pred.item()
                        
                        csv_writer.writerow([file_name, class_idx])
    

if __name__ == "__main__":
    save_prediction()
