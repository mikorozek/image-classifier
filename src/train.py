import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets import ProcessedImageDataset
from model import CustomCNN
import wandb

def train_model():
    processed_dir = './data/processed/'
    batch_size = 64
    num_epochs = 30
    learning_rate = 0.001
    validation_split = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wandb.init(
        project="image-classification-project",
        config={
            "architecture": "conv model training",
            "dataset": "custom",
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "validation_split": validation_split,
            "optimizer": "AdamW"
        }
    )
    
    full_dataset = ProcessedImageDataset(processed_dir)
    
    dataset_size = len(full_dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    del full_dataset
    del train_dataset
    del val_dataset
    
    model = CustomCNN() 
    
    model = model.to(device)
    wandb.watch(model, log="all")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        model.train()
        train_loss = 0.0
        train_corrects = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            
            print(f"Epoch {epoch+1}, Batch {i}/{len(train_loader)}, Train Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = float(train_corrects) / float(len(train_dataset))
        
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        all_preds = []
        all_labels = []
        
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = float(val_corrects) / float(len(val_dataset))
        
        print(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        log_dict = {
            "train_loss": epoch_train_loss,
            "train_accuracy": epoch_train_acc.item(),
            "val_loss": epoch_val_loss,
            "val_accuracy": epoch_val_acc.item(),
        }
        
        log_dict["confusion_matrix"] = wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=full_dataset.classes
        )
        
        wandb.log(log_dict)
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved new best model with accuracy: {best_val_acc:.4f}")
    wandb.finish()

if __name__ == "__main__":
    train_model()
