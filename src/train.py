import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import yaml
import os
from tqdm import tqdm

from model import FashionCNN
from data_loader import get_fashion_mnist_loaders

def train_epoch(model, loader, criterion, optimizer, device):
    """
    Addestra per un'epoca
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """
    Valuta il modello
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

def main(config_path='train_config.yaml'):
    # Carica configurazione
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # DataLoader
    train_loader, val_loader, _ = get_fashion_mnist_loaders(
        batch_size=config['batch_size'],
        val_split=config['val_split']
    )
    
    # Modello
    model = FashionCNN(num_classes=10).to(device)
    
    # Loss e Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Addestramento
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("Inizio addestramento...")
    for epoch in range(config['epochs']):
        print(f"\nEpoca {epoch+1}/{config['epochs']}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Salva storia
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Salva il miglior modello
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('artifacts', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, 'artifacts/best_model.pth')
            print(f"Modello salvato con accuracy: {val_acc:.2f}%")
    
    # Salva ultimo modello
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': config
    }, 'artifacts/last_model.pth')
    
    # Salva storia
    np.save('artifacts/training_history.npy', history)
    
    print(f"\nAddestramento completato! Miglior accuracy: {best_val_acc:.2f}%")
    return history

if __name__ == "__main__":
    main()
