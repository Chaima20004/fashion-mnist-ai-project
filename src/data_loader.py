import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

def get_fashion_mnist_loaders(batch_size=64, val_split=0.1):
    """
    Carica Fashion-MNIST e crea DataLoader per train, val, test
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Dataset completo
    full_train_dataset = datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Split train/val
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size]
    )
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def show_sample_images(loader, n_images=10):
    """
    Mostra immagini campione dal DataLoader
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    data_iter = iter(loader)
    images, labels = next(data_iter)
    
    fig, axes = plt.subplots(1, n_images, figsize=(15, 3))
    
    for i in range(n_images):
        img = images[i].numpy().squeeze()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(class_names[labels[i].item()])
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig
