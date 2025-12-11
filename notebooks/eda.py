"""
EDA - Fashion-MNIST Dataset
Analisi Esplorativa dei Dati
Studente: Chaima Chourabi (VR510606)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
import torch

# Configurazione
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("ANALISI DATASET FASHION-MNIST")
print("="*60)

# Carica dataset
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Informazioni base
print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Image shape: {train_dataset[0][0].shape}")
print(f"Number of classes: {len(train_dataset.classes)}")

# Nomi delle classi
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Distribuzione delle classi
train_labels = [label for _, label in train_dataset]
test_labels = [label for _, label in test_dataset]

# Plot distribuzione classi
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Train
train_counts = pd.Series(train_labels).value_counts().sort_index()
axes[0].bar(range(10), train_counts.values, color='skyblue')
axes[0].set_title('Distribuzione Classi - Train Set', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Classe', fontsize=12)
axes[0].set_ylabel('Numero di Immagini', fontsize=12)
axes[0].set_xticks(range(10))
axes[0].set_xticklabels(range(10), rotation=0)
axes[0].grid(axis='y', alpha=0.3)

# Aggiungi valori sulle barre
for i, count in enumerate(train_counts.values):
    axes[0].text(i, count + 100, str(count), ha='center', va='bottom')

# Test
test_counts = pd.Series(test_labels).value_counts().sort_index()
axes[1].bar(range(10), test_counts.values, color='lightcoral')
axes[1].set_title('Distribuzione Classi - Test Set', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Classe', fontsize=12)
axes[1].set_ylabel('Numero di Immagini', fontsize=12)
axes[1].set_xticks(range(10))
axes[1].set_xticklabels(range(10), rotation=0)
axes[1].grid(axis='y', alpha=0.3)

# Aggiungi valori sulle barre
for i, count in enumerate(test_counts.values):
    axes[1].text(i, count + 50, str(count), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('../images/class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualizza immagini per ogni classe
fig, axes = plt.subplots(2, 5, figsize=(16, 8))
fig.suptitle('Esempi per Classe - Fashion-MNIST', fontsize=16, fontweight='bold')

for class_id in range(10):
    # Trova la prima immagine di questa classe
    idx = next(i for i, (_, label) in enumerate(train_dataset) if label == class_id)
    img, label = train_dataset[idx]
    
    row, col = class_id // 5, class_id % 5
    axes[row, col].imshow(img.squeeze(), cmap='gray')
    axes[row, col].set_title(f'Classe {class_id}: {class_names[class_id]}', fontsize=11)
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('../images/sample_per_class.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistiche
train_images = torch.stack([img for img, _ in train_dataset[:1000]])
mean = train_images.mean()
std = train_images.std()

print("\n📊 STATISTICHE IMMAGINI (sample di 1000):")
print("-"*40)
print(f"Valore medio pixel: {mean:.4f}")
print(f"Deviazione standard: {std:.4f}")
print("-"*40)

# Conclusione
print("\n" + "="*60)
print("CONCLUSIONI EDA - FASHION-MNIST")
print("="*60)
print("""
1.  DATASET COMPLETO E BILANCIATO
   • 70,000 immagini totali (60k train, 10k test)
   • 6,000 immagini per classe nel train set
   • 1,000 immagini per classe nel test set

2.  CARATTERISTICHE IMMAGINI
   • Dimensioni: 28×28 pixel
   • Canali: 1 (scala di grigi)
   • Valori pixel normalizzati

3.  10 CLASSI DI ABBIGLIAMENTO
   • Classi ben definite e distinguibili
   • Distribuzione uniforme

4.  IDONEITÀ PER IL PROGETTO
   • Dataset di dimensioni adeguate
   • Problema di classificazione chiaro
   • Integrazione semplice con PyTorch
""")
print("="*60)
