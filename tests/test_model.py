import torch
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import FashionCNN, count_parameters
from src.data_loader import get_fashion_mnist_loaders

def test_model_creation():
    """Test che il modello viene creato correttamente"""
    model = FashionCNN(num_classes=10)
    assert model is not None
    assert isinstance(model, torch.nn.Module)
    
    # Verifica architettura
    assert hasattr(model, 'conv1')
    assert hasattr(model, 'conv2')
    assert hasattr(model, 'fc1')
    assert hasattr(model, 'fc2')
    assert hasattr(model, 'dropout')
    
    # Conta parametri
    params = count_parameters(model)
    assert params > 0
    print(f" Modello creato con {params:,} parametri")
    return True

def test_model_forward():
    """Test forward pass"""
    model = FashionCNN(num_classes=10)
    
    # Input dummy
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 28, 28)
    
    # Forward pass
    output = model(dummy_input)
    
    # Verifica dimensioni output
    assert output.shape == (batch_size, 10)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    print(f" Forward pass testato: output shape {output.shape}")
    return True

def test_data_loader():
    """Test che il data loader funzioni"""
    train_loader, val_loader, test_loader = get_fashion_mnist_loaders(
        batch_size=32, 
        val_split=0.1
    )
    
    # Verifica che i loader esistano
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None
    
    # Verifica batch
    for images, labels in train_loader:
        assert images.shape == (32, 1, 28, 28)
        assert labels.shape == (32,)
        break
    
    print(" DataLoader testato correttamente")
    return True

def test_model_parameters_update():
    """Test che i parametri si aggiornino durante il training"""
    model = FashionCNN(num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Salva parametri iniziali
    initial_params = []
    for param in model.parameters():
        initial_params.append(param.data.clone())
    
    # Un passo di training finto
    dummy_input = torch.randn(2, 1, 28, 28)
    dummy_labels = torch.randint(0, 10, (2,))
    
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, dummy_labels)
    loss.backward()
    optimizer.step()
    
    # Verifica che i parametri siano cambiati
    params_changed = False
    for initial, param in zip(initial_params, model.parameters()):
        if not torch.equal(initial, param.data):
            params_changed = True
            break
    
    assert params_changed, "I parametri dovrebbero essere cambiati dopo l'optimizer step"
    print(" Parametri si aggiornano correttamente durante il training")
    return True

def run_all_tests():
    """Esegue tutti i test"""
    print("\n" + "="*50)
    print(" ESECUZIONE TEST UNITARI")
    print("="*50)
    
    tests_passed = 0
    tests_failed = 0
    
    test_functions = [
        test_model_creation,
        test_model_forward,
        test_data_loader,
        test_model_parameters_update
    ]
    
    for test_func in test_functions:
        try:
            if test_func():
                tests_passed += 1
                print(f" {test_func.__name__}: PASSATO")
        except Exception as e:
            tests_failed += 1
            print(f" {test_func.__name__}: FALLITO - {str(e)}")
    
    print("\n" + "="*50)
    print(f" RISULTATI: {tests_passed} passati, {tests_failed} falliti")
    print("="*50)
    
    if tests_failed == 0:
        print(" TUTTI I TEST SONO PASSATI CON SUCCESSO!")
    else:
        print("  ALCUNI TEST SONO FALLITI")
    
    return tests_failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
