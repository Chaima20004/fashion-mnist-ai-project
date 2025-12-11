import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from model import FashionCNN

class FashionPredictor:
    def __init__(self, model_path='artifacts/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Carica modello
        self.model = FashionCNN(num_classes=10)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Trasformazioni
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    def predict(self, image):
        """
        Predice la classe di un'immagine
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        return {
            'class': self.class_names[predicted.item()],
            'class_id': predicted.item(),
            'probabilities': probabilities.squeeze().cpu().numpy().tolist()
        }
    
    def predict_batch(self, images):
        """
        Predice più immagini insieme
        """
        results = []
        for img in images:
            results.append(self.predict(img))
        return results

if __name__ == "__main__":
    # Esempio di utilizzo
    predictor = FashionPredictor()
    
    # Crea un'immagine casuale di test
    test_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    
    prediction = predictor.predict(test_image)
    print(f"Predizione: {prediction['class']}")
    print(f"Probabilità: {prediction['probabilities']}")
