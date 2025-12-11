import gradio as gr
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from predict import FashionPredictor
    predictor = FashionPredictor()
    MODEL_LOADED = True
except Exception as e:
    print(f"Errore nel caricamento del modello: {e}")
    MODEL_LOADED = False

def predict_image(image):
    """
    Funzione per Gradio che prende un'immagine e restituisce la predizione
    """
    if not MODEL_LOADED:
        return " Modello non caricato. Prima addestra il modello con train.py", None
    
    if image is None:
        return "📸 Per favore carica o disegna un'immagine", None
    
    try:
        # Converti in numpy array
        if isinstance(image, dict):  # Se viene dal sketchpad
            image = image['image']
        
        # Converti in PIL Image se necessario
        if isinstance(image, np.ndarray):
            if image.shape[-1] == 4:  # RGBA
                image = image[..., :3]  # Rimuovi canale alpha
            if image.shape[-1] == 3:  # RGB
                image = np.mean(image, axis=-1)  # Converti in scala di grigi
        
        result = predictor.predict(image)
        
        # Crea grafico delle probabilità
        fig, ax = plt.subplots(figsize=(10, 5))
        classes = predictor.class_names
        probabilities = result['probabilities']
        
        colors = ['lightblue' for _ in classes]
        colors[result['class_id']] = 'red'
        
        bars = ax.bar(classes, probabilities, color=colors)
        ax.set_xlabel('Classi', fontsize=12)
        ax.set_ylabel('Probabilità', fontsize=12)
        ax.set_title(f'Predizione: {result["class"]} (ID: {result["class_id"]})', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Aggiungi valori sulle barre
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.2%}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        return f" **Predizione**: {result['class']} (ID: {result['class_id']})", fig
    
    except Exception as e:
        return f" Errore: {str(e)}", None

# Crea l'interfaccia
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(
        label="Carica un'immagine (28x28px, sfondo scuro)",
        type="numpy",
        height=300
    ),
    outputs=[
        gr.Markdown(label="Risultato"),
        gr.Plot(label="Probabilità per classe")
    ],
    title=" Fashion-MNIST Classifier",
    description="""
    ## Classificatore di Abbigliamento
    
    Carica un'immagine di un capo di abbigliamento (o disegnala) e il modello AI lo classificherà tra:
    
    **0:** T-shirt/top, **1:** Trouser, **2:** Pullover, **3:** Dress, **4:** Coat  
    **5:** Sandal, **6:** Shirt, **7:** Sneaker, **8:** Bag, **9:** Ankle boot
    
     Per risultati migliori: immagini 28x28 pixel con sfondo scuro.
    """,
    theme="soft",
    examples=[
        [np.random.randint(0, 255, (28, 28), dtype=np.uint8) for _ in range(2)]
    ]
)

if __name__ == "__main__":
    print(" Avvio interfaccia Gradio...")
    print(" Apri http://localhost:7860 nel tuo browser")
    interface.launch(server_name="0.0.0.0", server_port=7860)
