# Dockerfile per addestramento modello Fashion-MNIST
FROM python:3.9-slim

# Imposta directory di lavoro
WORKDIR /app

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements e installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il codice sorgente
COPY src/ ./src/
COPY tests/ ./tests/
COPY notebooks/ ./notebooks/
COPY train_config.yaml .

# Crea directory necessarie
RUN mkdir -p data artifacts

# Comando di default: addestramento
CMD ["python", "src/train.py"]
