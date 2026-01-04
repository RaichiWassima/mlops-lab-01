FROM python:3.10-slim

# 1) Préparer le dossier de travail dans le conteneur
WORKDIR /app

# 2) Copier les dépendances et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copier le reste du projet dans l'image
COPY . .

# 4) Exposer le port de l'API (8000 dans notre lab)
EXPOSE 8000

# 5) Commande de lancement de l'API FastAPI
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
