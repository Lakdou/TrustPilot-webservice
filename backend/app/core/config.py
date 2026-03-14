"""
Configuration centralisée — chemins et constantes de l'application.
Les chemins sont basés sur APP_BASE_DIR (défaut : /app dans Docker).
"""

import os
from pathlib import Path

BASE_DIR   = Path(os.getenv("APP_BASE_DIR", "/app"))
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Fichiers de données
USERS_FILE      = str(DATA_DIR / "users.json")
PREDICTIONS_LOG = str(DATA_DIR / "predictions_log.jsonl")

# Modèles ML
MODEL_PATH      = str(MODELS_DIR / "trustpilot_lgbm_model.pkl")
VECTORIZER_PATH = str(MODELS_DIR / "tfidf_vectorizer.pkl")

# Règles métier
DAILY_QUOTA = 5
