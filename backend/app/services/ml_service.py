"""
Chargement et utilisation du modèle LightGBM + TF-IDF.
"""

import joblib
from ..core.config import MODEL_PATH, VECTORIZER_PATH

LABELS = {0: "Négatif", 1: "Neutre", 2: "Positif"}

_model = None
_vectorizer = None


def get_model():
    """Charge le modèle une seule fois (singleton)."""
    global _model, _vectorizer
    if _model is None:
        try:
            _model      = joblib.load(MODEL_PATH)
            _vectorizer = joblib.load(VECTORIZER_PATH)
            print("✅ Modèle LightGBM et TF-IDF chargés.")
        except Exception as e:
            print(f"❌ Erreur chargement modèle : {e}")
    return _model, _vectorizer


def predict(text: str) -> dict:
    """Prédit le sentiment d'un texte. Retourne sentiment, confidence, class_id."""
    model, vectorizer = get_model()
    if model is None or vectorizer is None:
        raise RuntimeError("Modèle non disponible.")

    vec        = vectorizer.transform([text])
    class_id   = int(model.predict(vec)[0])
    confidence = 100.0

    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(vec)[0]
        confidence = round(float(max(proba)) * 100, 2)

    return {
        "sentiment":  LABELS.get(class_id, "Inconnu"),
        "confidence": confidence,
        "class_id":   class_id,
    }
