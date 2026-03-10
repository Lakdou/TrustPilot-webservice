import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import joblib

# 🤫 Charge les variables d'environnement (pour le local et le cloud)
load_dotenv()

app = FastAPI(
    title="Trustpilot Sentiment API (LightGBM)",
    description="API sécurisée pour analyser le sentiment des avis clients en multiclasse",
    version="3.0"
)

# 🛡️ 1. Configuration de la sécurité
API_KEY = os.getenv("API_KEY") 
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Accès refusé. Clé API invalide.")

# 🧠 2. Chargement du TF-IDF et du Modèle
try:
    chargement = joblib.load("model_robust.pkl")
    if isinstance(chargement, dict):
        tfidf = chargement.get("tfidf")
        model = chargement.get("model")
        print("📦 Modèle et TF-IDF chargés avec succès !")
    else:
        model = chargement
        tfidf = None
except Exception as e:
    model = None
    tfidf = None
    print(f"❌ Erreur lors du chargement : {e}")

class Review(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API MLOps sécurisée. Allez sur /docs pour tester."}

# 🔒 3. Route de prédiction
@app.post("/predict")
def predict_sentiment(review: Review, api_key: str = Depends(get_api_key)):
    if model is None or tfidf is None:
        raise HTTPException(status_code=500, detail="Le modèle ou le TF-IDF n'est pas chargé.")
    
    # ÉTAPE A : TF-IDF (On récupère les 500 colonnes)
    texte_vectorise = tfidf.transform([review.text])
    
    # ÉTAPE B : VADER (On calcule la 501ème colonne)
    vader_score = sia.polarity_scores(review.text)['compound']
    vader_feature = np.array([[vader_score]])
    
    # ÉTAPE C : On colle les deux ensemble (500 + 1 = 501 colonnes)
    features_finales = hstack([texte_vectorise, vader_feature])
    
    # ÉTAPE D : On fait la prédiction avec le Modèle
    prediction = model.predict(features_finales)[0]
    
    # ÉTAPE C : Calcul de la confiance (probabilité)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texte_vectorise)[0]
        confidence = round(max(proba) * 100, 2)
    else:
        confidence = 100.0

    # ÉTAPE D : Mapping des 3 classes
    labels = {
        0: "Négatif",
        1: "Neutre",
        2: "Positif"
    }
    sentiment = labels.get(int(prediction), "Inconnu")
    
    return {
        "texte": review.text,
        "sentiment": sentiment,
        "prediction_score": f"{confidence}%",
        "class_id": int(prediction)
    }