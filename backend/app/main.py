"""
Point d'entrée FastAPI — assemble toutes les routes.
"""

from fastapi import FastAPI
from .routes import auth, predict, monitoring
from .services.ml_service import get_model

app = FastAPI(
    title="Trustpilot Sentiment API",
    description="API sécurisée · prédiction de sentiment · monitoring",
    version="4.0",
)

# Pré-chargement du modèle au démarrage
@app.on_event("startup")
def startup():
    get_model()


# Routes
app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(monitoring.router)


@app.get("/", tags=["Health"])
def root():
    return {"message": "Trustpilot Sentiment API v4.0 — /docs pour tester."}
