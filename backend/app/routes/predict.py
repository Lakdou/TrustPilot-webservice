"""
Route de prédiction de sentiment.
"""

from datetime import date
from fastapi import APIRouter, HTTPException, Depends

from ..schemas.models import Review
from ..services import ml_service, monitor_service
from ..services.users import get_users, save_users
from ..core.security import get_api_key

router = APIRouter(tags=["Prediction"])


@router.post("/predict")
def predict_sentiment(review: Review, api_key: str = Depends(get_api_key)):
    """Prédit le sentiment d'un texte et consomme 1 crédit quota."""
    try:
        result = ml_service.predict(review.text)
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Modèle non chargé.")

    # Mise à jour du quota + récupération du username
    users    = get_users()
    username = "unknown"
    for uname, user_data in users.items():
        if user_data.get("api_key") == api_key:
            username = uname
            if user_data.get("role") != "admin":
                today = date.today().isoformat()
                if user_data.get("last_request_date") != today:
                    user_data["daily_count"]       = 0
                    user_data["last_request_date"] = today
                user_data["daily_count"] = user_data.get("daily_count", 0) + 1
                save_users(users)
            break

    # Log silencieux pour le monitoring
    try:
        monitor_service.log_prediction(
            username   = username,
            text       = review.text,
            prediction = result["sentiment"],
            confidence = result["confidence"] / 100.0,
            class_id   = result["class_id"],
        )
    except Exception:
        pass

    return {
        "texte":            review.text,
        "sentiment":        result["sentiment"],
        "prediction_score": f"{result['confidence']}%",
        "class_id":         result["class_id"],
    }
