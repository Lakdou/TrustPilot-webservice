"""
Routes d'authentification : inscription, connexion, vérification quota (Nginx).
"""

from datetime import date
from fastapi import APIRouter, HTTPException, Header

from ..schemas.models import UserCreate, UserLogin
from ..services.users import get_users, save_users
from ..core.security import hash_password, generate_token
from ..core.config import DAILY_QUOTA

router = APIRouter(tags=["Auth"])


@router.post("/login")
def create_user(user: UserCreate):
    """Crée un nouveau compte utilisateur."""
    users = get_users()
    if user.username in users:
        raise HTTPException(status_code=400, detail="Cet utilisateur existe déjà.")
    users[user.username] = {
        "password": hash_password(user.password),
        "role":     user.role,
        "api_key":  None,
    }
    save_users(users)
    return {"message": f"Utilisateur '{user.username}' créé avec succès."}


@router.post("/token_API")
def generate_api_token(user: UserLogin):
    """Authentifie l'utilisateur et retourne un token API."""
    users = get_users()
    if (user.username not in users
            or users[user.username]["password"] != hash_password(user.password)):
        raise HTTPException(status_code=401, detail="Identifiants incorrects.")

    token = generate_token()
    users[user.username]["api_key"] = token
    save_users(users)

    return {
        "access_token": token,
        "role":         users[user.username]["role"],
        "username":     user.username,
    }


@router.get("/verify_admin")
def verify_admin(x_api_key: str = Header(None, alias="X-API-Key")):
    """
    Endpoint interne appelé par Nginx via auth_request.
    Vérifie la validité de la clé et le quota journalier.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Clé API manquante.")

    users = get_users()
    for _, user_data in users.items():
        if user_data.get("api_key") == x_api_key:
            if user_data.get("role") == "admin":
                return {"message": "Autorisé"}

            today      = date.today().isoformat()
            last_date  = user_data.get("last_request_date")
            count      = user_data.get("daily_count", 0) if last_date == today else 0

            if count >= DAILY_QUOTA:
                raise HTTPException(
                    status_code=403,
                    detail=f"Quota journalier atteint ({DAILY_QUOTA}/{DAILY_QUOTA}).",
                )
            return {"message": "Autorisé"}

    raise HTTPException(status_code=401, detail="Clé API invalide.")
