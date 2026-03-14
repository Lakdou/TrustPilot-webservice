"""
Sécurité : hachage des mots de passe, validation des clés API, contrôle d'accès.
"""

import hashlib
import secrets
from fastapi import HTTPException, Header, Security
from fastapi.security import APIKeyHeader

from ..services.users import get_users

api_key_header = APIKeyHeader(name="X-API-Key")


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def generate_token() -> str:
    return secrets.token_hex(32)


def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """Dépendance FastAPI — valide que la clé appartient à un utilisateur."""
    users = get_users()
    for user_data in users.values():
        if user_data.get("api_key") == api_key:
            return api_key
    raise HTTPException(status_code=403, detail="Accès refusé. Clé API invalide.")


def get_username_from_key(api_key: str) -> str | None:
    """Retourne le nom d'utilisateur associé à la clé, ou None."""
    users = get_users()
    for username, user_data in users.items():
        if user_data.get("api_key") == api_key:
            return username
    return None


def require_admin(x_api_key: str = Header(None, alias="X-API-Key")) -> str:
    """Dépendance FastAPI — lève 403 si l'utilisateur n'est pas admin."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Clé API manquante.")
    users = get_users()
    for username, user_data in users.items():
        if user_data.get("api_key") == x_api_key:
            if user_data.get("role") != "admin":
                raise HTTPException(status_code=403, detail="Accès admin requis.")
            return username
    raise HTTPException(status_code=401, detail="Clé API invalide.")
