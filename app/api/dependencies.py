from core.config import settings
from fastapi import Header, HTTPException


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in settings.api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def verify_admin_key(x_admin_key: str = Header(...)):
    if x_admin_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return x_admin_key
