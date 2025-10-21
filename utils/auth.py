# utils/auth.py
from fastapi import HTTPException
import os

# Simple token verification for demo. Replace with real JWT/Supabase JWT verification in production.
VALID_ADMIN_TOKEN = os.getenv("ADMIN_ACCESS_TOKEN", None)

def verify_token(token: str):
    if not VALID_ADMIN_TOKEN:
        # if no admin token configured, accept but log warning (or raise for stricter behavior)
        return True
    if not token or token != VALID_ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return True
