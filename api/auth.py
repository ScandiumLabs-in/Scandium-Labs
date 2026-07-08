import os
import sys
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()

_DEV_KEY = "dev-secret-key-not-for-production"
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", _DEV_KEY)
ALGORITHM = "HS256"

if SECRET_KEY == _DEV_KEY and os.environ.get("SCANDIUM_ENV", "development") != "development":
    print("FATAL: JWT_SECRET_KEY must be set in production mode", flush=True)
    sys.exit(1)


def create_access_token(user_id: str, expires_delta: timedelta = timedelta(days=7)):
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "exp": now + expires_delta,
        "iat": now,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")
