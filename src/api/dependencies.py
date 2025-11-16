"""API dependencies for authentication and authorization."""

import os
import secrets
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..config import get_settings


security = HTTPBearer()


def get_api_key() -> str:
    """Get API key from environment variable or generate default."""
    settings = get_settings()
    api_key = settings.FRAUD_DETECTION_API_KEY
    
    if api_key is None:
        # Generate a random key for development
        api_key = secrets.token_urlsafe(32)
        print(f"WARNING: Using randomly generated API key: {api_key}")
        print("Set FRAUD_DETECTION_API_KEY environment variable for production use.")
    
    return api_key


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key from Bearer token."""
    api_key = get_api_key()
    
    if credentials.credentials != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials



