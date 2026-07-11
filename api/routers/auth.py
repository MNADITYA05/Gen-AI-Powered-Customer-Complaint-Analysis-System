"""
Authentication endpoints.

POST /auth/register  — create a new user (admin only after first user)
POST /auth/login     — obtain a JWT token
GET  /auth/me        — return the current user's profile
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas.auth import Token, UserCreate, UserLogin, UserResponse
from core.auth import create_access_token, get_password_hash, verify_password
from core.database import get_db
from core.db_models import UserDoc, new_user
from api.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=Token)
def login(credentials: UserLogin, db=Depends(get_db)):
    """Exchange username + password for a JWT access token."""
    doc = db.users.find_one({"username": credentials.username})
    if not doc or not verify_password(credentials.password, doc["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    if not doc.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is disabled")

    token = create_access_token({"sub": doc["_id"], "role": doc["role"]})
    return Token(access_token=token)


@router.post("/register", response_model=UserResponse, status_code=201)
def register(payload: UserCreate, db=Depends(get_db)):
    """
    Register a new user.
    The first registered user is automatically admin.
    After that, only existing admins can create admin accounts
    (enforced at the client layer — open registration only creates agents).
    """
    if db.users.find_one({"username": payload.username}):
        raise HTTPException(status_code=409, detail="Username already taken")
    if db.users.find_one({"email": payload.email}):
        raise HTTPException(status_code=409, detail="Email already registered")

    # First ever user gets admin regardless of requested role
    is_first_user = db.users.count_documents({}) == 0
    role = "admin" if is_first_user else "agent"

    user_doc = new_user(
        username=payload.username,
        email=payload.email,
        hashed_password=get_password_hash(payload.password),
        role=role,
    )
    db.users.insert_one(user_doc)
    logger.info("Registered user '%s' with role '%s'", user_doc["username"], user_doc["role"])
    return UserResponse(**UserDoc(user_doc).to_dict())


@router.get("/me", response_model=UserResponse)
def me(current_user: UserDoc = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return UserResponse(**current_user.to_dict())
