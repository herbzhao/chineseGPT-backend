import os
from datetime import datetime, timedelta
from typing import Optional

from bson import ObjectId
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from pydantic import BaseModel, EmailStr, Field
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from pymongo.server_api import ServerApi

from mongo_access import (
    get_client,
    get_users_collection,
    verify_password,
    get_password_hash,
    decoding_token,
    authenticate_user,
    create_access_token,
)

load_dotenv()
if os.path.exists(".env.production") and os.getenv("ENVIRONMENT") == "production":
    print("GETTING PRODUCTION ENVIRONMENT VARIABLES")
    load_dotenv(".env.production")
else:
    if os.path.exists(".env.local"):
        load_dotenv(".env.local")


router = APIRouter()
mongo_client = get_client()
users_collection = get_users_collection(mongo_client)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class UserCreate(BaseModel):
    username: EmailStr
    password: str = Field(
        ...,
        min_length=8,
        max_length=20,
        regex="^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,20}$",
    )


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = decoding_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
            )
        user = users_collection.find_one({"username": username})
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )


@router.post("/users/")
async def create_user(user: UserCreate, request: Request):
    try:
        hashed_password = get_password_hash(user.password)
        users_collection.insert_one(
            {"username": user.username, "password": hashed_password}
        )
        access_token = create_access_token(data={"sub": user.username})

        return {"access_token": access_token, "token_type": "bearer"}
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Username already exists")


@router.post("/login/", response_model=Token)
async def login(user: UserLogin, request: Request):
    db_user = authenticate_user(user.username, user.password)
    if db_user is None:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}
