import os
from datetime import datetime, timedelta
from typing import Optional

import jwt
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from pymongo.server_api import ServerApi

from mongo_access import get_client, get_users_collection
from parameters import ACCESS_TOKEN_EXPIRE_MINUTES, ENCODING_ALGORITHM

load_dotenv()
if os.path.exists(".env.production") and os.getenv("ENVIRONMENT") == "production":
    print("GETTING PRODUCTION ENVIRONMENT VARIABLES")
    load_dotenv(".env.production")
else:
    if os.path.exists(".env.local"):
        load_dotenv(".env.local")

SECRET_KEY = os.getenv("SECRET_KEY")


router = APIRouter()
mongo_client = get_client()
users_collection = get_users_collection(mongo_client)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
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


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ENCODING_ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ENCODING_ALGORITHM])
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


def authenticate_user(username: str, password: str):
    user = users_collection.find_one({"username": username})
    if not user:
        return None
    if not verify_password(password, user["password"]):
        return None
    return user


def generate_access_token(username):
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )
    return access_token


@router.post("/users/")
async def create_user(user: UserCreate, request: Request):
    try:
        hashed_password = get_password_hash(user.password)
        users_collection.insert_one(
            {"username": user.username, "password": hashed_password}
        )
        access_token = generate_access_token(user.username)
        return {"access_token": access_token, "token_type": "bearer"}
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Username already exists")


@router.post("/login/", response_model=Token)
async def login(user: UserLogin, request: Request):
    db_user = authenticate_user(user.username, user.password)
    if db_user is None:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = generate_access_token(user.username)
    return {"access_token": access_token, "token_type": "bearer"}
