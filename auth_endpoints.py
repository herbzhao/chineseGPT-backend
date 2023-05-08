import os
from datetime import datetime, timedelta
from typing import Optional

import jwt
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, Field
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from pymongo.server_api import ServerApi
from datetime import datetime, timezone
from mongo_access import (
    authenticate_user,
    create_access_token,
    decoding_token,
    get_password_hash,
    update_credits,
)

load_dotenv()
if os.path.exists(".env.production") and os.getenv("ENVIRONMENT") == "production":
    print("GETTING PRODUCTION ENVIRONMENT VARIABLES")
    load_dotenv(".env.production")
else:
    if os.path.exists(".env.local"):
        load_dotenv(".env.local")


router = APIRouter()


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


@router.post("/register/")
async def register(user: UserCreate, request: Request):
    try:
        hashed_password = get_password_hash(user.password)
        request.app.state.users_collection.insert_one(
            {"username": user.username, "password": hashed_password}
        )
        access_token = create_access_token(data={"sub": user.username})

        return {"access_token": access_token, "token_type": "bearer"}
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Username already exists")


@router.post("/login/", response_model=Token)
async def login(user: UserLogin, request: Request):
    db_user = authenticate_user(
        request.app.state.users_collection, user.username, user.password
    )
    if db_user is None:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


# FastAPI uses the oauth2_scheme dependency to extract the token from the Authorization header.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# extracted token is passed to the get_current_user function as the token argument.
async def get_current_user(
    token: str = Depends(oauth2_scheme), request: Request = None
):
    try:
        payload = decoding_token(token)
        # Check if the token has expired
        expiration = datetime.fromtimestamp(payload.get("exp"), timezone.utc)
        if expiration < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
            )

        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
            )

        user = request.app.state.users_collection.find_one({"username": username})
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user

    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )


@router.get("/validate_token/")
async def validate_token(current_user: dict = Depends(get_current_user)):
    return {"valid": True}


class UserCredits(BaseModel):
    credits: int


@router.get("/get_credits/", response_model=UserCredits)
async def get_credits(current_user: dict = Depends(get_current_user)):
    user_credits = current_user.get("credits", 0)
    return {"credits": user_credits}


class EditCreditsInput(BaseModel):
    credits_delta: int


@router.post("/edit_credits/", response_model=UserCredits)
async def edit_credits(
    input_data: EditCreditsInput,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    user_credits = current_user.get("credits", 0)
    print(input_data)
    user_credits += input_data.credits_delta
    request.app.state.users_collection.update_one(
        {"username": current_user["username"]},
        {"$set": {"credits": user_credits}},
    )

    return {"credits": user_credits}


@router.post("/save_history/")
async def save_history(
    history: dict,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    # print(history)
    request.app.state.users_collection.update_one(
        {"username": current_user["username"]},
        {"$set": {"history": history["history"]}},
    )
    return {"success": True}


@router.get("/load_history/")
async def load_history(
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    user_history = current_user.get("history", [])
    # print(user_history)
    return {"history": user_history}
