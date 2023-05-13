import os
from datetime import datetime, timedelta
from typing import Optional

import jwt
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, Field
from pymongo import DESCENDING
from pymongo.errors import DuplicateKeyError
from datetime import datetime, timezone
from mongo_access import (
    authenticate_user,
    create_access_token,
    decoding_token,
    get_password_hash,
    update_credits,
)
from gpt_backends import chat, calculate_token_number


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
    # return username if token is valid
    return {"username": current_user["username"]}


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
    data: dict,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    unique_id = (
        current_user["username"] + "_" + data["history"][0]["time"].replace(".", "_")
    )
    # Convert the creation time to a datetime object
    creation_time = datetime.strptime(
        data["history"][0]["time"], "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    # use gpt3 to summarise the history into one sentence
    summary = ""
    new_history = {
        "username": current_user["username"],
        "messages": data["history"],
        "creation_time": creation_time,  # new field
        "last_updated": datetime.now(),
        "uid": unique_id,  # new field
        "summary": summary,
    }

    # Retrieve the existing history, if any
    existing_history = request.app.state.histories_collection.find_one(
        {"uid": unique_id}
    )
    # If there is no existing history or the messages have changed, update the history
    if (
        existing_history is None
        or existing_history["messages"] != new_history["messages"]
    ):
        request.app.state.histories_collection.update_one(
            {"uid": unique_id},
            {"$set": new_history},
            upsert=True,
        )

    return {"success": True}


@router.get("/load_history_messages/")
async def load_history(
    uid: Optional[str] = None,  # new parameter for unique ID
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    if uid:
        # If a unique ID is provided, return the specific chat history
        history = request.app.state.histories_collection.find_one(
            {"uid": uid}, {"_id": 0}
        )
    else:
        # Get the unique ID of the most recently updated chat history
        most_recent_chat = request.app.state.histories_collection.find_one(
            {"username": current_user["username"]},
            {"_id": 0},
            sort=[("last_updated", DESCENDING)],
        )
        history = most_recent_chat if most_recent_chat else None

    if history is not None:
        return {"messages": history["messages"]}
    else:
        raise {"messages": []}


# return the summary of the histories for retrieval by the uid later using the load_history endpoint
@router.get("/load_histories/")
async def load_histories(
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    histories = list(
        request.app.state.histories_collection.find(
            {"username": current_user["username"]},
            {"_id": 0, "uid": 1, "last_updated": 1, "summary": 1},
        ).sort("last_updated", DESCENDING)
    )

    if not histories:
        histories = []
    return {"histories": histories}


@router.delete("/delete_all_histories/")
async def delete_all_histories(
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    result = request.app.state.histories_collection.delete_many(
        {"username": current_user["username"]}
    )
    return {"deleted_count": result.deleted_count}


@router.delete("/delete_history/")
async def delete_history(
    uid: str,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    result = request.app.state.histories_collection.delete_one({"uid": uid})
    return {"deleted_count": result.deleted_count}
