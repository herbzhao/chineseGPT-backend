import asyncio
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from pydantic import BaseModel

import auth_endpoints
import chat_endpoints
from azure_synthesiser import AudioSynthesiser
from azure_transcriber import AudioTranscriber
from gpt_backends import chat
from parameters import (
    INITIAL_TIMEOUT_LENGTH,
    MP3_SENDING_CHUNK_SIZE,
    MP3_SENDING_TIMEOUT_LENGTH,
)


load_dotenv()

if os.path.exists(".env.production") and os.getenv("ENVIRONMENT") == "production":
    print("GETTING PRODUCTION ENVIRONMENT VARIABLES")
    load_dotenv(".env.production")
else:
    if os.path.exists(".env.local"):
        load_dotenv(".env.local")

app = FastAPI()
app.include_router(auth_endpoints.router)
app.include_router(chat_endpoints.router)

# set a default language on startup
# cors: https://fastapi.tiangolo.com/tutorial/cors/
frontend_url = []
frontend_url.append(os.getenv("FRONTEND_URL"))
frontend_url.append(os.getenv("FRONTEND_URL_NGROK"))
frontend_url.append(os.getenv("FRONTEND_URL_PRODUCTION"))

print("frontend_url: ", frontend_url)
origins = frontend_url

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# https://fastapi.tiangolo.com/advanced/events/
@app.on_event("startup")
async def startup_event():
    app.state.synthesiser = {}
    app.state.test_message = "test"
    # perform additional initialization tasks here


@app.get("/test")
def root():
    return {"msg": "fastapi is working"}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080, reload=False)
