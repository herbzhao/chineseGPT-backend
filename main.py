from fastapi import FastAPI, WebSocket, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import time
from backend_functions import chat
from pydantic import BaseModel
import uvicorn
from azure_transcriber import AudioTranscriber
from azure_synthesiser import AudioSynthesiser
import asyncio
import json
from parameters import (
    MP3_SENDING_CHUNK_SIZE,
    MP3_SENDING_TIMEOUT_LENGTH,
    INITIAL_TIMEOUT_LENGTH,
)
from uuid import uuid4

load_dotenv()

if os.path.exists(".env.production") and os.getenv("ENVIRONMENT") == "production":
    print("GETTING PRODUCTION ENVIRONMENT VARIABLES")
    load_dotenv(".env.production")
else:
    if os.path.exists(".env.local"):
        load_dotenv(".env.local")

app = FastAPI()
# set a default language on startup
# cors: https://fastapi.tiangolo.com/tutorial/cors/
frontend_url = os.getenv("FRONTEND_URL")
print("frontend_url: ", frontend_url)
origins = [frontend_url, "https://gpt.tiandqian.com"]
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
    app.state.generated_files = {}
    app.state.synthesiser = {}
    # perform additional initialization tasks here


async def get_session_id(session_id: str = ""):
    if not session_id:
        session_id = str(uuid4())
    return session_id


@app.get("/test")
def root():
    return {"msg": "fastapi is working"}


class PromptRequest(BaseModel):
    prompt: str
    history: list[dict]


@app.post("/chat")
def send_response(prompt_request: PromptRequest) -> None:
    prompt = prompt_request.prompt
    history = prompt_request.history
    print(f"Received prompt: {prompt}")
    print(f"Received history: {history}")
    print("not in streaming mode")
    response_message = chat(
        prompt=prompt,
        history=history,
        actor="personal assistant",
        max_tokens=500,
        accuracy="medium",
        stream=False,
        session_id="test_api",
    )
    print(f"response_message: {response_message}")
    return {"content": response_message["content"], "author": "bot", "loading": False}


# https://www.starlette.io/websockets/
@app.websocket("/chat/stream")
async def chat_stream(websocket: WebSocket):
    await websocket.accept()
    complete_response = ""
    synthesise_answer = True

    while True:
        prompt_request = await websocket.receive_json()
        prompt_request = PromptRequest(**prompt_request)
        prompt = prompt_request.prompt
        history = prompt_request.history
        # print(f"Received prompt: {prompt}")
        # print(f"Received history: {history}")
        # get generator data to client
        response_generator = chat(
            prompt=prompt,
            history=history,
            actor="personal assistant",
            max_tokens=500,
            accuracy="medium",
            stream=True,
            session_id="test_api",
        )
        # print("got response generator")
        if synthesise_answer:
            session_id = str(uuid4())

            await websocket.send_json(
                {
                    "session_id": session_id,
                }
            )
            for response_chunk in response_generator:
                chunk_message = response_chunk["choices"][0]["delta"]
                try:
                    await text_to_speech(chunk_message.content, session_id)
                    await websocket.send_json({"content": chunk_message.content})
                except AttributeError:
                    pass
                await asyncio.sleep(0.05)
        else:
            for response_chunk in response_generator:
                chunk_message = response_chunk["choices"][0]["delta"]
                try:
                    await websocket.send_json({"content": chunk_message.content})
                except AttributeError:
                    pass
                await asyncio.sleep(0.01)

        await websocket.send_json({"command": "DONE"})


class TextToSpeech(BaseModel):
    text: str


async def text_to_speech(text: str, session_id: str):
    if session_id not in app.state.synthesiser:
        # if the session id is not in the dictionary, start the synthesiser
        app.state.synthesiser[session_id] = AudioSynthesiser()
        audio_synthesiser = app.state.synthesiser[session_id]
        audio_synthesiser.speech_synthesis_to_push_audio_output_stream(language="zh-CN")
        audio_synthesiser.session_id = session_id
        asyncio.create_task(audio_synthesiser.process_text())
    else:
        audio_synthesiser = app.state.synthesiser[session_id]

    # print(f"sending text to backend: {text}")
    await audio_synthesiser.add_text(text)


# receive text and save a mp3 file
@app.post("/chat/stream/text_to_speech")
async def text_to_speech_endpoint(
    background_tasks: BackgroundTasks,
    text: TextToSpeech,
):
    text = text.text
    # generate a random session id
    session_id = str(uuid4())

    async def background_task_to_add_text(text, session_id):
        for i in range(0, len(text), 5):
            await text_to_speech(text[i : i + 5], session_id)
            await asyncio.sleep(0.1)

    # mimic the behaviour of the text is being generated by the chatbot
    background_tasks.add_task(background_task_to_add_text, text, session_id)

    return {
        "session_id": session_id,
    }


async def generate_mp3_stream_async(file_path):
    current_position = 0
    previous_file_size = 0
    timeout = time.time() + INITIAL_TIMEOUT_LENGTH
    print("checking filepath" + file_path)
    while not os.path.exists(file_path):
        await asyncio.sleep(0.5)

    while True:
        file_size = os.path.getsize(file_path)
        print(file_size)
        if time.time() > timeout:
            print("timeout reached")
            break
        if file_size > current_position:
            with open(file_path, "rb") as f:
                f.seek(current_position)
                # read file chunk by chunk
                chunk = f.read(MP3_SENDING_CHUNK_SIZE)
                while chunk:
                    current_position = f.tell()
                    yield chunk
                    print(f"current position: {current_position}")
                    chunk = f.read(MP3_SENDING_CHUNK_SIZE)
                    timeout = time.time() + MP3_SENDING_TIMEOUT_LENGTH
                    await asyncio.sleep(0.01)

        await asyncio.sleep(0.5)


def generate_mp3_stream(file_path):
    current_position = 0
    previous_file_size = 0
    timeout = time.time() + INITIAL_TIMEOUT_LENGTH
    print("checking filepath" + file_path)
    while not os.path.exists(file_path):
        time.sleep(0.5)

    while True:
        file_size = os.path.getsize(file_path)
        if time.time() > timeout:
            print("timeout reached")
            break
        if file_size > current_position:
            with open(file_path, "rb") as f:
                f.seek(current_position)
                # read file chunk by chunk
                chunk = f.read(MP3_SENDING_CHUNK_SIZE)
                while chunk:
                    current_position = f.tell()
                    yield chunk
                    print(f"current position: {current_position}")
                    chunk = f.read(MP3_SENDING_CHUNK_SIZE)
                    timeout = time.time() + MP3_SENDING_TIMEOUT_LENGTH

        time.sleep(0.5)


async def check_new_data(file_path, previous_file_size):
    while True:
        file_size = os.path.getsize(file_path)
        if previous_file_size < file_size:
            print(f"file size increased from {previous_file_size} to {file_size}")
            return True
        await asyncio.sleep(0.1)


# automatically serve the newly generated mp3 file
@app.get("/chat/stream/mp3http")
async def mp3_stream(session_id: str = Depends(get_session_id)):
    if session_id in app.state.synthesiser:
        print("session id found")
        file_path = f"output/synthesized/audio_{session_id}.mp3"
        print(f"streaming file path: {file_path}")

        print("streaming mp3 file")
        return StreamingResponse(
            generate_mp3_stream(file_path), media_type="audio/mpeg"
        )


@app.websocket("/chat/stream/azureTranscript")
async def azure_transcript_stream(websocket: WebSocket):
    start_time = time.time()
    await websocket.accept()
    audio_transcriber = await AudioTranscriber.create()

    sent_transcripts = ""
    print("websocket connected")

    async def reset_transcriber():
        nonlocal sent_transcripts
        sent_transcripts = ""
        # Restart the speech recognizer
        await audio_transcriber.restart_speech_recognizer()

    # Start a background task to periodically check for new transcripts
    async def transcripts_handler():
        nonlocal sent_transcripts
        while True:
            transcripts = " ".join(audio_transcriber.transcripts)
            if transcripts != sent_transcripts:
                await websocket.send_json({"transcripts": transcripts})
                sent_transcripts = transcripts
                # print(f"{time.time() - start_time}: sent {sent_transcripts}")
            await asyncio.sleep(0.1)

    async def handle_message(message):
        if "bytes" in message.keys():
            voice_chunk = message["bytes"]
            # Call the add_chunk method with the received voice_chunk
            await audio_transcriber.add_chunk(voice_chunk)
            # print(f"{time.time() - start_time}: added chunk")
            if audio_transcriber.transcription_complete:
                await websocket.send_json({"command": "DONE"})

        elif "text" in message.keys():
            json_message = json.loads(message["text"])
            if "command" in json_message and json_message["command"] == "RESET":
                await reset_transcriber()
            elif "language" in json_message:
                audio_transcriber.language = json_message["language"]
                print(f"Changed the language to: {audio_transcriber.language}")
                await reset_transcriber()

    print("start processing chunks")
    # change this depending on whether encoding is mp3 or wav
    asyncio.create_task(audio_transcriber.process_chunks_wav())
    print("starting transcripts handler")
    asyncio.create_task(transcripts_handler())
    while True:
        try:
            message = await websocket.receive()
            await handle_message(message)
        except asyncio.TimeoutError:
            continue
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080, reload=False)
