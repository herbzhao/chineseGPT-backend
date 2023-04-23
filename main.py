from fastapi import FastAPI, WebSocket, Depends
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
from parameters import MP3_SENDING_CHUNK_SIZE
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
    synthesise_answer = False
    if synthesise_answer:
        audio_synthesiser = AudioSynthesiser()
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
        for response_chunk in response_generator:
            chunk_message = response_chunk["choices"][0]["delta"]
            # print(f"chunk_message: {chunk_message}")
            if "content" in chunk_message:
                await websocket.send_json({"content": chunk_message.content})
                complete_response += chunk_message.content
                if synthesise_answer:
                    synthesised_file = audio_synthesiser.synthesis_to_mp3(
                        complete_response
                    )
                    websocket.send_bytes(audio_synthesiser.mp3_chunks)
                await asyncio.sleep(0.01)

        await websocket.send_json({"content": "DONE"})


def generate_mp3_stream(file_path):
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(MP3_SENDING_CHUNK_SIZE)
            if not chunk:
                break
            yield chunk


# automatically serve the newly generated mp3 file
@app.get("/chat/stream/mp3http")
async def mp3_stream(session_id: str = Depends(get_session_id)):
    if session_id in app.state.generated_files:
        print("session id found")
        file_path = app.state.generated_files[session_id]
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        print("streaming mp3 file")
        return StreamingResponse(
            generate_mp3_stream(file_path), media_type="audio/mpeg"
        )
    await asyncio.sleep(0.1)


class TextToSpeech(BaseModel):
    text: str


async def text_to_speech(text: str, session_id: str):
    audio_synthesiser = AudioSynthesiser()
    audio_synthesiser.speech_synthesis_to_push_audio_output_stream(language="zh-CN")
    asyncio.create_task(audio_synthesiser.process_text())
    await audio_synthesiser.add_text(text)
    while True:
        if audio_synthesiser.synthesis_complete:
            print("SYNTHESIS COMPLETE")
            break
        await asyncio.sleep(0.1)
    audio_synthesiser.stream_callback.save_to_file()
    audio_synthesiser.close()

    return audio_synthesiser.output_filename


# receive text and save a mp3 file
@app.post("/chat/stream/text_to_speech")
async def text_to_speech_endpoint(
    text: TextToSpeech, session_id: str = Depends(get_session_id)
):
    text = text.text
    output_filename = await text_to_speech(text, session_id)
    app.state.generated_files[session_id] = output_filename
    return {
        "file_path": app.state.generated_files[session_id],
        "session_id": session_id,
    }


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
