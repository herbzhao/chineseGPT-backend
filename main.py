from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import time
from backend_functions import chat
from audio_processing import transcribing_chunks_async
from pydantic import BaseModel
import uvicorn
from azure_audio import AzureAudio

load_dotenv()
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
if os.path.exists(".env.production") and os.getenv("ENVIRONMENT") == "production":
    print("GETTING PRODUCTION ENVIRONMENT VARIABLES")
    load_dotenv(".env.production")

app = FastAPI()
# set a default language on startup
# cors: https://fastapi.tiangolo.com/tutorial/cors/
frontend_url = os.getenv("FRONTEND_URL")
origins = [frontend_url]
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
    app.language = "en"
    # perform additional initialization tasks here


@app.get("/test")
def root():
    return {"msg": "fastapi is working"}


# https://www.starlette.io/websockets/
@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    # example async generator to test websocket streaming
    async def example_generator(data):
        for i in range(10):
            yield f"{data} + {i}"
            time.sleep(0.1)

    await websocket.accept()
    data = await websocket.receive_json()
    print(f"Received data: {data}")
    # send generator data to client
    async for value in example_generator(data["message"]):
        await websocket.send_json({"data": value})
    await websocket.send_json({"data": "completed"})
    # await websocket.close(code=1000, reason=None)


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
    while True:
        prompt_request = await websocket.receive_json()
        prompt_request = PromptRequest(**prompt_request)
        prompt = prompt_request.prompt
        history = prompt_request.history
        print(f"Received prompt: {prompt}")
        print(f"Received history: {history}")
        print("in streaming mode")
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
        print("got response generator")
        for response_chunk in response_generator:
            chunk_message = response_chunk["choices"][0]["delta"]
            if "content" in chunk_message:
                await websocket.send_json({"content": chunk_message.content})
        await websocket.send_json({"content": "DONE"})


class AudioMetaData(BaseModel):
    language: str


# an endpoint to receive metadata from the client
@app.post("/chat/stream/audioMetadata")
def receive_metadata(audio_metadata: AudioMetaData):
    app.language = audio_metadata.language
    print(f"Changed the language to: {app.language}")
    return {"msg": "received metadata"}


# https://www.starlette.io/websockets/
@app.websocket("/chat/stream/azureTranscript")
async def azure_transcript_stream(websocket: WebSocket):
    await websocket.accept()
    voice_chunks = []
    transcribed_segment_length = 0
    printed_transcripts_number = 0
    transcripts = []
    azure_audio = AzureAudio()
    print("websocket connected")
    while True:
        voice_chunk = await websocket.receive_bytes()
        voice_chunks.append(voice_chunk)
        azure_audio.process_chunks(voice_chunks)


# https://www.starlette.io/websockets/
@app.websocket("/chat/stream/audioTranscript")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    voice_chunks = []
    transcribed_segment_length = 0
    printed_transcripts_number = 0
    transcripts = []
    print("websocket connected")
    while True:
        voice_chunk = await websocket.receive_bytes()
        voice_chunks.append(voice_chunk)
        # save chunk to a local file
        with open(f"resources/chunks/{len(voice_chunks)}.webm", "wb") as f:
            f.write(voice_chunk)

        # a function to handle gradually increasing chunks
        (
            transcripts,
            transcribed_segment_length,
            stop_transcribing,
        ) = await transcribing_chunks_async(
            voice_chunks,
            transcribed_segment_length,
            transcripts=transcripts,
            language=app.language,
        )
        if not stop_transcribing:
            # only print the latest transcript
            for transcript in transcripts[printed_transcripts_number:]:
                print(transcript)
                await websocket.send_json({"transcript": transcript + " "})

            printed_transcripts_number = len(transcripts)

        else:
            await websocket.send_json(
                {"transcript": transcripts[-1], "command": "DONE"}
            )
            print(transcripts[-1])
            print("disconnecting websocket...")
            # await websocket.close(code=1000, reason=None)
            voice_chunks = []
            transcribed_segment_length = 0
            stop_transcribing = False
            break


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080, reload=False)
