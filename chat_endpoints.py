import asyncio
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import (APIRouter, BackgroundTasks, Depends, Request, WebSocket,
                     websockets)
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from azure_synthesiser import AudioSynthesiser
from azure_transcriber import AudioTranscriber
from gpt_backends import calculate_token_number, chat

router = APIRouter()
synthesisers = {}


async def get_session_id(session_id: str = ""):
    if not session_id:
        session_id = str(uuid4())
    return session_id


class PromptRequest(BaseModel):
    prompt: str
    history: list[dict]
    synthesise_switch: bool
    gpt4_switch: bool


async def text_to_speech(text: str, session_id: str, audio_language: str = "zh-CN"):
    if session_id not in synthesisers:
        # if the session id is not in the dictionary, start the synthesiser
        synthesisers[session_id] = AudioSynthesiser()
        synthesisers[session_id].language = audio_language
        synthesisers[session_id].start_speech_synthesis_using_push_stream()
        synthesisers[session_id].session_id = session_id
        asyncio.create_task(synthesisers[session_id].process_text())

    # print(f"sending text to backend: {text}")
    await synthesisers[session_id].add_text(text)


@router.websocket("/data_stream")
async def data_stream(websocket: WebSocket):
    async def check_synthesised_mp3_ready():
        nonlocal session_id
        if session_id in synthesisers:
            if (
                hasattr(synthesisers[session_id], "audio_ready")
                and synthesisers[session_id].audio_ready
            ):
                await websocket.send_json({"mp3_ready": session_id})
                synthesisers[session_id].audio_ready = False

    # Start a background task to periodically check for new transcripts
    async def prompt_handler(json_message):
        nonlocal session_id, audio_language
        prompt = json_message["prompt"]
        history = json_message["history"]
        synthesise_switch = json_message["synthesise_switch"]
        sleep_length = 0.05 if synthesise_switch else 0.01
        gpt4_switch = json_message["gpt4_switch"]
        model = "gpt-4" if gpt4_switch else "gpt-3.5-turbo"
        session_id = str(uuid4())

        response_generator, prompt_token_number = chat(
            prompt=prompt,
            history=history,
            actor="personal assistant",
            max_tokens=500,
            accuracy="medium",
            stream=True,
            model=model,
            session_id=session_id,
        )
        response = ""
        await websocket.send_json(
            {
                "session_id": session_id,
            }
        )
        for response_chunk in response_generator:
            try:
                # wait for "stop" command or otherwise keep sending response
                incoming_data = await asyncio.wait_for(
                    websocket.receive_json(), timeout=sleep_length
                )
                if incoming_data.get("command") == "stop":
                    break
            except asyncio.TimeoutError:
                chunk_message = response_chunk["choices"][0]["delta"]
                if hasattr(chunk_message, "content"):
                    if synthesise_switch:
                        await text_to_speech(
                            chunk_message.content, session_id, audio_language
                        )
                        await check_synthesised_mp3_ready()
                    await websocket.send_json({"content": chunk_message.content})
                    response += chunk_message.content

            response_token_number = calculate_token_number(
                [{"role": "assisstant", "content": response}]
            )
            if gpt4_switch:
                used_credits = (prompt_token_number + response_token_number) / 10
            else:
                used_credits = (prompt_token_number + response_token_number) / 100
            if synthesise_switch:
                used_credits += response_token_number / 10

        await websocket.send_json(
            {"command": "Answering complete", "usedCredits": used_credits}
        )

    async def voice_chunk_handler(voice_chunk):
        if audio_transcriber.transcription_complete:
            await websocket.send_json({"command": "Transcription complete"})
            # reset the transcriber
            audio_transcriber.reset_timeout()
            audio_transcriber.transcripts = []

        # Call the add_chunk method with the received voice_chunk
        await audio_transcriber.add_chunk(voice_chunk)

    async def transcribed_response_handler():
        nonlocal sent_transcripts
        while True:
            transcripts = " ".join(audio_transcriber.transcripts)
            if transcripts != sent_transcripts:
                await websocket.send_json({"transcripts": transcripts})
                sent_transcripts = transcripts
            await asyncio.sleep(0.1)

    await websocket.accept()
    print("websocket connected")
    audio_transcriber = await AudioTranscriber.create()
    sent_transcripts = ""
    print("start speech recognition")
    # change this depending on whether encoding is mp3 or wav
    asyncio.create_task(audio_transcriber.process_chunks_wav())
    print("start processing audio chunks")
    asyncio.create_task(transcribed_response_handler())
    print("starting transcripts handler")
    session_id = ""
    audio_language = "zh-CN"
    while True:
        try:
            incoming_data = await asyncio.wait_for(websocket.receive(), timeout=1)
            # print(incoming_data)
            if "text" in incoming_data.keys():
                json_message = json.loads(incoming_data["text"])
                if "prompt" in json_message:
                    await prompt_handler(json_message)

                if "command" in json_message:
                    if json_message["command"] == "STOP_ANSWERING":
                        pass

                if "command" in json_message:
                    if json_message["command"] == "RESET":
                        pass

                if "language" in json_message:
                    audio_language = json_message["language"]
                    audio_transcriber.language = json_message["language"]
                    print(f"Changed the language to: {audio_transcriber.language}")
                    sent_transcripts = ""
                    # Restart the speech recognizer
                    await audio_transcriber.restart_speech_recognizer()

            # transcribing incoming voice_chunks to text
            elif "bytes" in incoming_data.keys():
                voice_chunk = incoming_data["bytes"]
                await voice_chunk_handler(voice_chunk)

        except asyncio.TimeoutError:
            await check_synthesised_mp3_ready()

        # handling exception for websocket disconnection for websocket.receive()
        except (websockets.WebSocketDisconnect, RuntimeError):
            print("WebSocket disconnected")
            break

        except Exception as e:
            # handle all other exceptions
            print(f"An unexpected error occurred: {e}")
            break


# https://www.starlette.io/websockets/
@router.websocket("/chat/stream")
async def chat_stream(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            prompt_request = await websocket.receive_json()
            prompt_request = PromptRequest(**prompt_request)
            prompt = prompt_request.prompt
            history = prompt_request.history
            synthesise_switch = prompt_request.synthesise_switch
            gpt4_switch = prompt_request.gpt4_switch
            model = "gpt-4" if gpt4_switch else "gpt-3.5-turbo"
            response_generator, prompt_token_number = chat(
                prompt=prompt,
                history=history,
                actor="personal assistant",
                max_tokens=500,
                accuracy="medium",
                stream=True,
                model=model,
                session_id="test_api",
            )

            session_id = str(uuid4())
            await websocket.send_json(
                {
                    "session_id": session_id,
                }
            )
            sleep_length = 0.05 if synthesise_switch else 0.01

            response = ""
            for response_chunk in response_generator:
                try:
                    # wait for "stop" command or otherwise keep sending response
                    message = await asyncio.wait_for(
                        websocket.receive_json(), timeout=sleep_length
                    )
                    if message.get("command") == "stop":
                        break
                except asyncio.TimeoutError:
                    chunk_message = response_chunk["choices"][0]["delta"]
                    if hasattr(chunk_message, "content"):
                        if synthesise_switch:
                            await text_to_speech(chunk_message.content, session_id)
                        await websocket.send_json({"content": chunk_message.content})
                        response += chunk_message.content

            response_token_number = calculate_token_number(
                [{"role": "assisstant", "content": response}]
            )
            if gpt4_switch:
                used_credits = (prompt_token_number + response_token_number) / 10
            else:
                used_credits = (prompt_token_number + response_token_number) / 100

            await websocket.send_json({"command": "DONE", "usedCredits": used_credits})

        except (websockets.WebSocketDisconnect, RuntimeError):
            print("WebSocket disconnected")
            break

        except Exception as e:
            # handle all other exceptions
            print(f"An unexpected error occurred: {e}")
            break


class TextToSpeech(BaseModel):
    text: str


# receive text and save a mp3 file
@router.post("/chat/audio_synthesise/text_to_speech")
async def text_to_speech_endpoint(
    background_tasks: BackgroundTasks, text: TextToSpeech, request: Request
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


@router.websocket("/chat/audio_synthesise/check_new_mp3/{session_id}")
async def check_new_mp3_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    while True:
        if session_id in synthesisers:
            if synthesisers[session_id].audio_ready:
                await websocket.send_json({"status": "ready"})
                await websocket.close()
                break
        await asyncio.sleep(0.1)


@router.get("/chat/audio_synthesise/check_new_mp3")
async def check_new_mp3(session_id: str = Depends(get_session_id)):
    # check the files in the directory
    folder = f"output/synthesized/{session_id}"
    if not os.path.exists(folder):
        return {"available_sentences": 0}
    else:
        files = os.listdir(folder)
        return {"available_sentences": len(files)}


# automatically serve the newly generated mp3 file
@router.get("/chat/audio_synthesise/serve_mp3")
async def mp3_stream(
    session_id: str = Depends(get_session_id), request: Request = None
):
    if session_id in synthesisers:
        file_path = Path("output") / "synthesized" / f"{session_id}.mp3"

        def generator_audio():
            last_pos = 0
            timeout_count = 0
            while True:
                audio_data = synthesisers[session_id].stream_callback.get_audio_data()
                new_pos = len(audio_data)
                if new_pos > last_pos:
                    data = audio_data[last_pos:new_pos]
                    if data:
                        yield data
                        timeout_count = 0
                        last_pos = new_pos
                else:
                    time.sleep(0.5)  # Add delay between reads to reduce CPU usage
                    timeout_count += 1
                if synthesisers[session_id].synthesis_complete:
                    synthesisers[session_id].stop_speech_synthesis()
                    # synthesisers[session_id] = None
                    break

        headers = {"Transfer-Encoding": "chunked", "X-Content-Type-Options": "nosniff"}
        return StreamingResponse(
            generator_audio(), media_type="audio/mpeg", headers=headers
        )


@router.websocket("/chat/stream/azureTranscript")
async def azure_transcript_stream(websocket: WebSocket):
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
        nonlocal sent_transcripts
        # handle incoming data during transcription
        if "bytes" in message.keys():
            voice_chunk = message["bytes"]
            # Call the add_chunk method with the received voice_chunk
            await audio_transcriber.add_chunk(voice_chunk)
            # print(f"{time.time() - start_time}: added chunk")
            if audio_transcriber.transcription_complete:
                await websocket.send_json({"command": "DONE"})
                # reset the transcriber
                audio_transcriber.reset_timeout()
                audio_transcriber.transcripts = []

        elif "text" in message.keys():
            json_message = json.loads(message["text"])
            # if "command" in json_message and json_message["command"] == "RESET":
            #     await reset_transcriber()
            if "language" in json_message:
                audio_transcriber.language = json_message["language"]
                print(f"Changed the language to: {audio_transcriber.language}")
                await reset_transcriber()

    await websocket.accept()
    print("websocket connected")

    audio_transcriber = await AudioTranscriber.create()
    sent_transcripts = ""
    print("start speech recognition")

    # change this depending on whether encoding is mp3 or wav
    asyncio.create_task(audio_transcriber.process_chunks_wav())
    print("start processing audio chunks")

    asyncio.create_task(transcripts_handler())
    print("starting transcripts handler")

    while True:
        try:
            message = await websocket.receive()
            await handle_message(message)
            await asyncio.sleep(0.1)
        except asyncio.TimeoutError:
            await asyncio.sleep(0.1)
        except RuntimeError:
            print("WebSocket disconnected")
            audio_transcriber.close_session()
            break
