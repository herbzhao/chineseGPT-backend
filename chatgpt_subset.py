@app.websocket("/chat/stream/azureTranscript")
async def azure_transcript_stream(websocket: WebSocket):
    await websocket.accept()
    audio_transcriber = AudioTranscriber()
    sent_transcripts = ""
    print("websocket connected")

    async def reset_transcriber():
        nonlocal sent_transcripts
        audio_transcriber.transcripts.clear()
        sent_transcripts = ""

    # Start a background task to periodically check for new transcripts
    async def transcripts_handler():
        nonlocal sent_transcripts
        while True:
            transcripts = " ".join(audio_transcriber.transcripts)
            if transcripts != sent_transcripts:
                await websocket.send_json({"transcripts": transcripts})
                sent_transcripts = transcripts
            await asyncio.sleep(0.1)

    print("start processing chunks")
    asyncio.create_task(audio_transcriber.process_chunks())
    print("starting transcripts handler")
    asyncio.create_task(transcripts_handler())

    while True:
        try:
            message = await websocket.receive()
        except asyncio.TimeoutError:
            continue
        if "bytes" in message.keys():
            voice_chunk = message["bytes"]
            # Call the add_chunk method with the received voice_chunk
            await audio_transcriber.add_chunk(voice_chunk)

            if audio_transcriber.transcription_complete:
                await websocket.send_json({"command": "DONE"})

        elif message["type"] == "text":
            if message["text"] == "RESET":
                await reset_transcriber()



class AudioTranscriber:
    def __init__(self):
        load_dotenv()
        if os.path.exists(".env.local"):
            load_dotenv(".env.local")
        if (
            os.path.exists(".env.production")
            and os.getenv("ENVIRONMENT") == "production"
        ):
            print("GETTING PRODUCTION ENVIRONMENT VARIABLES")
            load_dotenv(".env.production")

        self.chunks = b""
        self.speech_recognizer = None
        self.audio_segments = []
        # time stamp of consumed segment alrady sent to azure
        self.consumed_segment_length = 0
        self.wav_files = []
        self.split_length = AUDIO_SEGMENT_SPLIT_LENGTH
        self.split_append_silence = self.split_length // 50
        self.transcripts = []
        # if no new transcript is received for x seconds, stop the stream
        self.timeout_length = AUDIO_TIMEOUT_LENGTH
        self.initial_timeout_diff = 100
        self.timeout = time.time() + self.initial_timeout_diff
        self.chunks_queue = asyncio.Queue()
        asyncio.create_task(self.start_transcriber())