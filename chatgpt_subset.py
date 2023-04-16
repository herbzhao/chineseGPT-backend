
@app.websocket("/chat/stream/azureTranscript")
async def azure_transcript_stream(websocket: WebSocket):
    ...
    async def handle_message(message):
        if "bytes" in message.keys():
            voice_chunk = message["bytes"]
            # Call the add_chunk method with the received voice_chunk
            await audio_transcriber.add_chunk(voice_chunk)
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

    ....

    while True:
        try:
            message = await websocket.receive()
            await handle_message(message)
        except asyncio.TimeoutError:
            continue
        await asyncio.sleep(0.1)


class AudioTranscriber:
    def __init__(self):

        self.chunks_queue = asyncio.Queue()
        self.language = "zh-CN"

    @classmethod
    async def create(cls):
        self = cls()
        await self.start_transcriber()
        return self

    async def restart_speech_recognizer(self):
        await self.close_session()
        # Start a new speech recognizer
        await self.start_transcriber()


            async def start_transcriber(self):
        speech_config = speechsdk.SpeechConfig(
            subscription=os.environ.get("SPEECH_KEY"),
            region=os.environ.get("SPEECH_REGION"),
        )
        speech_config.speech_recognition_language = self.language
        # Create a PushAudioInputStream object
        self.push_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)

        # Create a speech recognizer object using the PushAudioInputStream object
        self.speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        asyncio.create_task(self.transcribe_async())


    async def transcribe_async(self):
        # Start continuous recognition
        self.speech_recognizer.start_continuous_recognition()

        # loop serves as a mechanism to keep the program running while it's waiting for more audio chunks
        # and transcriptions. The timeout value is updated in the recognizing_callback function,
        #  This means that the loop will keep running until there's no new transcription received within the self.timeout_diff duration.
        while not self.transcription_complete:
            await asyncio.sleep(0.01)

        if self.transcription_complete:
            self.speech_recognizer.stop_continuous_recognition()
            self.push_stream.close()
