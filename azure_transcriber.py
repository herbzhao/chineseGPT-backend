import asyncio
import io
import os
import re
import threading
import time
import wave
from pathlib import Path

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from pydub import AudioSegment

from parameters import INITIAL_TIMEOUT_LENGTH, TRANSCRIBE_TIMEOUT_LENGTH


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
        self.split_append_silence = 0
        self.transcripts = []
        self.first_chunk = None
        self.reset_timeout()
        self.chunks_queue = asyncio.Queue()
        self.language = "zh-CN"

    @classmethod
    async def create(cls):
        self = cls()
        await self.start_transcriber()
        return self

    async def restart_speech_recognizer(self):
        try:
            await self.close_session()
        except Exception as e:
            print(e)
        # Start a new speech recognizer after a short delay
        await asyncio.sleep(1)
        await self.start_transcriber()

    def reset_timeout(self):
        self.timeout = time.time() + INITIAL_TIMEOUT_LENGTH

    def convert_audio_segment_to_wav(self, audio_segment, append_silence_length=0):
        silence = AudioSegment.silent(duration=append_silence_length)
        # append silence to the end of the audio segment
        audio_segment += silence
        # change it to (16 kHz, 16-bit, mono channel)
        audio_segment = audio_segment.set_frame_rate(16000)
        audio_segment = audio_segment.set_channels(1)
        audio_segment = audio_segment.set_sample_width(2)
        with io.BytesIO() as wav_file:
            audio_segment.export(wav_file, format="wav")
            audio_segment = AudioSegment.from_file(wav_file, format="wav")
        return audio_segment

    async def dummy_chunks_receiver(self, folder_name):
        filenames = [
            filename for filename in Path(folder_name).iterdir() if filename.is_file()
        ]
        # sort the filenames
        filenames = sorted(filenames, key=lambda x: int(x.stem.replace("audio", "")))
        # each file is a chunk
        for i, filename in enumerate(filenames):
            with open(filename, "rb") as f:
                # store all the chunks received so far
                chunk = f.read()
                # await self.add_chunk(chunk)
                await self.chunks_queue.put(chunk)

                # mimic receiving chunks every x seconds
                await asyncio.sleep(0.01)

    async def process_chunks_mp3(self):
        while True:
            chunk = await self.chunks_queue.get()
            # append the first chunk to the new chunk for necessary headings
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(chunk), "mp3")
                # remove the length of first chunk from the new audio segment
                audio_segment_wav = self.convert_audio_segment_to_wav(audio_segment)
                self.push_stream.write(audio_segment_wav.raw_data)
            except Exception as e:
                print(e)
                pass
            await asyncio.sleep(0.1)

    async def process_chunks_wav(self):
        while True:
            chunk = await self.chunks_queue.get()
            self.push_stream.write(chunk)
            await asyncio.sleep(0.1)

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

        self.speech_recognizer.recognizing.connect(
            lambda evt: self.recognizing_callback(evt)
        )
        self.speech_recognizer.recognized.connect(
            lambda evt: self.recognized_callback(evt)
        )
        self.speech_recognizer.session_started.connect(
            lambda evt: print(f"SESSION STARTED: {evt} {time.time()}")
        )
        self.speech_recognizer.session_stopped.connect(
            lambda evt: print("CLOSING on {}".format(evt))
        )
        self.speech_recognizer.canceled.connect(
            lambda evt: print("CANCELED {}".format(evt))
        )

        self.speech_recognizer.start_continuous_recognition()
        # asyncio.create_task(self.transcribe_async())

    # async def transcribe_async(self):
    # Start continuous recognition
    # self.speech_recognizer.start_continuous_recognition()

    # loop serves as a mechanism to keep the program running while it's waiting for more audio chunks
    # and transcriptions. The timeout value is updated in the recognizing_callback function,
    #  This means that the loop will keep running until there's no new transcription received within the self.timeout_diff duration.
    # while not self.transcription_complete:
    # await asyncio.sleep(0.05)

    # if self.transcription_complete:
    # self.speech_recognizer.stop_continuous_recognition()
    # self.push_stream.close()

    async def add_chunk(self, chunk):
        await self.chunks_queue.put(chunk)

    @property
    def transcription_complete(self):
        return time.time() > self.timeout

    async def close_session(self):
        # force timeout to kick in
        self.timeout = time.time() - TRANSCRIBE_TIMEOUT_LENGTH
        # self.transcription_complete = True
        if self.speech_recognizer:
            self.speech_recognizer.stop_continuous_recognition()
            self.speech_recognizer = None
        if self.push_stream:
            self.push_stream.close()
            self.push_stream = None
        self.transcripts.clear()
        self.reset_timeout()

    def recognizing_callback(self, evt: speechsdk.SpeechRecognitionEventArgs):
        if len(self.transcripts) == 0:
            self.transcripts.append(evt.result.text)
        else:
            self.transcripts[-1] = evt.result.text
        self.timeout = time.time() + TRANSCRIBE_TIMEOUT_LENGTH
        print(f"RECOGNIZING: {evt.result.text} at {time.time()}")

    def recognized_callback(self, evt: speechsdk.SpeechRecognitionEventArgs):
        self.transcripts[-1] = evt.result.text
        self.transcripts.append("")

    async def stop_callback(self, evt: speechsdk.SessionEventArgs):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print("CLOSING on {}".format(evt))
        await self.speech_recognizer.stop_continuous_recognition()
        self.push_stream.close()


if __name__ == "__main__":

    async def run_dummy():
        print(f"starting at {time.time()}")
        sent_transcripts = ""
        audio_transcriber = await AudioTranscriber.create()
        asyncio.create_task(audio_transcriber.dummy_chunks_receiver("resources/test"))
        asyncio.create_task(audio_transcriber.process_chunks_wav())
        while True:
            await asyncio.sleep(0.01)
            transcripts = " ".join(audio_transcriber.transcripts)
            print(audio_transcriber.transcripts)
            if transcripts != sent_transcripts:
                sent_transcripts = transcripts
                print(f"{time.time()}: sent {sent_transcripts}")

    asyncio.run(run_dummy())
