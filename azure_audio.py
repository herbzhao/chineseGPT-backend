import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import time
from pydub import AudioSegment
import time
import threading
import wave
from pathlib import Path
import asyncio
import io
import threading


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
        self.push_stream = None
        self.speech_recognizer = None
        self.audio_segments = []
        # time stamp of consumed segment alrady sent to azure
        self.consumed_segment_length = 0
        self.wav_files = []
        self.split_length = 2000
        self.split_append_silence = self.split_length // 10
        self.end_of_stream_silence = 2000
        self.transcripts = []
        self.processing_done = False
        # if no new transcript is received for x seconds, stop the stream
        self.timeout_diff = 2
        self.initial_timeout_diff = 10

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

    def calculate_split_lengths(self, complete_audio_segment):
        # divide any unconsumed audio segment into 1 second chunks
        dividers = list(
            range(
                self.consumed_segment_length,
                len(complete_audio_segment),
                self.split_length,
            )
        )
        split_lengths = [
            [dividers[i], dividers[i + 1]]
            for i, _ in enumerate(dividers)
            if i < len(dividers) - 1
        ]
        if len(split_lengths) > 0:
            # If all the chunks are used up, add the last chunk
            split_lengths[-1][1] = len(complete_audio_segment)
            self.consumed_segment_length = split_lengths[-1][1]

        return split_lengths

    async def add_chunk(self, chunk):
        await self.chunks_queue.put(chunk)

    async def dummy_chunks_receiver(self, folder_name):
        filenames = [
            filename for filename in Path(folder_name).iterdir() if filename.is_file()
        ]
        # each file is a chunk
        for i, filename in enumerate(filenames):
            with open(filename, "rb") as f:
                # store all the chunks received so far
                chunk = f.read()
                # await self.add_chunk(chunk)
                await self.chunks_queue.put(chunk)

                # mimic receiving chunks every x seconds
                await asyncio.sleep(2)

    async def chunks_handler(self):
        accumulated_chunks = b""
        while True:
            # get a chunk from the queue
            chunk = await self.chunks_queue.get()
            # store all the chunks received so far
            accumulated_chunks += chunk
            # convert chunks to audio segment
            complete_audio_segment = AudioSegment.from_file(
                io.BytesIO(accumulated_chunks), format="webm"
            )

            split_lengths = self.calculate_split_lengths(complete_audio_segment)
            # send the split audio segments to azure
            for j, (start, end) in enumerate(split_lengths):
                audio_segment = self.convert_audio_segment_to_wav(
                    complete_audio_segment[start:end],
                    append_silence_length=self.split_append_silence,
                )
                self.push_stream.write(audio_segment.raw_data)

    async def transcribe_async(self, language="zh-CN"):
        speech_config = speechsdk.SpeechConfig(
            subscription=os.environ.get("SPEECH_KEY"),
            region=os.environ.get("SPEECH_REGION"),
        )
        speech_config.speech_recognition_language = language
        # Create a PushAudioInputStream object
        self.push_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)

        # Create a speech recognizer object using the PushAudioInputStream object
        self.speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        def recognizing_callback(evt: speechsdk.SpeechRecognitionEventArgs):
            """callback for recognized event"""
            if len(self.transcripts) == 0:
                self.transcripts.append(evt.result.text)
            else:
                self.transcripts[-1] = evt.result.text
            print(self.transcripts[-1])
            self.timeout = time.time() + self.timeout_diff
            # print("RECOGNIZING: {}".format(evt))

        def recognized_callback(evt: speechsdk.SpeechRecognitionEventArgs):
            """callback for recognized event"""
            self.transcripts[-1] = evt.result.text
            self.transcripts.append("")
            print("recognized so far:")
            print(self.transcripts)
            # # Check if this is the final result
            # print("RECOGNIZED: {}".format(evt))

        # Connect callbacks to the events fired by the speech recognizer
        def stop_callback(evt: speechsdk.SessionEventArgs):
            """callback that signals to stop continuous recognition upon receiving an event `evt`"""
            self.completed = True
            print("CLOSING on {}".format(evt))

        self.speech_recognizer.recognizing.connect(
            lambda evt: recognizing_callback(evt)
        )
        self.speech_recognizer.recognized.connect(lambda evt: recognized_callback(evt))
        self.speech_recognizer.session_started.connect(
            lambda evt: print("SESSION STARTED: {}".format(evt))
        )
        self.speech_recognizer.session_stopped.connect(stop_callback)
        self.speech_recognizer.canceled.connect(
            lambda evt: print("CANCELED {}".format(evt))
        )

        # Start continuous recognition
        result_future = self.speech_recognizer.start_continuous_recognition_async()
        # wait for voidfuture, so we know engine initialization is done.
        result_future.get()
        #  have a greater timeout value for the first transcription
        self.timeout = time.time() + self.initial_timeout_diff

        # loop serves as a mechanism to keep the program running while it's waiting for more audio chunks
        # and transcriptions. The timeout value is updated in the recognizing_callback function,
        #  This means that the loop will keep running until there's no new transcription received within the self.timeout_diff duration.
        while time.time() < self.timeout:
            await asyncio.sleep(0.1)

        print("done")
        self.speech_recognizer.stop_continuous_recognition_async()
        self.push_stream.close()

    async def run(self):
        self.audio_queue = asyncio.Queue()
        self.chunks_queue = asyncio.Queue()

        task1 = asyncio.create_task(self.transcribe_async())
        task2 = asyncio.create_task(self.chunks_handler())

        await task1
        await task2

    async def run_dummy(self):
        self.audio_queue = asyncio.Queue()
        self.chunks_queue = asyncio.Queue()

        task1 = asyncio.create_task(self.transcribe_async())
        task2 = asyncio.create_task(self.dummy_chunks_receiver("resources/chunks"))
        task3 = asyncio.create_task(self.chunks_handler())

        await task1
        await task2
        await task3


if __name__ == "__main__":
    # async def main():
    audio_transcripter = AudioTranscriber()
    asyncio.run(audio_transcripter.run_dummy())
