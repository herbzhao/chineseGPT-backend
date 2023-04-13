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

load_dotenv()
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
if os.path.exists(".env.production") and os.getenv("ENVIRONMENT") == "production":
    print("GETTING PRODUCTION ENVIRONMENT VARIABLES")
    load_dotenv(".env.production")


class AudioTranscripter:
    def __init__(self):
        self.chunks = b""
        self.audio_segments = []
        # time stamp of consumed segment alrady sent to azure
        self.consumed_segment_length = 0
        self.wav_files = []
        self.split_length = 4000

    def chunk_receiver(self, folder_name):
        # mimic the behaviour of receiving chunk every 2.2 seconds
        # and append chunk to a growing stream
        # open files in the folder
        # for each file, read the file and append to the stream
        filenames = [
            filename for filename in Path(folder_name).iterdir() if filename.is_file()
        ]
        for filename in filenames:
            print(filename)
            # async sleep for 2.2 seconds
            with open(filename, "rb") as f:
                self.chunks += f.read()
            # convert chunks to audio segment
            audio_segment = AudioSegment.from_file(
                io.BytesIO(self.chunks), format="webm"
            )
            # split the audio segment into 1 second chunks
            audio_segments = []
            # divide any unconsumed audio segment into 1 second chunks
            # TODO: also need to take care of the case where the audio segment is shorter than 1 second in the end
            dividers = list(
                range(
                    self.consumed_segment_length, len(audio_segment), self.split_length
                )
            )
            divided_lengths = [
                [dividers[i], dividers[i + 1]]
                for i, _ in enumerate(dividers)
                if i < len(dividers) - 1
            ]
            for start, end in divided_lengths:
                audio_segments.append(audio_segment[start:end])

            self.consumed_segment_length = dividers[-1]
            # store the audio segments
            self.audio_segments += audio_segments

            for i, audio_segment in enumerate(audio_segments):
                # convert audio segment to wav and save to bytesIO
                # with io.BytesIO() as wav_file:
                audio_segment.export(
                    f"resources/chunks/split/{i+len(self.audio_segments)}.wav",
                    format="wav",
                )
                # send the wav file to azure
                # self.push_stream.write(wav_file.getvalue())
                # print("sending to azure")

    def stream_recognize_async(self):
        speech_config = speechsdk.SpeechConfig(
            subscription=os.environ.get("SPEECH_KEY"),
            region=os.environ.get("SPEECH_REGION"),
        )
        speech_config.speech_recognition_language = "zh-CN"
        # Create a PushAudioInputStream object
        self.push_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)

        # Create a speech recognizer object using the PushAudioInputStream object
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )
        done = False

        # Connect callbacks to the events fired by the speech recognizer
        def stop_cb(evt: speechsdk.SessionEventArgs):
            """callback that signals to stop continuous recognition upon receiving an event `evt`"""
            print("CLOSING on {}".format(evt))
            nonlocal done
            done = True

        def recognizing_callback(evt: speechsdk.SpeechRecognitionEventArgs):
            """callback for recognized event"""
            print("RECOGNIZED: {}".format(evt))

        def recognized_callback(evt: speechsdk.SpeechRecognitionEventArgs):
            """callback for recognized event"""
            print("RECOGNIZED: {}".format(evt))

        speech_recognizer.recognizing.connect(lambda evt: recognizing_callback(evt))
        speech_recognizer.recognized.connect(lambda evt: recognized_callback(evt))
        speech_recognizer.session_started.connect(
            lambda evt: print("SESSION STARTED: {}".format(evt))
        )
        speech_recognizer.session_stopped.connect(stop_cb)
        speech_recognizer.canceled.connect(lambda evt: print("CANCELED {}".format(evt)))

        # Start continuous recognition
        speech_recognizer.start_continuous_recognition_async()

        filenames = [
            filename
            for filename in Path("resources/chunks/split").iterdir()
            if filename.is_file()
        ]
        for filename in filenames:
            with open(filename, "rb") as f:
                self.push_stream.write(f.read())
                print("sending to azure")
                # time.sleep(5)

        time.sleep(20)
        # stop_cb("STOP")
        # # Stop continuous recognition
        # speech_recognizer.stop_continuous_recognition_async()

        # # Close the PushAudioInputStream object
        # self.push_stream.close()


if __name__ == "__main__":
    # async def main():
    audio_transcripter = AudioTranscripter()
    # audio_transcripter.chunk_receiver("resources/chunks")
    audio_transcripter.stream_recognize_async()
    # await audio_transcripter.chunk_receiver(r"resources/chunks")

    # asyncio.run(main())
