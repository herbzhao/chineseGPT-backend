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

load_dotenv()
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
if os.path.exists(".env.production") and os.getenv("ENVIRONMENT") == "production":
    print("GETTING PRODUCTION ENVIRONMENT VARIABLES")
    load_dotenv(".env.production")


class AudioTranscripter:
    def __init__(self):
        self.chunks = b""
        self.push_stream = None
        self.speech_recognizer = None
        self.audio_segments = []
        # time stamp of consumed segment alrady sent to azure
        self.consumed_segment_length = 0
        self.wav_files = []
        self.split_length = 2000
        self.split_append_silence = self.split_length // 10
        self.transcripts = []

    def convert_webm_to_wav(self, webm_file, output_filename, append_silence_length=0):
        audio_segment = AudioSegment.from_file(webm_file, format="webm")
        silence = AudioSegment.silent(duration=append_silence_length)
        # append silence to the end of the audio segment
        audio_segment += silence

        # change it to (16 kHz, 16-bit, mono channel)
        audio_segment = audio_segment.set_frame_rate(16000)
        audio_segment = audio_segment.set_channels(1)
        audio_segment = audio_segment.set_sample_width(2)

        audio_segment.export(output_filename, format="wav")
        return output_filename

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

    def get_split_lengths(self, complete_audio_segment):
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

    async def chunk_handler(self, folder_name):
        # mimic the behaviour of receiving chunk every 2.2 seconds
        # and append chunk to a growing stream
        # open files in the folder
        # for each file, read the file and append to the stream
        filenames = [
            filename for filename in Path(folder_name).iterdir() if filename.is_file()
        ]
        # each file is a chunk
        for i, filename in enumerate(filenames):
            with open(filename, "rb") as f:
                # store all the chunks received so far
                self.chunks += f.read()
            # convert chunks to audio segment
            complete_audio_segment = AudioSegment.from_file(
                io.BytesIO(self.chunks), format="webm"
            )

            split_lengths = self.get_split_lengths(complete_audio_segment)
            print(split_lengths)

            # send the split audio segments to azure
            for j, (start, end) in enumerate(split_lengths):
                audio_segment = self.convert_audio_segment_to_wav(
                    complete_audio_segment[start:end],
                    append_silence_length=self.split_append_silence,
                )
                self.push_stream.write(audio_segment.raw_data)
                # read the bytesIO object
                await asyncio.sleep(2)

    async def stream_recognize_async(self):
        speech_config = speechsdk.SpeechConfig(
            subscription=os.environ.get("SPEECH_KEY"),
            region=os.environ.get("SPEECH_REGION"),
        )
        speech_config.speech_recognition_language = "zh-CN"
        # Create a PushAudioInputStream object
        self.push_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)

        # Create a speech recognizer object using the PushAudioInputStream object
        self.speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )
        done = False

        # Connect callbacks to the events fired by the speech recognizer
        def stop_cb(evt: speechsdk.SessionEventArgs):
            """callback that signals to stop continuous recognition upon receiving an event `evt`"""
            print("CLOSING on {}".format(evt))
            print(self.transcripts)
            nonlocal done
            done = True

        def recognizing_callback(evt: speechsdk.SpeechRecognitionEventArgs):
            """callback for recognized event"""
            if len(self.transcripts) == 0:
                self.transcripts.append(evt.result.text)
            else:
                self.transcripts[-1] = evt.result.text
            print(self.transcripts[-1])
            print("RECOGNIZING: {}".format(evt))

        def recognized_callback(evt: speechsdk.SpeechRecognitionEventArgs):
            """callback for recognized event"""
            self.transcripts[-1] = evt.result.text
            self.transcripts.append([])
            print(self.transcripts)
            print("RECOGNIZED: {}".format(evt))

        self.speech_recognizer.recognizing.connect(
            lambda evt: recognizing_callback(evt)
        )
        self.speech_recognizer.recognized.connect(lambda evt: recognized_callback(evt))
        self.speech_recognizer.session_started.connect(
            lambda evt: print("SESSION STARTED: {}".format(evt))
        )
        self.speech_recognizer.session_stopped.connect(stop_cb)
        self.speech_recognizer.canceled.connect(
            lambda evt: print("CANCELED {}".format(evt))
        )

        # Start continuous recognition
        self.speech_recognizer.start_continuous_recognition_async()

    async def run(self):
        task1 = asyncio.create_task(self.stream_recognize_async())
        task2 = asyncio.create_task(self.chunk_handler("resources/chunks"))
        await task1
        await task2

        audio_transcripter.speech_recognizer.stop_continuous_recognition_async()
        audio_transcripter.push_stream.close()


if __name__ == "__main__":
    # async def main():
    audio_transcripter = AudioTranscripter()
    asyncio.run(audio_transcripter.run())

    # audio_transcripter.stream_recognize_async()
    # asyncio.run(audio_transcripter.chunk_receiver("resources/chunks"))
    # # stop the stream
    # audio_transcripter.speech_recognizer.stop_continuous_recognition_async()
    # audio_transcripter.push_stream.close()
    # # await audio_transcripter.chunk_receiver(r"resources/chunks")

    # asyncio.run(main())
