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
from parameters import AUDIO_SEGMENT_SPLIT_LENGTH, AUDIO_TIMEOUT_LENGTH
import re


class AudioSynthesiser:
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

        # self.delimiters = "[。！？，。,  . \n]"
        # self.delimiters = "[！？。 . \n]"
        self.delimiters = "[\n]"
        self.text_queue = asyncio.Queue()

    def speech_synthesis_to_push_audio_output_stream(self):
        """performs speech synthesis and push audio output to a stream"""

        class PushAudioOutputStreamSampleCallback(
            speechsdk.audio.PushAudioOutputStreamCallback
        ):
            """
            Example class that implements the PushAudioOutputStreamCallback, which is used to show
            how to push output audio to a stream
            """

            def __init__(self) -> None:
                super().__init__()
                self._audio_data = bytes(0)
                self._closed = False

            def write(self, audio_buffer: memoryview) -> int:
                """
                The callback function which is invoked when the synthesizer has an output audio chunk
                to write out
                """
                self._audio_data += audio_buffer
                print("{} bytes received.".format(audio_buffer.nbytes))
                return audio_buffer.nbytes

            def close(self) -> None:
                """
                The callback function which is invoked when the synthesizer is about to close the
                stream.
                """
                self._closed = True
                print("Push audio output stream closed.")

            def get_audio_data(self) -> bytes:
                return self._audio_data

            def get_audio_size(self) -> int:
                return len(self._audio_data)

            def save_to_file(self) -> None:
                audio_data = self.get_audio_data()
                # split the audio_data into three parts
                # and save them to three different files
                split_points = [
                    len(audio_data) // 3,
                    len(audio_data) * 2 // 3,
                    len(audio_data),
                ]
                for i, pt in enumerate(split_points):
                    segment = audio_data[:pt]
                    filename = f"resources/synthesized/audio{i}.mp3"
                    with open(filename, "ab") as f:
                        f.write(segment)

                print(f"Audio data saved to {os.path.abspath(filename)}")

        # Creates an instance of a speech config with specified subscription key and service region.
        speech_config = speechsdk.SpeechConfig(
            subscription=os.environ.get("SPEECH_KEY"),
            region=os.environ.get("SPEECH_REGION"),
        )
        # Sets the synthesis output format.
        # The full list of supported format can be found here:
        # https://docs.microsoft.com/azure/cognitive-services/speech-service/rest-text-to-speech#audio-outputs
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
        language = "zh-CN"
        speech_config.speech_synthesis_language = language

        # Creates customized instance of PushAudioOutputStreamCallback
        self.stream_callback = PushAudioOutputStreamSampleCallback()
        # Creates audio output stream from the callback
        self.push_stream = speechsdk.audio.PushAudioOutputStream(self.stream_callback)
        # Creates a speech synthesizer using push stream as audio output.
        stream_config = speechsdk.audio.AudioOutputConfig(stream=self.push_stream)
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=stream_config
        )

        # # Destroys result which is necessary for destroying speech synthesizer
        # del result
        # # Destroys the synthesizer in order to close the output stream.
        # del self.speech_synthesizer

    # add prompt to the queue
    async def add_text(self, chunk):
        await self.text_queue.put(chunk)

    # digest the text in the queue
    async def process_text(self):
        while True:
            text = await self.text_queue.get()
            result = self.speech_synthesizer.speak_text_async(text).get()
            await asyncio.sleep(0.1)

    async def dummy_text_receiver(self):
        texts = ["今天的天气怎么样?", "明天我需要准备什么?", "请提醒我下午两点开会。"]
        for text in texts:
            await self.add_text(text)
            # mimic receiving chunks every x seconds
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    audio_synthesiser = AudioSynthesiser()

    async def run_dummy():
        print(f"starting at {time.time()}")
        audio_synthesiser = AudioSynthesiser()
        audio_synthesiser.speech_synthesis_to_push_audio_output_stream()

        asyncio.create_task(audio_synthesiser.dummy_text_receiver())
        asyncio.create_task(audio_synthesiser.process_text())

        await asyncio.sleep(5)
        audio_synthesiser.stream_callback.save_to_file()

    asyncio.run(run_dummy())
