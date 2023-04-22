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
from parameters import SYNTHESIS_TIMEOUT_LENGTH, DELIMITERS, LANGUAGE_VOICE_MAP
import re
from blingfire import text_to_sentences


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

        self.initial_timeout_length = 3600
        self.reset_timeout()
        self.text_queue = asyncio.Queue()

    def speech_synthesis_to_push_audio_output_stream(self, language: str = "zh-CN"):
        """performs speech synthesis and push audio output to a stream"""

        class PushAudioOutputStreamSampleCallback(
            speechsdk.audio.PushAudioOutputStreamCallback
        ):
            """
            Example class that implements the PushAudioOutputStreamCallback, which is used to show
            how to push output audio to a stream
            """

            def __init__(self, parent: "AudioSynthesiser") -> None:
                # allow child class to call the constructor of its parent class.
                # calling super().__init__(), the PushAudioOutputStreamSampleCallback class inherits
                # the properties and methods of the speechsdk.audio.PushAudioOutputStreamCallback class
                #  and can then add or modify them as needed.
                super().__init__()
                self.parent = parent
                self._audio_data = bytes(0)
                self._closed = False

            def write(self, audio_buffer: memoryview) -> int:
                """
                The callback function which is invoked when the synthesizer has an output audio chunk
                to write out
                """
                self._audio_data += audio_buffer
                self.parent.timeout = time.time() + SYNTHESIS_TIMEOUT_LENGTH
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
                # split_points = [
                #     len(audio_data) // 3,
                #     len(audio_data) * 2 // 3,
                #     len(audio_data),
                # ]
                # for i, pt in enumerate(split_points):
                # segment = audio_data[:pt]
                filename = f"output/synthesized/audio_{time.time()}.mp3"
                with open(filename, "ab") as f:
                    f.write(audio_data)

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

        if language in LANGUAGE_VOICE_MAP:
            voice = LANGUAGE_VOICE_MAP[language]
            speech_config.speech_synthesis_voice_name = voice
        else:
            speech_config.speech_synthesis_language = language

        # Creates customized instance of PushAudioOutputStreamCallback
        self.stream_callback = PushAudioOutputStreamSampleCallback(parent=self)
        # Creates audio output stream from the callback
        self.push_stream = speechsdk.audio.PushAudioOutputStream(self.stream_callback)
        # Creates a speech synthesizer using push stream as audio output.
        stream_config = speechsdk.audio.AudioOutputConfig(stream=self.push_stream)
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=stream_config
        )

    def close(self):
        del self.speech_synthesizer
        del self.result

    async def add_text(self, chunk: str) -> None:
        """Add a chunk of text to the text queue."""
        await self.text_queue.put(chunk)

    # digest the text in the queue
    async def process_text(self) -> None:
        """Process the text in the text queue."""
        while True:
            text = await self.text_queue.get()
            self.result = self.speech_synthesizer.speak_text_async(text).get()
            await asyncio.sleep(0.1)

    async def dummy_text_receiver(self):
        """Dummy text receiver to mimic receiving chunks of text."""
        texts = "“布姐”是指《JOJO的奇妙冒险》中的角色布鲁诺·布加拉提（Bruno Bucciarati）。他是第五部《黄金之风》中的主要角色之一，也是乔鲁诺·乔巴纳的盟友和领袖。布鲁诺·布加拉提是一个拥有强大替身能力的黑帮分子。”。"
        text_to_synthesise = ""
        accumulated_text = ""
        for i, chunk in enumerate(texts):
            # use bling fire to split the text into sentences
            # accumulated_text += chunk
            # sentences = text_to_sentences(accumulated_text)
            # sentences = sentences.split("\n")
            # if len(sentences) > 1:
            #     print(f"adding text: {sentences[0]}")
            #     await self.add_text(sentences[0])
            #     accumulated_text = sentences[1]

            # # for the last sentence
            # if i == len(texts) - 1:
            #     print(f"adding text: {accumulated_text}")
            #     await self.add_text(accumulated_text)

            # use a delimiter to split the text into sentences
            text_to_synthesise += chunk
            if chunk in DELIMITERS:
                print(f"adding text: {text_to_synthesise}")
                await self.add_text(text_to_synthesise)
                text_to_synthesise = ""

            # for the last sentence
            if i == len(texts) - 1:
                print(f"adding text: {text_to_synthesise}")
                await self.add_text(text_to_synthesise)
                text_to_synthesise = ""

            await asyncio.sleep(0.01)

    def reset_timeout(self):
        """Reset the timeout at the start of each synthesis"""
        self.timeout = time.time() + self.initial_timeout_length

    @property
    def synthesis_complete(self):
        return time.time() > self.timeout


if __name__ == "__main__":
    audio_synthesiser = AudioSynthesiser()

    async def run_dummy():
        print(f"starting at {time.time()}")
        audio_synthesiser = AudioSynthesiser()
        audio_synthesiser.speech_synthesis_to_push_audio_output_stream(language="zh-CN")

        asyncio.create_task(audio_synthesiser.dummy_text_receiver())
        asyncio.create_task(audio_synthesiser.process_text())
        # continue to run until synthesis is complete
        while True:
            if audio_synthesiser.synthesis_complete:
                print("SYNTHESIS COMPLETE")
                break
            await asyncio.sleep(0.1)

        audio_synthesiser.stream_callback.save_to_file()
        audio_synthesiser.close()

    asyncio.run(run_dummy())
