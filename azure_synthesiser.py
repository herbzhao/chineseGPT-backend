import asyncio
import io
import os
import re
import threading
import time
import wave
from pathlib import Path

import azure.cognitiveservices.speech as speechsdk
from blingfire import text_to_sentences
from dotenv import load_dotenv
from pydub import AudioSegment

from parameters import (
    DELIMITERS_REGEX,
    INITIAL_TIMEOUT_LENGTH,
    LANGUAGE_VOICE_MAP,
    SYNTHESIS_TIMEOUT_LENGTH,
    TEXT_RECEIVE_TIMEOUT_LENGTH,
)


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

        self.reset_timeout()
        self.text_queue = asyncio.Queue()
        self.output_filename = ""
        self.session_id = None
        self.audio_ready = False
        self.output_folder = Path("output") / "synthesized"

    async def speech_synthesis_to_mp3(
        self, text: str = "", filename: str = "", language: str = "zh-CN"
    ):
        """performs speech synthesis to mp3 file"""
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

        # save to mp3 file
        audio_config = speechsdk.audio.AudioOutputConfig(filename=filename)

        # Create a BytesIO object to store the synthesized audio
        # output_stream = io.BytesIO()
        # audio_config = speechsdk.audio.AudioOutputConfig(stream=output_stream)

        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=audio_config
        )
        # Performs speech synthesis to mp3 file.
        result = speech_synthesizer.speak_text(text)

        # Checks result.
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized and saved to {}".format(filename))

        else:
            print("Speech synthesis failed")
            return None

    def start_speech_synthesis_using_push_stream(self, language: str = "zh-CN"):
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
                # print("{} bytes received.".format(audio_buffer.nbytes))
                # filename = self.parent.output_folder / f"{self.parent.session_id}.mp3"
                # # append the newly received audio data to the existing file
                # with open(filename, "ab") as f:
                #     # print(f"Saving audio to {filename}")
                #     f.write(audio_buffer)
                self.parent.audio_ready = True
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
                filename = f"output/synthesized/audio_{time.time()}.mp3"
                with open(filename, "ab") as f:
                    f.write(audio_data)

                self.parent.output_filename = filename
                print(f"Audio data saved to {os.path.abspath(filename)}")

            def save_to_file_session(self, session_id: str) -> None:
                audio_data = self.get_audio_data()
                filename = f"output/synthesized/audio_{session_id}.mp3"
                # append the newly received audio data to the existing file
                with open(filename, "ab") as f:
                    print(f"Saving audio to {filename}")
                    f.write(audio_data)
                # print(f"Audio data saved to {os.path.abspath(filename)}")

                # clean up the audio data
                self._audio_data = bytes(0)

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

    def stop_speech_synthesis(self):
        try:
            del self.speech_synthesizer
            del self.result
        except:
            pass

    async def add_text(self, chunk: str) -> None:
        """Add a chunk of text to the text queue, waiting to be processed."""
        await self.text_queue.put(chunk)

    async def process_text(self) -> None:
        """Process the text in the text queue and synthesise the text."""
        accumulated_text = ""
        # this timeout is for receiving new text
        text_receive_timeout_length = INITIAL_TIMEOUT_LENGTH
        while True:
            await asyncio.sleep(0.1)
            try:
                text = await asyncio.wait_for(
                    self.text_queue.get(), timeout=text_receive_timeout_length
                )
                text_receive_timeout_length = TEXT_RECEIVE_TIMEOUT_LENGTH
                # use bling fire to split the text into sentences
                accumulated_text += text
                # sentences = text_to_sentences(accumulated_text)
                # sentences = sentences.split("\n")
                sentences = self.split_text_to_sentences(accumulated_text)

                if len(sentences) > 1:
                    # print(f"synthesising: {sentences[0]}")
                    self.speech_synthesizer.speak_text_async(sentences[0])
                    accumulated_text = "".join(sentences[1:])

            # set a timeout, if the timeout is reached, then synthesise the rest of the text
            except asyncio.TimeoutError:
                # set a timeout, if the timeout is reached, then synthesise the rest of the text
                if accumulated_text:
                    # print(f"synthesizing final sentence: {accumulated_text}")
                    self.synthesis_timeout = time.time() + SYNTHESIS_TIMEOUT_LENGTH
                    self.speech_synthesizer.speak_text_async(accumulated_text)
                    # reset the loop
                    accumulated_text = ""
                    synthesised_sentence = 0
                    text_receive_timeout_length = INITIAL_TIMEOUT_LENGTH

    async def dummy_text_receiver(self):
        """Dummy text receiver to mimic receiving chunks of text."""
        texts = (
            "有一天，小花在公园里玩耍，发现了一个漂亮的蝴蝶。小花兴奋地追赶着蝴蝶，但是它不小心跑到了一个陌生的地方。小花感到害怕和孤独，不知道该怎么回家。"
        )
        text_to_synthesise = ""
        accumulated_text = ""
        # split the text by 3 elements and send to the queue
        for i in range(0, len(texts), 3):
            await self.add_text(texts[i : i + 3])
            await asyncio.sleep(0.01)

    def reset_timeout(self):
        """Reset the timeout at the start of each synthesis"""
        self.synthesis_timeout = time.time() + INITIAL_TIMEOUT_LENGTH

    def split_text_to_sentences(self, text):
        """Split text into sentences using bling fire."""
        DELIMITERS = ["。", "！", "？", "，", "。", ",", "."]
        DELIMITERS_REGEX = "|".join(map(re.escape, DELIMITERS))
        text = text.replace("\n", "")
        sentences = []
        last_pos = 0

        # Iterate through the matches of the delimiters
        for match in re.finditer(DELIMITERS_REGEX, text):
            start, end = match.start(), match.end()
            sentence = text[last_pos:end]  # Include the delimiter in the sentence
            sentences.append(sentence)
            last_pos = end

        # Add the remaining part of the text, if any
        if last_pos < len(text):
            sentences.append(text[last_pos:])

        return sentences

    @property
    def synthesis_complete(self):
        return time.time() > self.synthesis_timeout


if __name__ == "__main__":
    audio_synthesiser = AudioSynthesiser()
    # filename = Path("output") / "synthesized" / (str(time.time()) + ".mp3")
    # absolute_path = os.path.abspath(filename)
    # audio_synthesiser.speech_synthesis_to_mp3("你好一二三", absolute_path)

    async def run_dummy():
        print(f"starting at {time.time()}")
        audio_synthesiser = AudioSynthesiser()
        audio_synthesiser.session_id = "test"
        asyncio.create_task(audio_synthesiser.dummy_text_receiver())
        asyncio.create_task(audio_synthesiser.process_text())
        # continue to run until synthesis is complete
        while True:
            if audio_synthesiser.synthesis_complete:
                print("SYNTHESIS COMPLETE")
                break
            await asyncio.sleep(0.1)

    asyncio.run(run_dummy())
