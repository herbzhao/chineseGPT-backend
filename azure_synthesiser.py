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


    def synthesis_to_mp3(
        self, input_text, output_folder=Path("resources") / "synthesized"
    ):
        """performs speech synthesis to a mp3 file"""
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
        # female voice
        # voice = "zh-CN-XiaochenNeural"
        # speech_config.speech_synthesis_voice_name = voice

        # Receives a text from console input and synthesizes it to mp3 file.
        #  split text by comma and full stop to allow mp3 be sent earlier
        #  save to io.BytesIO
        output_file = io.BytesIO()
        output_file.filename = "resources/output.mp3"

        output_name = output_folder / f"{input_text[:10]}.mp3"
        file_config = speechsdk.audio.AudioOutputConfig(filename=output_file.filename)
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=file_config
        )

        result = self.speech_synthesizer.speak_text_async(input_text).get()
        # Check result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"Speech synthesized and the audio was saved to [{output_name}]")
            # save bytesio to file
            with open(output_name, "wb") as f:
                f.write(output_file.getvalue())

        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))

        return output_name

    def speech_synthesis_to_speaker(self, input_text) -> None:
        """performs speech synthesis to the default speaker"""
        # Creates an instance of a speech config with specified subscription key and service region.
        speech_config = speechsdk.SpeechConfig(
            subscription=os.environ.get("SPEECH_KEY"),
            region=os.environ.get("SPEECH_REGION"),
        )
        # Creates a speech synthesizer using the default speaker as audio output.
        # The full list of supported languages can be found here:
        # https://docs.microsoft.com/azure/cognitive-services/speech-service/language-support#text-to-speech
        # https://aka.ms/csspeech/voicenames
        language = "zh-CN"
        speech_config.speech_synthesis_language = language
        # female voice
        voice = "zh-CN-XiaochenNeural"
        # voice = "zh-CN-XiaoxiaoNeural"
        # male voice
        # voice = "zh-CN-YunxiaNeural"

        speech_config.speech_synthesis_voice_name = voice

        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

        result = speech_synthesizer.speak_text_async(input_text).get()

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
        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=stream_config
        )

        # Receives a text from console input and synthesizes it to stream output.
        text = "你好，世界。"
        result = speech_synthesizer.speak_text_async(text).get()
        # Check result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(
                "Speech synthesized for text [{}], and the audio was written to output stream.".format(
                    text
                )
            )
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))
        # Destroys result which is necessary for destroying speech synthesizer
        del result

        # Destroys the synthesizer in order to close the output stream.
        del speech_synthesizer

        print(
            "Totally {} bytes received.".format(self.stream_callback.get_audio_size())
        )


if __name__ == "__main__":
    audio_synthesiser = AudioSynthesiser()

    audio_synthesiser.speech_synthesis_to_push_audio_output_stream()
    audio_synthesiser.stream_callback.save_to_file()
