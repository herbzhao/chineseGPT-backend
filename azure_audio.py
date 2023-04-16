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
        self.initial_timeout_length = 3600
        self.reset_timeout()
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

    def reset_timeout(self):
        self.timeout = time.time() + self.initial_timeout_length

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
                await asyncio.sleep(0.5)

    async def process_chunks(self):
        accumulated_chunks = b""
        while True:
            # get a chunk from the queue
            chunk = await self.chunks_queue.get()
            # store all the chunks received so far
            accumulated_chunks += chunk
            # convert chunks to audio segment
            try:
                complete_audio_segment = AudioSegment.from_file(
                    io.BytesIO(accumulated_chunks), format="webm"
                )
            except:
                print("Error converting audio segment")
                continue

            split_lengths = self.calculate_split_lengths(complete_audio_segment)
            # send the split audio segments to azure
            for j, (start, end) in enumerate(split_lengths):
                audio_segment = self.convert_audio_segment_to_wav(
                    complete_audio_segment[start:end],
                    append_silence_length=self.split_append_silence,
                )
                self.push_stream.write(audio_segment.raw_data)

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
            lambda evt: print("SESSION STARTED: {}".format(evt))
        )
        self.speech_recognizer.session_stopped.connect(
            lambda evt: print("CLOSING on {}".format(evt))
        )
        self.speech_recognizer.canceled.connect(
            lambda evt: print("CANCELED {}".format(evt))
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

    async def add_chunk(self, chunk):
        await self.chunks_queue.put(chunk)

    @property
    def transcription_complete(self):
        return time.time() > self.timeout

    async def close_session(self):
        # force timeout to kick in
        self.timeout = time.time() - self.timeout_length
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
        self.timeout = time.time() + self.timeout_length

    def recognized_callback(self, evt: speechsdk.SpeechRecognitionEventArgs):
        self.transcripts[-1] = evt.result.text
        self.transcripts.append("")

    async def stop_callback(self, evt: speechsdk.SessionEventArgs):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print("CLOSING on {}".format(evt))
        await self.speech_recognizer.stop_continuous_recognition()
        self.push_stream.close()


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

    def speech_synthesis_to_mp3_file(
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
        voice = "zh-CN-XiaochenNeural"
        speech_config.speech_synthesis_voice_name = voice

        # Receives a text from console input and synthesizes it to mp3 file.
        #  split text by comma and full stop to allow mp3 be sent earlier
        for i, text in enumerate(re.split(self.delimiters, input_text)):
            output_name = output_folder / f"{i}_{text}.mp3"
            file_config = speechsdk.audio.AudioOutputConfig(filename=str(output_name))
            self.speech_synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config, audio_config=file_config
            )

            result = self.speech_synthesizer.speak_text_async(text).get()
            # Check result
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print(
                    "Speech synthesized for text [{}], and the audio was saved to [{}]".format(
                        text, output_name
                    )
                )
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                print(
                    "Speech synthesis canceled: {}".format(cancellation_details.reason)
                )
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    print(
                        "Error details: {}".format(cancellation_details.error_details)
                    )

    def speech_synthesis_to_speaker(self) -> None:
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

        # Receives a text from console input and synthesizes it to speaker.
        while True:
            print("Enter some text that you want to speak, Ctrl-Z to exit")
            try:
                text = input()
            except EOFError:
                break
            result = speech_synthesizer.speak_text_async(text).get()
            # Check result
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print("Speech synthesized to speaker for text [{}]".format(text))
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                print(
                    "Speech synthesis canceled: {}".format(cancellation_details.reason)
                )
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    print(
                        "Error details: {}".format(cancellation_details.error_details)
                    )


if __name__ == "__main__":
    # async def main():
    # audio_transcripter = AudioTranscriber()
    # asyncio.run(audio_transcripter.run_dummy())

    audio_synthesiser = AudioSynthesiser()
    # audio_synthesiser.speech_synthesis_to_speaker()
    audio_synthesiser.speech_synthesis_to_mp3_file(
        "如果您的整个牙齿都呈现黄色，建议您咨询一位牙医，以了解如何最好地改善牙齿颜色。牙医可能会建议您接受美白治疗或其他牙齿美容程序。在此之前，您可以尝试使用含氢氧化物的漱口水和牙膏，以帮助减轻牙齿的黄色。此外，定期刷牙和使用牙线也是保持口腔卫生的重要步骤。"
    )
