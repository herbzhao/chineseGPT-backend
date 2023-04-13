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
        self.split_length = 1000
        self.split_append_silence = 100
        self.end_of_stream_silence = 2000

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

    def chunk_receiver(self, folder_name):
        # mimic the behaviour of receiving chunk every 2.2 seconds
        # and append chunk to a growing stream
        # open files in the folder
        # for each file, read the file and append to the stream
        filenames = [
            filename for filename in Path(folder_name).iterdir() if filename.is_file()
        ]
        combined_audio_segment = AudioSegment.empty()
        for i, filename in enumerate(filenames):
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
            # Take care of the case where the audio segment is shorter than 1 second in the end
            if i == len(filenames) - 1:
                divided_lengths.append([dividers[-1], len(audio_segment)])

            print(divided_lengths)
            for j, (start, end) in enumerate(divided_lengths):
                webm_file = io.BytesIO()
                audio_segment[start:end].export(webm_file, format="webm")
                self.audio_segments.append(audio_segment[start:end])
                combined_audio_segment += audio_segment[start:end]

                append_silence_length = (
                    self.end_of_stream_silence
                    if ((j == len(divided_lengths) - 1) and (i == len(filenames) - 1))
                    else self.split_append_silence
                )

                self.convert_webm_to_wav(
                    webm_file,
                    f"resources/chunks/split/{len(self.audio_segments)}.wav",
                    append_silence_length=append_silence_length,
                )

            self.consumed_segment_length = dividers[-1]

        # combine the split audio segments into one audio segment for debugging
        combined_audio_segment = combined_audio_segment.set_frame_rate(16000)
        combined_audio_segment = combined_audio_segment.set_channels(1)
        combined_audio_segment = combined_audio_segment.set_sample_width(2)
        combined_audio_segment.export(
            "resources/chunks/complete/combined_audio_segment.wav", format="wav"
        )

        #  save the entire chunks as a file for debugging
        # save the output to a bytesio
        wav_file = io.BytesIO()
        output_file = self.convert_webm_to_wav(
            io.BytesIO(self.chunks),
            wav_file,
            append_silence_length=self.end_of_stream_silence,
        )
        output_file = self.convert_webm_to_wav(
            io.BytesIO(self.chunks),
            "resources/chunks/complete/entire_chunks.wav",
            append_silence_length=self.end_of_stream_silence,
        )

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
            print("RECOGNIZING: {}".format(evt))

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

        # 1: send entire audio at once
        # with open("resources/chunks/complete/entire_chunks.wav", "rb") as f:
        #     audio = AudioSegment.from_file(f)
        #     self.push_stream.write(audio.raw_data)
        #     time.sleep(5)

        # 2: split entire audio into chunks
        # with open("resources/chunks/complete/entire_chunks.wav", "rb") as f:
        #     audio = AudioSegment.from_file(f)
        #     for i in range(0, len(audio), 1000):
        #         self.push_stream.write(audio[i : i + 1000].raw_data)
        #         # write to a file for debugging
        #         audio[i : i + 1000].export(
        #             f"resources/chunks/split/2nd way_{i}.wav", format="wav"
        #         )
        #         time.sleep(2)

        # 3: read chunks
        # filenames = [
        #     filename
        #     for filename in Path("resources/chunks/split").iterdir()
        #     if filename.is_file()
        # ]
        # # sort the filenames by the number in the filename
        # filenames = sorted(filenames, key=lambda x: int(x.stem))
        # for filename in filenames:
        #     with open(filename, "rb") as f:
        #         print(filename)
        #         audio = AudioSegment.from_file(f)
        #         self.push_stream.write(audio.raw_data)
        #         time.sleep(2)

        # Stop continuous recognition
        speech_recognizer.stop_continuous_recognition_async()

        # Close the PushAudioInputStream object
        self.push_stream.close()


if __name__ == "__main__":
    # async def main():
    audio_transcripter = AudioTranscripter()
    audio_transcripter.chunk_receiver("resources/chunks")
    audio_transcripter.stream_recognize_async()
    # await audio_transcripter.chunk_receiver(r"resources/chunks")

    # asyncio.run(main())
