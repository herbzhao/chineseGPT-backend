import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import time
import pydub
import time
import threading
import wave

load_dotenv()
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
if os.path.exists(".env.production") and os.getenv("ENVIRONMENT") == "production":
    print("GETTING PRODUCTION ENVIRONMENT VARIABLES")
    load_dotenv(".env.production")


def stream_recognize_async():
    speech_config = speechsdk.SpeechConfig(
        subscription=os.environ.get("SPEECH_KEY"),
        region=os.environ.get("SPEECH_REGION"),
    )
    speech_config.speech_recognition_language = "zh-CN"
    # Create a PushAudioInputStream object
    push_stream = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

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

    speech_recognizer.recognizing.connect(
        lambda evt: print("RECOGNIZING: {}".format(evt))
    )
    speech_recognizer.recognized.connect(
        lambda evt: print("RECOGNIZED: {}".format(evt))
    )
    speech_recognizer.session_started.connect(
        lambda evt: print("SESSION STARTED: {}".format(evt))
    )
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(lambda evt: print("CANCELED {}".format(evt)))

    # Start continuous recognition
    speech_recognizer.start_continuous_recognition_async()

    with open("resources/9ede5220-d7c6-11ed-baaf-7705498ae62c.wav", "rb") as f:
        audio = pydub.AudioSegment.from_file(f)
        for i in range(0, len(audio), 2000):
            push_stream.write(audio[i : i + 2000].raw_data)
            time.sleep(2.2)

    # Stop continuous recognition
    speech_recognizer.stop_continuous_recognition_async()

    # Close the PushAudioInputStream object
    push_stream.close()


if __name__ == "__main__":
    stream_recognize_async()
