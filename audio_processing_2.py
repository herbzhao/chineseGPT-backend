import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import time
from pydub import AudioSegment
import time
import threading
import wave
from pathlib import Path
import threading

load_dotenv()
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
if os.path.exists(".env.production") and os.getenv("ENVIRONMENT") == "production":
    print("GETTING PRODUCTION ENVIRONMENT VARIABLES")
    load_dotenv(".env.production")


# read chunks from folder
def read_chunks_from_folder(folder):
    chunks = []
    for file in os.listdir(folder):
        if file.endswith(".webm"):
            with open(os.path.join(folder, file), "rb") as f:
                chunks.append(f.read())

    # join the 1st chunk with the rest of chunks
    first_chunk = chunks[0]
    chunks = [b"".join([first_chunk, chunk]) for chunk in chunks[1:]]
    # save chunks to file
    for i, chunk in enumerate(chunks):
        with open(f"resources/chunks/append/{i}_append.webm", "wb") as f:
            f.write(chunk)

    return chunks


def push_stream_writer(stream):
    # The number of bytes to push per buffer
    n_bytes = 6400
    wav_fh = wave.open("resources/chunks/backup/whatstheweatherlike.wav", "rb")
    # start pushing data until all data has been read from the file
    try:
        while True:
            frames = wav_fh.readframes(n_bytes // 2)
            print(f"{time.time()}: sending bytes")
            # print("read {} bytes".format(len(frames)))
            if not frames:
                break
            stream.write(frames)
            time.sleep(0.1)
    finally:
        wav_fh.close()
        stream.close()  # must be done to signal the end of stream


# check azure lowest latency for speech to text
def push_stream_writer(stream):
    chunks = []
    folder = "resources/chunks"
    for file in os.listdir(folder):
        if file.endswith(".webm"):
            with open(os.path.join(folder, file), "rb") as f:
                chunks.append(f.read())
    # start pushing data until all data has been read from the file
    try:
        for chunk in chunks:
            stream.write(chunk)
            time.sleep(0.01)
    finally:
        stream.close()  # must be done to signal the end of stream


def speech_recognition_with_push_stream():
    """gives an example how to use a push audio stream to recognize speech from a custom audio
    source"""
    speech_config = speechsdk.SpeechConfig(
        subscription=os.environ.get("SPEECH_KEY"),
        region=os.environ.get("SPEECH_REGION"),
    )
    speech_config.speech_recognition_language = "zh-CN"

    # setup the audio stream
    stream = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    # instantiate the speech recognizer with push stream input
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )
    recognition_done = threading.Event()

    # Connect callbacks to the events fired by the speech recognizer
    def session_stopped_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print("SESSION STOPPED: {}".format(evt))
        recognition_done.set()

    speech_recognizer.recognizing.connect(
        lambda evt: print(f"RECOGNIZING: {evt} at {time.time()}")
    )
    speech_recognizer.recognized.connect(
        lambda evt: print(f"RECOGNIZED: {evt} at {time.time()}")
    )
    speech_recognizer.session_started.connect(
        lambda evt: print("SESSION STARTED: {}".format(evt))
    )
    speech_recognizer.session_stopped.connect(session_stopped_cb)
    speech_recognizer.canceled.connect(lambda evt: print("CANCELED {}".format(evt)))

    # start push stream writer thread
    push_stream_writer_thread = threading.Thread(
        target=push_stream_writer, args=[stream]
    )
    push_stream_writer_thread.start()

    # start continuous speech recognition
    speech_recognizer.start_continuous_recognition()

    # wait until all input processed
    recognition_done.wait()

    # stop recognition and clean up
    speech_recognizer.stop_continuous_recognition()
    push_stream_writer_thread.join()


if __name__ == "__main__":
    speech_recognition_with_push_stream()
