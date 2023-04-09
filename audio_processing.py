# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
from pathlib import Path
import io
from typing import List, Optional

from pydub import (
    AudioSegment,
    silence,
)
from fastlid import fastlid

from parameters import LANGUAGE_DETECT_CONFIDENCE, EXPECTED_LANGUAGE
from backend_functions import voice_to_text


def load_chunks_from_folder(folder_name: str) -> List[bytes]:
    """Load chunks from a folder.

    Args:
        folder_name: The name of the folder.

    Returns:
        A list of chunks.
    """
    chunks = []
    for file in Path(folder_name).iterdir():
        if file.is_file():
            with open(file, "rb") as f:
                chunks.append(f.read())
    return chunks


def join_webm_chunks_to_segment(chunks: List[bytes]) -> bytes:
    """Join all available chunks into a single byte string.

    Args:
        chunks (list): A list of byte strings.

    Returns:
        list: A list of io.BytesIO objects.
    """
    # include all previous chunks (the headers in the current chunk)
    joined_chunks = b"".join(chunks)
    # save the byte string to a different filename on disk
    joined_chunks_io = io.BytesIO(joined_chunks)
    # convert the byte string to an AudioSegment
    segment = AudioSegment.from_file(joined_chunks_io, format="webm")

    return segment


def split_segment_by_silence(
    segment: AudioSegment, silence_durations: int = 200, silence_thresh: int = -50
) -> list[AudioSegment]:
    """Remove silence from the beginning and end of each segment in a list of segments.

    Args:
        segments (AudioSegment): AudioSegment object.
        silence_durations (int, optional): Duration of silence required to trigger removal. Defaults to 200.
        silence_thresh (int, optional): Threshold in dBFS below which samples are considered silence. Defaults to -50.

    Returns:
        list[AudioSegment]: List of AudioSegment objects with silence removed from the beginning and end.
    """

    non_silent_ranges = silence.detect_nonsilent(
        segment, min_silence_len=silence_durations, silence_thresh=silence_thresh
    )
    # discard the range that ends together with the end of the segment
    non_silent_ranges = [r for r in non_silent_ranges if r[1] != len(segment)]
    split_segments = [segment[r[0] : r[1]] for r in non_silent_ranges]
    # print(f"Split {len(segment)} into {len(split_segments)} segments")

    return split_segments


def convert_chunks_to_segments(chunks: List[bytes]) -> List[AudioSegment]:
    """Load audio segments from chunks of bytes.

    Args:
        chunks: list of bytes

    Returns:
        list of AudioSegment
    """
    segments = []
    for i, chunk in enumerate(chunks):
        segments.append(AudioSegment.from_file(chunk, format="webm"))
    return segments


def load_segments_io(segments_io) -> List[AudioSegment]:
    """Load audio segments from a list of io.BytesIO objects.

    Args:
        segments_io: list of io.BytesIO objects

    Returns:
        list of AudioSegment
    """
    segments = []
    for i, segment_io in enumerate(segments_io):
        segments.append(AudioSegment.from_file(segment_io, format="webm"))
    return segments


def save_segments(segments: List[AudioSegment], folder_name: str) -> None:
    """Save a list of audio segments to a folder.

    Args:
        segments (List[AudioSegment]): The list of audio segments to save.
        folder_name (str): The name of the folder to save the audio segments to.
    """
    # save the audio segments to different files
    for i, segment in enumerate(segments):
        segment.export(f"{folder_name}/segment{i}.webm", format="webm")


def save_segments_to_io(segments: List[AudioSegment]) -> List[io.BytesIO]:
    """Save a list of audio segments to a list of io.BytesIO objects.

    Args:
        segments (List[AudioSegment]): The list of audio segments to save.

    Returns:
        List[io.BytesIO]: A list of io.BytesIO objects.
    """
    # save the audio segments to different files
    segments_io = []
    for i, segment in enumerate(segments):
        segment_io = io.BytesIO()
        segment_io.name = f"segment{i}.webm"
        segment.export(segment_io, format="webm")
        segments_io.append(segment_io)

    return segments_io


def remove_overlap(segments: List[AudioSegment]) -> List[AudioSegment]:
    """Remove overlapping segments in the list of segments.

    Args:
        segments: A list of audio segments.

    Returns:
        A list of audio segments with overlapping segments removed.
    """
    # Only select the duration of the segments without overlap
    segments = [
        segment[len(segments[i - 1]) :] if i > 0 else segment
        for i, segment in enumerate(segments)
    ]

    return segments


def remove_silence(
    segment: AudioSegment,
    silcence_durations: int = 200,
    silence_thresh: float = -50,
) -> list[AudioSegment]:
    """Remove silence from the beginning and end of each segment in a list of segments.

    Args:
        segments (list): List of AudioSegment objects.
        silence_durations (int, optional): Duration of silence required to trigger removal. Defaults to 200.
        silence_thresh (int, optional): Threshold in dBFS below which samples are considered silence. Defaults to -50.

    Returns:
        list[AudioSegment]: List of AudioSegment objects with silence removed from the beginning and end.
    """
    # remove the leading and trailing silence
    segment = silence.detect_silence(
        segment, min_silence_len=silcence_durations, silence_thresh=silence_thresh
    )
    # remove the 1st and
    return segment


def detect_break(
    segments: list[AudioSegment],
    silcence_durations: int = 200,
    silence_thresh: float = -50,
) -> bool:
    """Detects breaks in audio recording

    Args:
        segments (np.ndarray): Array of audio segments
        silence_durations (int): Duration of silence to detect a break
        silence_thresh (float): Threshold of silence to detect a break

    Returns:
        bool: the user stopped speaking
    """
    # if the trailing silence is longer than the silence duration, then the user stopped speaking
    trailing_silence = silence.detect_leading_silence(segments[-1], silence_thresh=-50)
    if trailing_silence > silcence_durations:
        return True


if __name__ == "__main__":
    chunks = load_chunks_from_folder("resources/chunks")
    segment = join_webm_chunks_to_segment(chunks)

    transcribed_segment_length = 0
    language = None
    for i, _ in enumerate(chunks):
        segment = join_webm_chunks_to_segment(chunks[: i + 1])
        split_segments = split_segment_by_silence(segment)
        # don't transcribe the segments that have already been transcribed
        split_segments = split_segments[transcribed_segment_length:]
        segments_io = save_segments_to_io(split_segments)
        for segment_io in segments_io:
            transcript = voice_to_text(segment_io, language=language)
            # use the language of the first segment
            if language is None:
                fastlid.set_languages = EXPECTED_LANGUAGE
                predicted_language = fastlid(transcript)
                # if the confidence is low, then don't set the language yet
                if predicted_language[1] > LANGUAGE_DETECT_CONFIDENCE:
                    language = predicted_language[0]
                    print("set language to", language)
            print(transcript)

        transcribed_segment_length += len(split_segments)

        if i == len(chunks) - 1:
            # save the audio segments to different files
            segments_io = save_segments_to_io([segment])
            transcript = voice_to_text(segments_io[0], language=language)
            print(transcript)
