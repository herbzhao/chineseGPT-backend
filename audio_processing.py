# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
from pathlib import Path
import io
from typing import List, Optional

from pydub import (
    AudioSegment,
    silence,
)

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


def join_webm_chunks(chunks: List[bytes]) -> bytes:
    """Join chunks is a function that takes a list of byte strings as input, and returns a list of io.BytesIO objects as output.

    Args:
        chunks (list): A list of byte strings.

    Returns:
        list: A list of io.BytesIO objects.
    """
    joined_chunks_io = []
    # include all previous chunks (the headers in the current chunk)
    joined_chunks = [b"".join(chunks[: i + 1]) for i, _ in enumerate(chunks)]
    # save the byte string to a different filename on disk
    for i, chunk in enumerate(joined_chunks):
        # save the byte string to io.BytesIO
        joined_chunks_io.append(io.BytesIO(chunk))

    return joined_chunks_io


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
    segments: list, silence_durations: int = 200, silence_thresh: int = -50
) -> list[AudioSegment]:
    """Remove silence from the beginning and end of each segment in a list of segments.

    Args:
        segments (list): List of AudioSegment objects.
        silence_durations (int, optional): Duration of silence required to trigger removal. Defaults to 200.
        silence_thresh (int, optional): Threshold in dBFS below which samples are considered silence. Defaults to -50.

    Returns:
        list[AudioSegment]: List of AudioSegment objects with silence removed from the beginning and end.
    """
    # use split_on_silence to split the audio segments into smaller segments using the last semgent
    split_segments = silence.split_on_silence(
        segments[-1], min_silence_len=silence_durations, silence_thresh=silence_thresh
    )

    return split_segments


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
    joined_chunks_io = join_webm_chunks(chunks)
    segments = convert_chunks_to_segments(joined_chunks_io)
    save_segments(segments, "resources/chunks/segments")
    non_overlap_segments = remove_overlap(segments)
    save_segments(non_overlap_segments, "resources/chunks/non_overlap")
    segments_io = save_segments_to_io(non_overlap_segments)
    for segment_io in segments_io:
        voice_to_text(segment_io)

    segments_io = save_segments_to_io(segments)
    # run against the complete segment
    voice_to_text(segments_io[-1])

    # segments = remove_silence(segments)
    # save_segments(segments, "resources/chunks/silence_split")
    # voice_to_text()
