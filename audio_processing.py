# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
from pathlib import Path
import io
from typing import List, Optional
import uuid
from pydub import (
    AudioSegment,
    silence,
)
from fastlid import fastlid

from parameters import (
    LANGUAGE_DETECT_CONFIDENCE,
    EXPECTED_LANGUAGE,
    SENTENCE_BREAK_DURATION,
    SILENCE_THRESHOLD,
    SILENCE_SPLIT_DURATION,
)
from backend_functions import voice_to_text, voice_to_text_async


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
    segment: AudioSegment,
    silence_durations: int = SILENCE_SPLIT_DURATION,
    silence_thresh: int = SILENCE_THRESHOLD,
    maximum_length: int = 2000,
) -> list[AudioSegment]:
    """Remove silence from the beginning and end of each segment in a list of segments.

    Args:
        segments (AudioSegment): AudioSegment object.
        silence_durations (int, optional): Duration of silence required to trigger removal. Defaults to 200.
        silence_thresh (int, optional): Threshold in dBFS below which samples are considered silence. Defaults to -50.
        maximum_length (int, optional): Maximum length of a segment. Defaults to 2000. will break up even if no silence is detected.

    Returns:
        list[AudioSegment]: List of AudioSegment objects with silence removed from the beginning and end.
    """

    non_silent_ranges = silence.detect_nonsilent(
        segment, min_silence_len=silence_durations, silence_thresh=silence_thresh
    )

    # discard the range that ends together with the end of the segment
    non_silent_ranges = [r for r in non_silent_ranges if r[1] != len(segment)]

    split_segments = [segment[r[0] : r[1]] for r in non_silent_ranges]
    # print(f"Split {len(segment)} into {len(split_segments)} se gments")

    return split_segments, non_silent_ranges


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
        # generate a random uuid
        segment_io.name = f"{str(uuid.uuid4())}.webm"
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
    silence_thresh: int = SILENCE_THRESHOLD,
) -> list[AudioSegment]:
    """Remove silence from the beginning and end of each segment in a list of segments.

    Args:
        segments (list): List of AudioSegment objects.
        silence_durations (int, optional): Duration of silence required to trigger removal. Defaults to 200.
        silence_thresh (int, optional): Threshold in dBFS below which samples are considered silence. Defaults to -50.

    Returns:
        list[AudioSegment]: List of AudioSegment objects with silence removed from the beginning and end.
    """
    # join up the non-silent segments
    segments = silence.detect_nonsilent()
    # remove the 1st and
    return segment


def detect_audio_stop(
    non_silent_ranges, segment_length, silence_durations=SENTENCE_BREAK_DURATION
):
    # silent ranges are the ranges that are not in the non_silent_ranges
    silent_ranges = []
    for i, r in enumerate(non_silent_ranges):
        if i == 0:
            # silent_ranges.append((0, r[0]))
            pass  # ignore the first range
        elif i == len(non_silent_ranges) - 1:
            silent_ranges.append((non_silent_ranges[i - 1][1], r[0]))
            silent_ranges.append((r[1], segment_length))
        else:
            silent_ranges.append((non_silent_ranges[i - 1][1], r[0]))

    if len(silent_ranges) > 0:
        if silent_ranges[-1][1] - silent_ranges[-1][0] > silence_durations:
            return True

    return False


def detect_audio_stop_old(
    segment: AudioSegment,
    silcence_durations: int = SENTENCE_BREAK_DURATION,
    silence_thresh: int = SILENCE_THRESHOLD,
) -> bool:
    """Detects breaks in audio recording

    Args:
        segments (np.ndarray): Array of audio segments
        silence_durations (int): Duration of silence to detect a break
        silence_thresh (float): Threshold of silence to detect a break

    Returns:
        bool: the user stopped speaking
    """
    # if there is any silence is longer than the silence duration, then the user stopped speaking
    silcence_segments = silence.detect_nonsilent(
        segment, silence_thresh=silence_thresh, min_silence_len=silcence_durations
    )
    # make sure the silence is not at the beginning of the segment
    silcence_segments = [s for s in silcence_segments if s[0] != 0]
    return True if len(silcence_segments) > 0 else False


def transcribing_chunks(chunks, transcribed_segment_length=0):
    segment = join_webm_chunks_to_segment(chunks)
    split_segments, non_silent_ranges = split_segment_by_silence(segment)
    # check if the segment contains end of speech
    if detect_audio_stop(non_silent_ranges, len(segment)):
        # combine the split segments into one segment
        segment = sum(split_segments)
        segment_io = save_segments_to_io([segment])[0]
        # save the segment to a file
        save_segments([segment], "resources/chunks/segments")
        print("Transcribing the entire segment")
        transcripts = [voice_to_text(segment_io)]
        stop_transcribing = True
    else:
        # don't transcribe the segments that have already been transcribed
        split_segments = split_segments[transcribed_segment_length:]
        segments_io = save_segments_to_io(split_segments)
        transcripts = []
        for segment_io in segments_io:
            transcripts.append(voice_to_text(segment_io))
        transcribed_segment_length += len(split_segments)
        stop_transcribing = False
    return transcripts, transcribed_segment_length, stop_transcribing


async def transcribing_chunks_async(chunks, transcribed_segment_length=0):
    segment = join_webm_chunks_to_segment(chunks)
    split_segments, non_silent_ranges = split_segment_by_silence(segment)
    # check if the segment contains end of speech
    if detect_audio_stop(non_silent_ranges, len(segment)):
        # combine the split segments into one segment
        segment = sum(split_segments)
        segment_io = save_segments_to_io([segment])[0]
        # save the segment to a file
        save_segments([segment], "resources/chunks/segments")
        print("Transcribing the entire segment")
        transcripts = [await voice_to_text_async(segment_io)]
        stop_transcribing = True
    else:
        # don't transcribe the segments that have already been transcribed
        split_segments = split_segments[transcribed_segment_length:]
        segments_io = save_segments_to_io(split_segments)
        transcripts = []
        for segment_io in segments_io:
            transcripts.append(await voice_to_text_async(segment_io))
        transcribed_segment_length += len(split_segments)
        stop_transcribing = False
    return transcripts, transcribed_segment_length, stop_transcribing


if __name__ == "__main__":
    chunks = load_chunks_from_folder("resources/chunks")
    transcribed_segment_length = 0
    for i, _ in enumerate(chunks):
        (
            transcripts,
            transcribed_segment_length,
            stop_transcribing,
        ) = transcribing_chunks(chunks[: i + 1], transcribed_segment_length)
        print(transcripts)
        if stop_transcribing:
            break
