from pydub import AudioSegment

print("loading mp3 file...")
segment = AudioSegment.from_file("recorded.mp3", "mp3")
print("loading wav file...")
segment = AudioSegment.from_file("recorded.wav", "wav")

print("Done!")
