import sounddevice as sd
import numpy as np
import wave

SAMPLE_RATE = 16000
DURATION = 5
CHANNELS = 1

print("Speak now...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype=np.int16
)

sd.wait()

print("Recording finished.")

with wave.open("recording.wav", "wb") as wav_file:
    wav_file.setnchannels(CHANNELS)
    wav_file.setsampwidth(2)
    wav_file.setframerate(SAMPLE_RATE)
    wav_file.writeframes(audio.tobytes())

print("Saved as recording.wav")