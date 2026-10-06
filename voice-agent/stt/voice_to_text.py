import os
import wave

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SAMPLE_RATE = 16000
DURATION = 5
CHANNELS = 1
AUDIO_FILE = "recording.wav"


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# --------------------------------------------------
# Step 1: Record audio
# --------------------------------------------------

print("🎤 Speak now...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype=np.int16
)

sd.wait()

print("✅ Recording finished.")


# --------------------------------------------------
# Step 2: Save audio
# --------------------------------------------------

with wave.open(AUDIO_FILE, "wb") as wav_file:

    wav_file.setnchannels(CHANNELS)
    wav_file.setsampwidth(2)
    wav_file.setframerate(SAMPLE_RATE)

    wav_file.writeframes(audio.tobytes())


print(f"💾 Audio saved to {AUDIO_FILE}")


# --------------------------------------------------
# Step 3: Send audio to STT
# --------------------------------------------------

print("🧠 Converting speech to text...")

with open(AUDIO_FILE, "rb") as audio_file:

    transcription = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_file
    )


# --------------------------------------------------
# Step 4: Display text
# --------------------------------------------------

print("\n📝 You said:")
print(transcription.text)


user_text = transcription.text

response = client.responses.create(
    model="gpt-6-luna",
    instructions=(
        "You are a helpful voice assistant. "
        "Keep your answers concise and conversational."
    ),
    input=user_text
)

print("\n🤖 Agent:")
print(response.output_text)