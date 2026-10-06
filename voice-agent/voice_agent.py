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
OUTPUT_FILE = "response.mp3"


# --------------------------------------------------
# Setup
# --------------------------------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# ==================================================
# 1. SPEECH TO TEXT
# ==================================================

print("🎤 Speak now...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype=np.int16
)

sd.wait()

print("✅ Recording finished.")


# Save audio
with wave.open(AUDIO_FILE, "wb") as wav_file:

    wav_file.setnchannels(CHANNELS)
    wav_file.setsampwidth(2)
    wav_file.setframerate(SAMPLE_RATE)

    wav_file.writeframes(audio.tobytes())


# Send audio to STT
with open(AUDIO_FILE, "rb") as audio_file:

    transcription = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_file
    )


user_text = transcription.text

print("\n📝 You:")
print(user_text)


# ==================================================
# 2. LLM
# ==================================================

response = client.responses.create(
    model="gpt-6-luna",
    instructions=(
        "You are a helpful voice assistant. "
        "Keep your answers concise and conversational."
    ),
    input=user_text
)

agent_text = response.output_text

print("\n🤖 Agent:")
print(agent_text)


# ==================================================
# 3. TEXT TO SPEECH
# ==================================================

with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="alloy",
    input=agent_text
) as speech_response:

    speech_response.stream_to_file(OUTPUT_FILE)


print("\n🔊 Speech generated:")
print(OUTPUT_FILE)