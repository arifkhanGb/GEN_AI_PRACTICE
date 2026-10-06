import os

from dotenv import load_dotenv
from openai import OpenAI


# Load variables from .env
load_dotenv()

# Create OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# Open the recorded audio file
with open("recording.wav", "rb") as audio_file:

    # Send audio to the transcription model
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_file
    )


# Print the generated text
print("You said:")
print(transcription.text)