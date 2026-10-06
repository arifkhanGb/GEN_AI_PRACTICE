import os

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# # Text we want the agent to speak
# text = "Hello. I am your voice assistant."
llm_response = response.output_text

tts_text = llm_response


# Generate speech
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="alloy",
    input=tts_text
) as response:

    response.stream_to_file("response.mp3")


print("🔊 Speech generated: response.mp3")