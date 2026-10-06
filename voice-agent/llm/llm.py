import os

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()

# Create OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# User input
user_text = "What is Python?"


# Send text to the LLM
response = client.responses.create(
    model="gpt-6-luna",
    input=user_text
)


# Get the generated response
print("🤖 Agent:")
print(response.output_text)