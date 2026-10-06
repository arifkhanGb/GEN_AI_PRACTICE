import asyncio
import io
import os
import threading
import time

import numpy as np
import pygame
import sounddevice as sd
import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

LLM_MODEL = "gpt-5.6-luna"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "coral"

BARGE_IN_RMS_THRESHOLD = 0.10
BARGE_IN_CONSECUTIVE_FRAMES = 5


# ============================================================
# GLOBAL STATE
# ============================================================

class VoiceAgentState:

    def __init__(self):
        self.audio_queue = asyncio.Queue()

        self.current_audio = None

        self.is_playing = False

        # Used to invalidate old TTS requests
        self.playback_session = 0

        self.lock = threading.Lock()


state = VoiceAgentState()


# ============================================================
# 1. SPEECH TO TEXT
# ============================================================

recognizer = sr.Recognizer()


def listen_for_speech():

    with sr.Microphone() as source:

        print("\n🎤 Listening...")

        recognizer.adjust_for_ambient_noise(
            source,
            duration=0.5
        )

        audio = recognizer.listen(
            source
        )

    print("🧠 Converting speech to text...")

    try:

        # Uses Google's speech recognition service
        text = recognizer.recognize_google(audio)

        return text

    except sr.UnknownValueError:

        print("❌ Could not understand speech.")

        return ""

    except sr.RequestError as error:

        print(f"❌ STT error: {error}")

        return ""


# ============================================================
# 2. STREAMING LLM
# ============================================================

async def llm_streaming(user_text):

    print("\n🤖 LLM started...")

    response = client.responses.create(

        model=LLM_MODEL,

        instructions=(
            "You are a helpful voice assistant. "
            "Answer conversationally. "
            "Keep responses concise. "
            "Prefer complete sentences because each sentence "
            "will be converted to speech."
        ),

        input=user_text,

        stream=True
    )

    sentence_buffer = ""
    full_response = ""

    for event in response:

        if event.type == "response.output_text.delta":

            delta = event.delta

            full_response += delta
            sentence_buffer += delta

            sentences, sentence_buffer = extract_sentences(
                sentence_buffer
            )

            for sentence in sentences:

                yield {
                    "text": sentence,
                    "is_final": False
                }

    # Handle leftover text
    leftover = sentence_buffer.strip()

    if leftover:

        yield {
            "text": leftover,
            "is_final": True
        }


def extract_sentences(text):

    sentences = []

    while True:

        sentence_end = -1

        for punctuation in [".", "?", "!"]:

            index = text.find(punctuation)

            if index != -1:

                if sentence_end == -1:
                    sentence_end = index
                else:
                    sentence_end = min(
                        sentence_end,
                        index
                    )

        if sentence_end == -1:
            break

        sentence = text[:sentence_end + 1].strip()

        text = text[sentence_end + 1:]

        if sentence:
            sentences.append(sentence)

    return sentences, text


# ============================================================
# 3. TEXT TO SPEECH
# ============================================================

async def generate_speech(text):

    current_session = state.playback_session

    print(f"🔊 TTS: {text}")

    response = client.audio.speech.create(

        model=TTS_MODEL,

        voice=TTS_VOICE,

        input=text,

        instructions=(
            "Speak in a cheerful, warm and natural tone."
        )
    )

    audio_bytes = response.read()

    # If user interrupted the agent while TTS
    # was being generated, discard this audio.
    if current_session != state.playback_session:

        print("🛑 Discarding old TTS audio.")

        return

    await state.audio_queue.put(
        (
            current_session,
            audio_bytes
        )
    )


# ============================================================
# AUDIO PLAYBACK WORKER
# ============================================================

async def audio_playback_worker():

    while True:

        session, audio_bytes = await state.audio_queue.get()

        try:

            if session != state.playback_session:

                continue

            state.is_playing = True

            await play_audio(audio_bytes)

        finally:

            state.is_playing = False

            state.audio_queue.task_done()


async def play_audio(audio_bytes):

    audio_file = io.BytesIO(audio_bytes)

    pygame.mixer.music.load(
        audio_file,
        "mp3"
    )

    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():

        # Allow other async tasks to run
        await asyncio.sleep(0.05)


# ============================================================
# INTERRUPT PLAYBACK
# ============================================================

def interrupt_playback():

    print("🛑 Interrupting agent playback...")

    # Create a new playback session.
    #
    # Any TTS request belonging to the old session
    # will now be considered invalid.
    state.playback_session += 1

    # Stop currently playing audio
    pygame.mixer.music.stop()

    state.is_playing = False

    # Clear queued audio
    clear_audio_queue()


def clear_audio_queue():

    while not state.audio_queue.empty():

        try:

            state.audio_queue.get_nowait()

            state.audio_queue.task_done()

        except asyncio.QueueEmpty:

            break


# ============================================================
# BARGE-IN MONITOR
# ============================================================

class BargeInMonitor:

    def __init__(self):

        self.running = False

        self.loud_frames = 0

        self.loop = None

    def start(self, loop):

        self.loop = loop
        self.running = True

        self.stream = sd.InputStream(
            channels=1,
            samplerate=16000,
            blocksize=1024,
            callback=self.audio_callback
        )

        self.stream.start()

        print("🎧 Barge-in monitor started.")

    def stop(self):

        self.running = False

        if hasattr(self, "stream"):

            self.stream.stop()
            self.stream.close()

    def audio_callback(
        self,
        indata,
        frames,
        time_info,
        status
    ):

        if not self.running:
            return

        # Only monitor microphone while
        # the agent is speaking.
        if not state.is_playing:

            self.loud_frames = 0

            return

        audio = indata[:, 0]

        rms = np.sqrt(
            np.mean(
                np.square(audio)
            )
        )

        if rms >= BARGE_IN_RMS_THRESHOLD:

            self.loud_frames += 1

            if (
                self.loud_frames
                >= BARGE_IN_CONSECUTIVE_FRAMES
            ):

                print(
                    f"🛑 Barge-in detected "
                    f"(RMS={rms:.3f})"
                )

                self.loud_frames = 0

                # Callback runs in another thread.
                # Schedule interrupt safely on asyncio loop.
                self.loop.call_soon_threadsafe(
                    interrupt_playback
                )

        else:

            self.loud_frames = 0


# ============================================================
# MAIN VOICE AGENT
# ============================================================

async def main():

    pygame.mixer.init()

    playback_worker = asyncio.create_task(
        audio_playback_worker()
    )

    loop = asyncio.get_running_loop()

    barge_in_monitor = BargeInMonitor()

    barge_in_monitor.start(loop)

    print("\n===================================")
    print("🎙️ Python Voice Agent")
    print("===================================")

    try:

        while True:

            # ----------------------------------------
            # STT
            # ----------------------------------------

            user_text = await asyncio.to_thread(
                listen_for_speech
            )

            if not user_text:

                continue

            print(f"\n👤 User: {user_text}")

            # ----------------------------------------
            # Stop previous agent response
            # ----------------------------------------

            interrupt_playback()

            # ----------------------------------------
            # LLM STREAMING
            # ----------------------------------------

            async for chunk in llm_streaming(
                user_text
            ):

                sentence = chunk["text"]

                print(
                    f"🤖 Agent: {sentence}"
                )

                # ------------------------------------
                # TTS
                # ------------------------------------

                asyncio.create_task(
                    generate_speech(sentence)
                )

    except KeyboardInterrupt:

        print("\n👋 Stopping voice agent...")

    finally:

        barge_in_monitor.stop()

        playback_worker.cancel()

        pygame.mixer.quit()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())