# 🎙️ Basic Python Voice Agent

A beginner-friendly Voice Agent built with Python that follows a simple three-step architecture:

```text
🎤 Speech
    ↓
1. Speech-to-Text (STT)
    ↓
📝 Text
    ↓
2. LLM
    ↓
💬 Response
    ↓
3. Text-to-Speech (TTS)
    ↓
🔊 Speech
```

The purpose of this project is to understand the **fundamental architecture and working of a Voice Agent** without introducing unnecessary complexity.

---

## 🏗️ Architecture

```text
                         VOICE AGENT
                              │
                              ▼
                     🎤 User speaks
                              │
                              ▼
                  ┌─────────────────────┐
                  │ 1. SPEECH-TO-TEXT   │
                  │        (STT)        │
                  └──────────┬──────────┘
                             │
                             ▼
                       📝 User Text
                             │
                             ▼
                  ┌─────────────────────┐
                  │      2. LLM         │
                  │                     │
                  │ Understands input   │
                  │ and generates reply │
                  └──────────┬──────────┘
                             │
                             ▼
                     💬 Response Text
                             │
                             ▼
                  ┌─────────────────────┐
                  │ 3. TEXT-TO-SPEECH  │
                  │        (TTS)        │
                  └──────────┬──────────┘
                             │
                             ▼
                       🔊 Audio Output
                             │
                             ▼
                           USER
```

---

## 🔄 Data Flow

The complete data flow is:

```text
🎤 Microphone
     │
     ▼
┌───────────────┐
│      STT      │
│ Speech → Text │
└───────┬───────┘
        │
        ▼
📝 "What is Java?"
        │
        ▼
┌───────────────┐
│      LLM      │
│   Text → Text │
└───────┬───────┘
        │
        ▼
💬 "Java is a programming language..."
        │
        ▼
┌────────────────┐
│      TTS       │
│  Text → Audio  │
└────────┬───────┘
         │
         ▼
🔊 Agent speaks
```

---

## 📁 Project Structure

```text
voice-agent/
│
├── .env
├── .gitignore
├── requirements.txt
├── README.md
│
├── voice_agent.py
│
├── recording.wav
└── response.mp3
```

### Files

| File               | Description                                   |
| ------------------ | --------------------------------------------- |
| `voice_agent.py`   | Main Voice Agent application                  |
| `.env`             | Stores the API key                            |
| `.gitignore`       | Prevents sensitive/unnecessary files from Git |
| `requirements.txt` | Project dependencies                          |
| `recording.wav`    | Recorded user speech                          |
| `response.mp3`     | Generated agent speech                        |
| `README.md`        | Project documentation                         |

---

# 🧩 Core Components

## 1. Speech-to-Text (STT)

STT converts the user's spoken audio into text.

```text
🎤 "What is Python?"
        │
        ▼
       STT
        │
        ▼
📝 "What is Python?"
```

### Responsibility

```text
Audio → Text
```

The STT component does not generate the answer.

---

## 2. LLM

The LLM receives the text produced by STT and generates a response.

```text
📝 "What is Python?"
        │
        ▼
       LLM
        │
        ▼
💬 "Python is a high-level programming language..."
```

### Responsibility

```text
Text → Text
```

The LLM is responsible for understanding the user's request and generating a response.

---

## 3. Text-to-Speech (TTS)

TTS converts the LLM's response into audio.

```text
💬 "Python is a programming language..."
              │
              ▼
             TTS
              │
              ▼
        🔊 Audio Output
```

### Responsibility

```text
Text → Audio
```

---

# 🛠️ Technologies Used

| Technology    | Purpose                         |
| ------------- | ------------------------------- |
| Python        | Application development         |
| SoundDevice   | Microphone recording            |
| NumPy         | Audio data handling             |
| Wave          | WAV file creation               |
| OpenAI STT    | Speech-to-text                  |
| OpenAI LLM    | Response generation             |
| OpenAI TTS    | Text-to-speech                  |
| python-dotenv | Environment variable management |
| playsound3    | Playing generated audio         |

---

# ⚙️ Setup

## 1. Clone the Repository

```bash
git clone <repository-url>
cd voice-agent
```

---

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install sounddevice numpy openai python-dotenv playsound3
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

Never commit `.env` to Git.

---

# 🚫 .gitignore

Example `.gitignore`:

```gitignore
.venv/
.env
__pycache__/
*.pyc

recording.wav
response.mp3
```

---

# ▶️ Run the Application

Start the Voice Agent:

```bash
python voice_agent.py
```

The application will:

```text
1. 🎤 Record your voice
        ↓
2. 📝 Convert speech to text
        ↓
3. 🧠 Send text to the LLM
        ↓
4. 💬 Generate a response
        ↓
5. 🔊 Convert response to speech
        ↓
6. 🎧 Play the response
```

---

# 🧪 Example

### Step 1 — User speaks

```text
🎤 "What is Java?"
```

### Step 2 — STT

```text
📝 "What is Java?"
```

### Step 3 — LLM

```text
💬 "Java is a high-level, object-oriented programming language."
```

### Step 4 — TTS

```text
🔊 Agent speaks the response
```

---

# 🧠 Core Concept

The entire Voice Agent can be understood using three transformations:

```text
┌──────────────────────┐
│ Speech → Text        │
│       STT            │
└──────────────────────┘

           ↓

┌──────────────────────┐
│ Text → Text          │
│       LLM            │
└──────────────────────┘

           ↓

┌──────────────────────┐
│ Text → Speech        │
│       TTS            │
└──────────────────────┘
```

Or simply:

```text
🎤 AUDIO
   ↓
  STT
   ↓
📝 TEXT
   ↓
  LLM
   ↓
💬 TEXT
   ↓
  TTS
   ↓
🔊 AUDIO
```

---

# 🎯 Project Scope

This project intentionally focuses only on the three fundamental components:

```text
✅ Speech-to-Text
✅ LLM
✅ Text-to-Speech
```

The following are **not part of the current implementation**:

```text
❌ Voice Activity Detection (VAD)
❌ MCP
❌ Tools
❌ RAG
❌ Agent Memory
❌ Multi-Agent Systems
❌ WebSockets
❌ Complex Workflows
❌ Streaming Architecture
❌ Production Infrastructure
```

The goal is to first understand the basic Voice Agent architecture clearly.

---

# 📚 What This Project Teaches

By completing this project, you understand:

* How microphone audio is captured
* How audio is converted into text
* How text is sent to an LLM
* How an LLM generates a response
* How response text is converted into audio
* How the three components work together
* How a basic Voice Agent is structured

---

# 🚀 Final Architecture

```text
                    ┌──────────────────┐
                    │       USER       │
                    └────────┬─────────┘
                             │
                         🎤 Speech
                             │
                             ▼
                    ┌──────────────────┐
                    │       STT        │
                    │ Speech → Text    │
                    └────────┬─────────┘
                             │
                         📝 Text
                             │
                             ▼
                    ┌──────────────────┐
                    │       LLM        │
                    │  Text → Text     │
                    └────────┬─────────┘
                             │
                      💬 Response
                             │
                             ▼
                    ┌──────────────────┐
                    │       TTS        │
                    │ Text → Speech    │
                    └────────┬─────────┘
                             │
                         🔊 Audio
                             │
                             ▼
                    ┌──────────────────┐
                    │       USER       │
                    └──────────────────┘
```

---

## ⭐ One-Line Architecture

```text
🎤 Speech → STT → 📝 Text → LLM → 💬 Response → TTS → 🔊 Speech
```

This is the complete foundation of the basic Python Voice Agent.
