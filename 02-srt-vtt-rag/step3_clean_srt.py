import re
from pathlib import Path

def clean_text(text):
    """
    Clean subtitle text before it is used for RAG.

    The goal is NOT to destroy useful information.
    We only remove obvious formatting/noise.
    """

    # -----------------------------------------------------
    # 1. Remove HTML/XML-style subtitle tags
    # -----------------------------------------------------
    #
    # Example:
    #
    # <i>Hello</i>
    #
    # becomes:
    #
    # Hello
    #
    text = re.sub(r"<[^>]+>", "", text)


    # -----------------------------------------------------
    # 2. Remove common subtitle sound-effect markers
    # -----------------------------------------------------
    #
    # Examples:
    #
    # [Music]
    # [Applause]
    # [Laughter]
    #
    # These usually don't contribute useful semantic
    # information for a technical RAG system.
    #
    text = re.sub(
        r"\[(music|applause|laughter|laughing|cheering)\]",
        "",
        text,
        flags=re.IGNORECASE
    )


    # -----------------------------------------------------
    # 3. Normalize multiple spaces
    # -----------------------------------------------------
    #
    # Example:
    #
    # "Spring    Boot     is"
    #
    # becomes:
    #
    # "Spring Boot is"
    #
    text = re.sub(r"\s+", " ", text)


    # -----------------------------------------------------
    # 4. Remove leading/trailing whitespace
    # -----------------------------------------------------

    text = text.strip()


    return text

test_text = "<i>Hello</i>     [Music]    Spring    Boot"

cleaned = clean_text(test_text)

# print("Original:")
# print(test_text)

# print("\nCleaned:")
# print(cleaned)  



# =========================================================
# STEP 1: Locate SRT file
# =========================================================

srt_path = Path("data/01_native-components-vs-core-components_epm.srt")



# =========================================================
# STEP 2: Check file
# =========================================================

if not srt_path.exists():
    print(f"SRT file not found: {srt_path}")
    exit()


# =========================================================
# STEP 3: Load SRT
# =========================================================

srt_text = srt_path.read_text(
    encoding="utf-8",
    errors="replace"
)


# =========================================================
# STEP 4: Split into subtitle blocks
# =========================================================

blocks = srt_text.strip().split("\n\n")


# =========================================================
# STEP 5: Store parsed subtitles
# =========================================================

subtitles = []


# =========================================================
# STEP 6: Parse every subtitle
# =========================================================

for block in blocks:

    lines = block.splitlines()


    # Ignore malformed blocks.
    if len(lines) < 3:
        continue


    # Subtitle number
    subtitle_number = int(lines[0])


    # Timestamp
    timestamp = lines[1]


    # Separate start/end timestamps
    start_time, end_time = timestamp.split(" --> ")


    # -----------------------------------------------------
    # Subtitle text
    # -----------------------------------------------------
    #
    # lines[2:] is used because subtitle text can occupy
    # multiple lines.
    #
    subtitle_text = " ".join(lines[2:])


    # -----------------------------------------------------
    # Clean subtitle text
    # -----------------------------------------------------

    subtitle_text = clean_text(subtitle_text)


    # -----------------------------------------------------
    # Create structured subtitle object
    # -----------------------------------------------------

    subtitle = {
        "index": subtitle_number,
        "start": start_time,
        "end": end_time,
        "text": subtitle_text
    }


    # Store it
    subtitles.append(subtitle)


    # =========================================================
# STEP 7: Display results
# =========================================================

print(f"Total subtitles parsed: {len(subtitles)}")


for subtitle in subtitles:

    print("\n-----------------------------")

    print("Index :", subtitle["index"])
    print("Start :", subtitle["start"])
    print("End   :", subtitle["end"])
    print("Text  :", subtitle["text"])