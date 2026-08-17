import re
from pathlib import Path

def timestamp_to_seconds(timestamp):
    """
    Convert an SRT timestamp into total seconds.

    Example:

    60 + 23 + 0.450 = 83.450 seconds

    00:01:23,450

    becomes:

    83.45
    """

    # -----------------------------------------------------
    # SRT uses a comma between seconds and milliseconds:
    #
    # 00:01:23,450
    #
    # Python's float representation normally uses a dot:
    #
    # 23.450
    #
    # So replace the comma with a dot.
    # -----------------------------------------------------

    timestamp = timestamp.replace(",", ".")


    # -----------------------------------------------------
    # Split:
    #
    # HH:MM:SS.mmm
    #
    # into:
    #
    # HH
    # MM
    # SS.mmm
    # -----------------------------------------------------

    hours, minutes, seconds = timestamp.split(":")


    # Convert each part to numbers.
    hours = int(hours)
    minutes = int(minutes)
    seconds = float(seconds)


    # -----------------------------------------------------
    # Convert everything into seconds.
    # -----------------------------------------------------

    total_seconds = (
        hours * 3600
        + minutes * 60
        + seconds
    )


    return total_seconds

# timestamp = "00:01:23,450"

# seconds = timestamp_to_seconds(timestamp)

# print(seconds)

# Test multiple timestamps
timestamps = [
    "00:00:00,000",
    "00:00:01,500",
    "00:01:00,000",
    "00:01:23,450",
    "01:00:00,000",
    "01:10:25,750"
]

for timestamp in timestamps:

    seconds = timestamp_to_seconds(timestamp)

    print(timestamp, "=>", seconds)


def clean_text(text):

    # Remove HTML/XML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove common sound markers
    text = re.sub(
        r"\[(music|applause|laughter|laughing|cheering)\]",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

 # ---------------------------------------------------------
# Load SRT
# ---------------------------------------------------------

srt_path = Path("data/01_native-components-vs-core-components_epm.srt")

srt_text = srt_path.read_text(
    encoding="utf-8",
    errors="replace"
)


# ---------------------------------------------------------
# Split into blocks
# ---------------------------------------------------------

blocks = srt_text.strip().split("\n\n")


# ---------------------------------------------------------
# Parse subtitles
# ---------------------------------------------------------

subtitles = []


for block in blocks:

    lines = block.splitlines()

    if len(lines) < 3:
        continue

    subtitle_number = int(lines[0])

    timestamp = lines[1]

    start_time, end_time = timestamp.split(" --> ")

    subtitle_text = " ".join(lines[2:])

    subtitle_text = clean_text(subtitle_text)

    # Convert timestamps to seconds
    start_seconds = timestamp_to_seconds(start_time)
    end_seconds = timestamp_to_seconds(end_time)

    subtitle = {
        "index": subtitle_number,

        # Human-readable timestamps
        "start": start_time,
        "end": end_time,

        # Machine-friendly timestamps
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,

        # Cleaned text
        "text": subtitle_text
    }

    subtitles.append(subtitle)


# ---------------------------------------------------------
# Inspect results
# ---------------------------------------------------------

for subtitle in subtitles:

    duration = (
        subtitle["end_seconds"]
        - subtitle["start_seconds"]
    )

    print("\n-----------------------------")

    print("Index          :", subtitle["index"])
    print("Start          :", subtitle["start"])
    print("End            :", subtitle["end"])
    print("Start seconds  :", subtitle["start_seconds"])
    print("End seconds    :", subtitle["end_seconds"])
    print("Duration       :", duration)
    print("Text           :", subtitle["text"])


# Now create the reverse function

def seconds_to_timestamp(total_seconds):
    """
    Convert seconds back into SRT timestamp format.

    Example:

    83.45

    becomes:

    00:01:23,450
    """

    # -----------------------------------------------------
    # Calculate hours
    # -----------------------------------------------------

    hours = int(total_seconds // 3600)


    # -----------------------------------------------------
    # Calculate remaining seconds after removing hours
    # -----------------------------------------------------

    remaining = total_seconds % 3600


    # -----------------------------------------------------
    # Calculate minutes
    # -----------------------------------------------------

    minutes = int(remaining // 60)


    # -----------------------------------------------------
    # Calculate remaining seconds
    # -----------------------------------------------------

    seconds = remaining % 60


    # -----------------------------------------------------
    # Extract the whole seconds and milliseconds.
    # -----------------------------------------------------

    whole_seconds = int(seconds)

    milliseconds = round(
        (seconds - whole_seconds) * 1000
    )


    # -----------------------------------------------------
    # Handle the rare case where rounding produces
    # exactly 1000 milliseconds.
    # -----------------------------------------------------

    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0


    # -----------------------------------------------------
    # Format as:
    #
    # HH:MM:SS,mmm
    # -----------------------------------------------------

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{whole_seconds:02d},"
        f"{milliseconds:03d}"
    )

seconds = 83.45

timestamp = seconds_to_timestamp(seconds)

# print(timestamp)    
