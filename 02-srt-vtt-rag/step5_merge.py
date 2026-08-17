import re
from pathlib import Path


# =========================================================
# STEP 1: Convert SRT timestamp to seconds
# =========================================================

def timestamp_to_seconds(timestamp):
    """
    Convert an SRT timestamp such as:

        00:01:23,450

    into:

        83.45 seconds
    """

    # SRT uses comma for milliseconds.
    # Python's float uses a dot.
    timestamp = timestamp.replace(",", ".")

    # Split HH:MM:SS.milliseconds
    hours, minutes, seconds = timestamp.split(":")

    hours = int(hours)
    minutes = int(minutes)
    seconds = float(seconds)

    # Convert everything to seconds.
    total_seconds = (
        hours * 3600
        + minutes * 60
        + seconds
    )

    return total_seconds


# =========================================================
# STEP 2: Clean subtitle text
# =========================================================

def clean_text(text):
    """
    Remove obvious formatting/noise from subtitle text.
    """

    # Remove HTML/XML tags.
    #
    # Example:
    # <i>Hello</i>
    #
    # becomes:
    # Hello
    text = re.sub(r"<[^>]+>", "", text)

    # Remove common sound-effect markers.
    text = re.sub(
        r"\[(music|applause|laughter|laughing|cheering)\]",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Replace multiple spaces/newlines with one space.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# STEP 3: Load and parse SRT
# =========================================================

def parse_srt(file_path):
    """
    Read an SRT file and convert it into a list of
    structured subtitle dictionaries.
    """

    # Read the complete SRT file.
    srt_text = Path(file_path).read_text(
        encoding="utf-8",
        errors="replace"
    )

    # Subtitle blocks are normally separated by blank lines.
    blocks = srt_text.strip().split("\n\n")

    # This list will contain all parsed subtitles.
    subtitles = []

    # Process every subtitle block.
    for block in blocks:

        # Split block into individual lines.
        lines = block.splitlines()

        # A valid basic subtitle needs:
        #
        # line 0 -> number
        # line 1 -> timestamp
        # line 2+ -> text
        if len(lines) < 3:
            continue

        # Subtitle number.
        subtitle_number = int(lines[0])

        # Timestamp line.
        timestamp = lines[1]

        # Separate start and end timestamp.
        start_time, end_time = timestamp.split(" --> ")

        # Subtitle text can contain multiple lines.
        subtitle_text = " ".join(lines[2:])

        # Clean the text.
        subtitle_text = clean_text(subtitle_text)

        # Convert timestamps into seconds.
        start_seconds = timestamp_to_seconds(start_time)
        end_seconds = timestamp_to_seconds(end_time)

        # Create structured subtitle object.
        subtitle = {
            "index": subtitle_number,

            # Human-readable timestamps.
            "start": start_time,
            "end": end_time,

            # Machine-friendly timestamps.
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,

            # Cleaned subtitle text.
            "text": subtitle_text
        }

        # Add subtitle to the list.
        subtitles.append(subtitle)

    return subtitles


# =========================================================
# STEP 4: Merge subtitles
# =========================================================

def merge_subtitles(
    subtitles,
    max_duration=30,
    max_gap=2
):
    """
    Merge consecutive subtitles into larger segments.

    max_duration:
        Maximum duration allowed for a segment.

    max_gap:
        Maximum time gap allowed between subtitles.
    """

    # Final list of merged segments.
    segments = []

    # Currently active segment.
    current_segment = None

    # Process subtitles one by one.
    for subtitle in subtitles:

        # -------------------------------------------------
        # First subtitle
        # -------------------------------------------------

        if current_segment is None:

            current_segment = {
                "subtitle_start": subtitle["index"],
                "subtitle_end": subtitle["index"],

                "start": subtitle["start"],
                "start_seconds": subtitle["start_seconds"],

                "end": subtitle["end"],
                "end_seconds": subtitle["end_seconds"],

                "text": subtitle["text"]
            }

            continue

        # -------------------------------------------------
        # Calculate gap
        # -------------------------------------------------

        gap = (
            subtitle["start_seconds"]
            - current_segment["end_seconds"]
        )

        # -------------------------------------------------
        # Calculate duration if we merge
        # -------------------------------------------------

        new_duration = (
            subtitle["end_seconds"]
            - current_segment["start_seconds"]
        )

        # -------------------------------------------------
        # Decide whether this subtitle can be merged
        # -------------------------------------------------

        can_merge = (
            gap <= max_gap
            and new_duration <= max_duration
        )

        if can_merge:

            # Add text.
            current_segment["text"] += " " + subtitle["text"]

            # Extend end time.
            current_segment["end"] = subtitle["end"]

            current_segment["end_seconds"] = subtitle["end_seconds"]

            # Remember the final subtitle number.
            current_segment["subtitle_end"] = subtitle["index"]

        else:

            # Current segment is finished.
            segments.append(current_segment)

            # Start a new segment.
            current_segment = {
                "subtitle_start": subtitle["index"],
                "subtitle_end": subtitle["index"],

                "start": subtitle["start"],
                "start_seconds": subtitle["start_seconds"],

                "end": subtitle["end"],
                "end_seconds": subtitle["end_seconds"],

                "text": subtitle["text"]
            }

    # -----------------------------------------------------
    # Add the final segment
    # -----------------------------------------------------

    if current_segment is not None:
        segments.append(current_segment)

    return segments


# =========================================================
# MAIN PROGRAM
# =========================================================

# Path to your SRT file.
srt_path = Path("data/01_native-components-vs-core-components_epm.srt")


# ---------------------------------------------------------
# IMPORTANT:
#
# This is where `subtitles` gets CREATED.
#
# Previously this was missing from your step5_merge.py,
# which caused:
#
# NameError: name 'subtitles' is not defined
# ---------------------------------------------------------

subtitles = parse_srt(srt_path)


print(f"Total subtitles parsed: {len(subtitles)}")


# =========================================================
# Merge subtitles into larger segments
# =========================================================

segments = merge_subtitles(
    subtitles,
    max_duration=30,
    max_gap=2
)


# =========================================================
# Display merged segments
# =========================================================

print("\n")
print("=" * 60)
print("MERGED SEGMENTS")
print("=" * 60)


for i, segment in enumerate(segments, start=1):

    duration = (
        segment["end_seconds"]
        - segment["start_seconds"]
    )

    print("\n" + "-" * 60)

    print("Segment       :", i)

    print(
        "Subtitle range:",
        segment["subtitle_start"],
        "->",
        segment["subtitle_end"]
    )

    print("Start         :", segment["start"])

    print("End           :", segment["end"])

    print("Duration      :", duration, "seconds")

    print("Text          :", segment["text"])
