import re
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

# Maximum duration allowed when merging subtitles.
MAX_SEGMENT_DURATION = 30


# Maximum time gap allowed between two subtitles.
#
# If the gap is larger than this, we assume there may be
# a pause or topic/scene change.
MAX_GAP = 2


# Maximum characters allowed in one RAG chunk.
#
# IMPORTANT:
# This is temporary.
# Later we will replace character-based chunking with
# token-based chunking.
MAX_CHARS = 200


# Number of subtitles to carry from the previous chunk.
#
# Example:
#
# Chunk 1 -> subtitles 1, 2, 3
# Chunk 2 -> subtitles 3, 4, 5
#
# Subtitle 3 is the overlap.
OVERLAP_SUBTITLES = 1


# =========================================================
# FUNCTION 1
# Convert SRT timestamp into seconds
# =========================================================

def timestamp_to_seconds(timestamp):
    """
    Convert:

        00:01:23,450

    into:

        83.45 seconds
    """

    # SRT uses ',' for milliseconds.
    # Python's decimal representation uses '.'
    timestamp = timestamp.replace(",", ".")

    # Split HH:MM:SS.milliseconds
    hours, minutes, seconds = timestamp.split(":")

    hours = int(hours)
    minutes = int(minutes)
    seconds = float(seconds)

    # Convert everything to total seconds.
    total_seconds = (
        hours * 3600
        + minutes * 60
        + seconds
    )

    return total_seconds


# =========================================================
# FUNCTION 2
# Clean subtitle text
# =========================================================

def clean_text(text):
    """
    Remove obvious subtitle formatting/noise.
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

    # Normalize multiple spaces.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# FUNCTION 3
# Parse the SRT file
# =========================================================

def parse_srt(srt_path):
    """
    Read and parse an SRT file.

    Returns a list of subtitle dictionaries.
    """

    # -----------------------------------------------------
    # Read complete SRT file
    # -----------------------------------------------------

    srt_text = srt_path.read_text(
        encoding="utf-8",
        errors="replace"
    )


    # -----------------------------------------------------
    # Separate subtitle blocks
    # -----------------------------------------------------

    blocks = re.split(
        r"\n\s*\n",
        srt_text.strip()
    )


    subtitles = []


    # -----------------------------------------------------
    # Parse each block
    # -----------------------------------------------------

    for block in blocks:

        lines = block.splitlines()


        # A valid subtitle needs at least:
        #
        # line 0 -> subtitle number
        # line 1 -> timestamp
        # line 2 -> text
        if len(lines) < 3:
            continue


        # -------------------------------------------------
        # Subtitle number
        # -------------------------------------------------

        try:
            subtitle_number = int(lines[0].strip())

        except ValueError:
            # Ignore malformed blocks.
            continue


        # -------------------------------------------------
        # Timestamp
        # -------------------------------------------------

        timestamp = lines[1].strip()


        # Example:
        #
        # 00:00:01,000 --> 00:00:04,000
        #

        if " --> " not in timestamp:
            continue


        start_time, end_time = timestamp.split(
            " --> ",
            maxsplit=1
        )


        # -------------------------------------------------
        # Subtitle text
        # -------------------------------------------------

        # lines[2:] is important because subtitle text
        # can contain multiple lines.
        subtitle_text = " ".join(lines[2:])


        # Clean the text.
        subtitle_text = clean_text(
            subtitle_text
        )


        # Skip completely empty subtitles.
        if not subtitle_text:
            continue


        # -------------------------------------------------
        # Convert timestamps to seconds
        # -------------------------------------------------

        start_seconds = timestamp_to_seconds(
            start_time
        )

        end_seconds = timestamp_to_seconds(
            end_time
        )


        # -------------------------------------------------
        # Create subtitle object
        # -------------------------------------------------

        subtitle = {

            # Original subtitle number
            "index": subtitle_number,

            # Human-readable timestamps
            "start": start_time,
            "end": end_time,

            # Numeric timestamps
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,

            # Cleaned text
            "text": subtitle_text
        }


        # Add subtitle to list.
        subtitles.append(subtitle)


    return subtitles


# =========================================================
# FUNCTION 4
# Merge subtitles into larger segments
# =========================================================

def merge_subtitles(
    subtitles,
    max_duration=MAX_SEGMENT_DURATION,
    max_gap=MAX_GAP
):
    """
    Merge consecutive subtitles.

    IMPORTANT:
    We preserve the original subtitle objects.

    This allows us to recover exact timestamps later.
    """

    segments = []

    current_segment = None


    # -----------------------------------------------------
    # Process subtitles one by one
    # -----------------------------------------------------

    for subtitle in subtitles:


        # =================================================
        # CASE 1
        # No current segment exists.
        # =================================================

        if current_segment is None:

            current_segment = {

                # IMPORTANT:
                # Preserve original subtitles.
                "subtitles": [subtitle],

                # Segment starts at first subtitle.
                "start": subtitle["start"],
                "start_seconds": subtitle["start_seconds"],

                # Segment ends at first subtitle initially.
                "end": subtitle["end"],
                "end_seconds": subtitle["end_seconds"]
            }

            continue


        # =================================================
        # Calculate gap
        # =================================================

        gap = (
            subtitle["start_seconds"]
            - current_segment["end_seconds"]
        )


        # =================================================
        # Calculate duration if we merge
        # =================================================

        new_duration = (
            subtitle["end_seconds"]
            - current_segment["start_seconds"]
        )


        # =================================================
        # Decide whether to merge
        # =================================================

        can_merge = (
            gap <= max_gap
            and new_duration <= max_duration
        )


        # =================================================
        # CASE 2
        # Merge subtitle into current segment.
        # =================================================

        if can_merge:

            # Preserve original subtitle object.
            current_segment["subtitles"].append(
                subtitle
            )

            # Extend segment end time.
            current_segment["end"] = (
                subtitle["end"]
            )

            current_segment["end_seconds"] = (
                subtitle["end_seconds"]
            )


        # =================================================
        # CASE 3
        # Start a new segment.
        # =================================================

        else:

            # Save completed segment.
            segments.append(
                current_segment
            )


            # Create new segment.
            current_segment = {

                "subtitles": [subtitle],

                "start": subtitle["start"],
                "start_seconds": subtitle["start_seconds"],

                "end": subtitle["end"],
                "end_seconds": subtitle["end_seconds"]
            }


    # -----------------------------------------------------
    # Add final segment
    # -----------------------------------------------------

    if current_segment is not None:

        segments.append(
            current_segment
        )


    return segments


# =========================================================
# FUNCTION 5
# Create one RAG chunk from subtitles
# =========================================================

def create_chunk_from_subtitles(
    subtitles
):
    """
    Create a RAG chunk from a list of subtitles.

    The chunk timestamp is calculated from:

        first subtitle START
        +
        last subtitle END
    """

    # -----------------------------------------------------
    # Combine subtitle text
    # -----------------------------------------------------

    text = " ".join(
        subtitle["text"]
        for subtitle in subtitles
    )


    # -----------------------------------------------------
    # Create chunk
    # -----------------------------------------------------

    chunk = {

        # Main content that will eventually be embedded.
        "text": text,

        # -------------------------------------------------
        # EXACT VIDEO TIMESTAMP
        # -------------------------------------------------

        "start": subtitles[0]["start"],

        "start_seconds": (
            subtitles[0]["start_seconds"]
        ),

        "end": subtitles[-1]["end"],

        "end_seconds": (
            subtitles[-1]["end_seconds"]
        ),

        # -------------------------------------------------
        # SOURCE SUBTITLE RANGE
        # -------------------------------------------------

        "subtitle_start": (
            subtitles[0]["index"]
        ),

        "subtitle_end": (
            subtitles[-1]["index"]
        )
    }


    return chunk


# =========================================================
# FUNCTION 6
# Create chunks from one segment
# =========================================================

def chunk_segment(
    segment,
    max_chars=MAX_CHARS,
    overlap_subtitles=OVERLAP_SUBTITLES
):
    """
    Split a merged segment into smaller RAG chunks.

    Important:
    We NEVER split a subtitle in the middle.

    Therefore every chunk keeps valid video timestamps.
    """

    subtitles = segment["subtitles"]

    chunks = []

    current_subtitles = []

    current_length = 0


    # -----------------------------------------------------
    # Process each subtitle
    # -----------------------------------------------------

    for subtitle in subtitles:

        subtitle_text = subtitle["text"]

        subtitle_length = len(
            subtitle_text
        )


        # -------------------------------------------------
        # Check whether adding this subtitle would exceed
        # the chunk size.
        # -------------------------------------------------

        would_exceed_limit = (
            current_subtitles
            and
            current_length + subtitle_length
            > max_chars
        )


        if would_exceed_limit:

            # =============================================
            # Create completed chunk
            # =============================================

            chunk = create_chunk_from_subtitles(
                current_subtitles
            )

            chunks.append(chunk)


            # =============================================
            # Preserve overlap
            # =============================================

            if overlap_subtitles > 0:

                current_subtitles = (
                    current_subtitles[
                        -overlap_subtitles:
                    ]
                )

            else:

                current_subtitles = []


            # Recalculate current character count.
            current_length = sum(
                len(item["text"])
                for item in current_subtitles
            )


        # -------------------------------------------------
        # Add current subtitle
        # -------------------------------------------------

        current_subtitles.append(
            subtitle
        )

        current_length += subtitle_length


    # -----------------------------------------------------
    # Save final chunk
    # -----------------------------------------------------

    if current_subtitles:

        chunk = create_chunk_from_subtitles(
            current_subtitles
        )

        chunks.append(chunk)


    return chunks


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    # -----------------------------------------------------
    # Locate SRT file
    # -----------------------------------------------------

    srt_path = Path(
        "data/01_native-components-vs-core-components_epm.srt"
    )


    # -----------------------------------------------------
    # Check file
    # -----------------------------------------------------

    if not srt_path.exists():

        print(
            f"SRT file not found: {srt_path}"
        )

        return


    # -----------------------------------------------------
    # Parse SRT
    # -----------------------------------------------------

    subtitles = parse_srt(
        srt_path
    )


    print(
        f"Subtitles parsed: {len(subtitles)}"
    )


    # -----------------------------------------------------
    # Merge subtitles into segments
    # -----------------------------------------------------

    segments = merge_subtitles(
        subtitles
    )


    print(
        f"Segments created: {len(segments)}"
    )


    # -----------------------------------------------------
    # Create RAG chunks
    # -----------------------------------------------------

    all_chunks = []


    for segment in segments:

        chunks = chunk_segment(
            segment
        )

        all_chunks.extend(
            chunks
        )


    # -----------------------------------------------------
    # Display final chunks
    # -----------------------------------------------------

    print(
        f"RAG chunks created: {len(all_chunks)}"
    )


    for i, chunk in enumerate(
        all_chunks,
        start=1
    ):

        print("\n")
        print("=" * 60)
        print(f"CHUNK {i}")
        print("=" * 60)

        print(
            "Timestamp:",
            chunk["start"],
            "-->",
            chunk["end"]
        )

        print(
            "Seconds:",
            chunk["start_seconds"],
            "-->",
            chunk["end_seconds"]
        )

        print(
            "Subtitle range:",
            chunk["subtitle_start"],
            "-->",
            chunk["subtitle_end"]
        )

        print(
            "\nText:"
        )

        print(
            chunk["text"]
        )
        print(
                "Characters:",
                len(chunk["text"])
            )
       


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()