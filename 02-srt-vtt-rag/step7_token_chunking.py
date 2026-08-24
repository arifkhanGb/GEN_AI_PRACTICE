import re
import json
from pathlib import Path

import tiktoken

# =========================================================
# CONFIGURATION
# =========================================================

# Target maximum number of tokens per chunk.
#
# This is much larger than our previous 100-token test.
MAX_TOKENS = 250


# Target overlap between consecutive chunks.
#
# We will try to preserve approximately this many tokens
# from the previous chunk.
OVERLAP_TOKENS = 50

# =========================================================
# TOKENIZER
# =========================================================

# Load tokenizer used for token counting.
TOKENIZER = tiktoken.get_encoding(
    "cl100k_base"
)

def count_tokens(text):
    """
    Count the number of tokens in a piece of text.
    """

    return len(
        TOKENIZER.encode(text)
    )

def timestamp_to_seconds(timestamp):
    """
    Convert SRT timestamp:

        00:01:23,450

    into:

        83.45 seconds
    """

    timestamp = timestamp.replace(
        ",",
        "."
    )

    hours, minutes, seconds = timestamp.split(":")

    return (
        int(hours) * 3600
        + int(minutes) * 60
        + float(seconds)
    )

def clean_text(text):
    """
    Clean subtitle text.
    """

    # Remove HTML tags such as <i>...</i>
    text = re.sub(
        r"<[^>]+>",
        "",
        text
    )

    # Remove common sound effects.
    text = re.sub(
        r"\[(music|applause|laughter|laughing|cheering)\]",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

def parse_srt(srt_path):
    """
    Parse an SRT file into subtitle objects.
    """

    # -----------------------------------------------------
    # Read SRT
    # -----------------------------------------------------

    srt_text = srt_path.read_text(
        encoding="utf-8",
        errors="replace"
    )


    # -----------------------------------------------------
    # Split subtitle blocks
    # -----------------------------------------------------

    blocks = re.split(
        r"\n\s*\n",
        srt_text.strip()
    )


    subtitles = []


    # -----------------------------------------------------
    # Process each block
    # -----------------------------------------------------

    for block in blocks:

        lines = block.splitlines()


        # Minimum:
        #
        # 1 -> subtitle number
        # 2 -> timestamp
        # 3 -> text
        if len(lines) < 3:
            continue


        # -------------------------------------------------
        # Subtitle number
        # -------------------------------------------------

        try:
            index = int(
                lines[0].strip()
            )

        except ValueError:
            continue


        # -------------------------------------------------
        # Timestamp
        # -------------------------------------------------

        timestamp_line = lines[1].strip()


        if " --> " not in timestamp_line:
            continue


        start_time, end_time = timestamp_line.split(
            " --> ",
            maxsplit=1
        )


        # -------------------------------------------------
        # Subtitle text
        # -------------------------------------------------

        text = " ".join(
            lines[2:]
        )


        text = clean_text(
            text
        )


        if not text:
            continue


        # -------------------------------------------------
        # Numeric timestamps
        # -------------------------------------------------

        start_seconds = timestamp_to_seconds(
            start_time
        )

        end_seconds = timestamp_to_seconds(
            end_time
        )


        # -------------------------------------------------
        # Token count
        # -------------------------------------------------

        token_count = count_tokens(
            text
        )


        # -------------------------------------------------
        # Store subtitle
        # -------------------------------------------------

        subtitle = {

            "index": index,

            "start": start_time,

            "end": end_time,

            "start_seconds": start_seconds,

            "end_seconds": end_seconds,

            "text": text,

            "token_count": token_count
        }


        subtitles.append(
            subtitle
        )


    return subtitles

def split_text_by_tokens(text, max_tokens):
    """
    Split text into token-sized pieces.

    This function is only used when a single subtitle
    is larger than MAX_TOKENS.

    IMPORTANT:
    We are splitting the TEXT only.

    We are NOT creating fake timestamps.
    """

    # -----------------------------------------------------
    # Convert text into token IDs
    # -----------------------------------------------------

    token_ids = TOKENIZER.encode(
        text
    )


    # -----------------------------------------------------
    # Split token IDs into smaller groups
    # -----------------------------------------------------

    chunks = []

    for start in range(
        0,
        len(token_ids),
        max_tokens
    ):

        end = start + max_tokens

        chunk_token_ids = (
            token_ids[start:end]
        )


        # Convert token IDs back to text.
        chunk_text = TOKENIZER.decode(
            chunk_token_ids
        )


        chunks.append(
            chunk_text
        )


    return chunks

def create_chunk(
    subtitles,
    text=None,
    split_from_oversized_subtitle=False,
    source_subtitle_index=None,
    split_part=None,
    split_total=None
):
    """
    Create a RAG chunk.

    Optional metadata is used when a single oversized
    subtitle has been split.
    """

    # -----------------------------------------------------
    # Normal case:
    # Build text from subtitles.
    # -----------------------------------------------------

    if text is None:

        text = " ".join(
            subtitle["text"]
            for subtitle in subtitles
        )


    # -----------------------------------------------------
    # Count tokens
    # -----------------------------------------------------

    token_count = count_tokens(
        text
    )


    # -----------------------------------------------------
    # Create metadata
    # -----------------------------------------------------

    return {

        # Main text.
        "text": text,

        # Size information.
        "token_count": token_count,

        "character_count": len(text),

        # Video timestamp.
        "start": subtitles[0]["start"],

        "start_seconds": (
            subtitles[0]["start_seconds"]
        ),

        "end": subtitles[-1]["end"],

        "end_seconds": (
            subtitles[-1]["end_seconds"]
        ),

        # Source subtitle range.
        "subtitle_start": (
            subtitles[0]["index"]
        ),

        "subtitle_end": (
            subtitles[-1]["index"]
        ),

        # -------------------------------------------------
        # Oversized subtitle metadata
        # -------------------------------------------------

        "split_from_oversized_subtitle": (
            split_from_oversized_subtitle
        ),

        "source_subtitle_index": (
            source_subtitle_index
        ),

        "split_part": split_part,

        "split_total": split_total
    }

def create_token_chunks(
    subtitles,
    max_tokens=MAX_TOKENS,
    overlap_tokens=OVERLAP_TOKENS
):
    """
    Create token-aware RAG chunks.

    Handles:

    1. Normal subtitles.
    2. Token-aware chunk sizing.
    3. Token-based overlap.
    4. Oversized subtitles.
    5. Exact source timestamps.
    """

    chunks = []

    current_subtitles = []

    current_tokens = 0


    # =====================================================
    # Process subtitles
    # =====================================================

    for subtitle in subtitles:

        subtitle_tokens = (
            subtitle["token_count"]
        )


        # =================================================
        # CASE 1
        # Subtitle itself is larger than MAX_TOKENS.
        # =================================================

        if subtitle_tokens > max_tokens:

            # -------------------------------------------------
            # First save the current chunk.
            # -------------------------------------------------

            if current_subtitles:

                chunk = create_chunk(
                    current_subtitles
                )

                chunks.append(
                    chunk
                )

                current_subtitles = []

                current_tokens = 0


            # -------------------------------------------------
            # Split oversized subtitle text.
            # -------------------------------------------------

            text_parts = split_text_by_tokens(
                subtitle["text"],
                max_tokens
            )


            total_parts = len(
                text_parts
            )


            # -------------------------------------------------
            # Create chunks for each text part.
            # -------------------------------------------------

            for part_number, part_text in enumerate(
                text_parts,
                start=1
            ):

                chunk = create_chunk(

                    # Keep the ORIGINAL subtitle.
                    #
                    # Therefore the timestamp remains
                    # truthful.
                    subtitles=[subtitle],

                    # But use only this text part.
                    text=part_text,

                    # Mark that we split the subtitle.
                    split_from_oversized_subtitle=True,

                    # Remember source subtitle.
                    source_subtitle_index=(
                        subtitle["index"]
                    ),

                    # Part information.
                    split_part=part_number,

                    split_total=total_parts
                )


                chunks.append(
                    chunk
                )


            # -------------------------------------------------
            # Continue with next subtitle.
            # -------------------------------------------------

            continue


        # =================================================
        # CASE 2
        # Normal subtitle.
        # =================================================

        would_exceed = (
            current_subtitles
            and
            current_tokens + subtitle_tokens
            > max_tokens
        )


        if would_exceed:

            # -------------------------------------------------
            # Save current chunk.
            # -------------------------------------------------

            chunk = create_chunk(
                current_subtitles
            )

            chunks.append(
                chunk
            )


            # -------------------------------------------------
            # Create token-based overlap.
            # -------------------------------------------------

            overlap_subtitles = []

            overlap_count = 0


            for previous_subtitle in reversed(
                current_subtitles
            ):

                overlap_subtitles.insert(
                    0,
                    previous_subtitle
                )


                overlap_count += (
                    previous_subtitle["token_count"]
                )


                if overlap_count >= overlap_tokens:
                    break


            # -------------------------------------------------
            # Start next chunk with overlap.
            # -------------------------------------------------

            current_subtitles = (
                overlap_subtitles.copy()
            )


            current_tokens = sum(
                item["token_count"]
                for item in current_subtitles
            )


        # -----------------------------------------------------
        # Add current subtitle.
        # -----------------------------------------------------

        current_subtitles.append(
            subtitle
        )

        current_tokens += (
            subtitle_tokens
        )


    # =====================================================
    # Save final chunk.
    # =====================================================

    if current_subtitles:

        chunk = create_chunk(
            current_subtitles
        )

        chunks.append(
            chunk
        )


    return chunks

def test_oversized_subtitle():
    """
    Test what happens when one subtitle exceeds
    MAX_TOKENS.
    """

    long_text = (
        "React Native is a framework "
        "for building cross platform applications. "
        * 100
    )


    test_subtitle = {

        "index": 999,

        "start": "00:10:00,000",

        "end": "00:10:30,000",

        "start_seconds": 600.0,

        "end_seconds": 630.0,

        "text": long_text,

        "token_count": count_tokens(
            long_text
        )
    }


    print(
        "Test subtitle tokens:",
        test_subtitle["token_count"]
    )


    chunks = create_token_chunks(
        [test_subtitle]
    )


    print(
        "Test chunks:",
        len(chunks)
    )


    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        print(
            "Part:",
            i,
            "/",
            chunk["split_total"]
        )

        print(
            "Tokens:",
            chunk["token_count"]
        )

        print(
            "Timestamp:",
            chunk["start"],
            "-->",
            chunk["end"]
        )



# Add a JSON-saving function
def save_chunks_to_json(
    chunks,
    output_path,
    source_file
):
    """
    Save RAG chunks as structured JSON.

    Parameters
    ----------
    chunks:
        List of generated RAG chunks.

    output_path:
        Where the JSON file should be written.

    source_file:
        Original SRT filename.
    """

    # =====================================================
    # Build final JSON structure
    # =====================================================

    data = {

        # -------------------------------------------------
        # Metadata about this dataset
        # -------------------------------------------------

        "source_file": source_file,

        "total_chunks": len(chunks),

        "chunking_config": {

            "max_tokens": MAX_TOKENS,

            "overlap_tokens": OVERLAP_TOKENS
        },

        # -------------------------------------------------
        # Actual RAG chunks
        # -------------------------------------------------

        "chunks": []
    }


    # =====================================================
    # Add chunk IDs
    # =====================================================

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        # Make a copy so we don't modify
        # the original chunk object.
        chunk_data = chunk.copy()


        # -------------------------------------------------
        # Unique chunk ID
        # -------------------------------------------------

        chunk_data["chunk_id"] = (
            f"chunk_{index:04d}"
        )


        # -------------------------------------------------
        # Add source information
        # -------------------------------------------------

        chunk_data["source_file"] = (
            source_file
        )


        # -------------------------------------------------
        # Add chunk to final dataset
        # -------------------------------------------------

        data["chunks"].append(
            chunk_data
        )


    # =====================================================
    # Make sure output directory exists
    # =====================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # =====================================================
    # Write JSON
    # =====================================================

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


    print(
        f"\nJSON saved to: {output_path}"
    )

            

def main():

    # -----------------------------------------------------
    # SRT file
    # -----------------------------------------------------

    srt_path = Path(
        "data/01_native-components-vs-core-components_epm.srt"
    )


    # -----------------------------------------------------
    # Verify file
    # -----------------------------------------------------

    if not srt_path.exists():

        print(
            f"SRT file not found: {srt_path}"
        )

        return


    # -----------------------------------------------------
    # Parse subtitles
    # -----------------------------------------------------

    subtitles = parse_srt(
        srt_path
    )


    print(
        f"Subtitles parsed: {len(subtitles)}"
    )


    # -----------------------------------------------------
    # Create token-aware chunks
    # -----------------------------------------------------

    chunks = create_token_chunks(
        subtitles
    )

    print(
        f"RAG chunks created: {len(chunks)}"
    )

    # -----------------------------------------------------
    # Save chunks to JSON
    # -----------------------------------------------------

    output_path = Path(
        "output/chunks.json"
    )

    save_chunks_to_json(

        chunks=chunks,

        output_path=output_path,

        source_file=srt_path.name
    )


    print(
        f"RAG chunks created: {len(chunks)}"
    )


    # -----------------------------------------------------
    # Display chunks
    # -----------------------------------------------------

    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        print("\n")

        print("=" * 60)

        print(
            f"CHUNK {i}"
        )

        print("=" * 60)


        print(
            "Timestamp:",
            chunk["start"],
            "-->",
            chunk["end"]
        )


        print(
            "Seconds:",
            round(chunk["start_seconds"], 3),
            "-->",
            round(chunk["end_seconds"], 3)
        )


        print(
            "Subtitle range:",
            chunk["subtitle_start"],
            "-->",
            chunk["subtitle_end"]
        )


        print(
            "Characters:",
            chunk["character_count"]
        )


        print(
            "Tokens:",
            chunk["token_count"]
        )

        print(
            "Oversized subtitle split:",
            chunk["split_from_oversized_subtitle"]
        )
        if chunk["split_from_oversized_subtitle"]:

            print(
                "Source subtitle:",
                chunk["source_subtitle_index"]
            )

            print(
                "Split part:",
                chunk["split_part"],
                "/",
                chunk["split_total"]
            )
 


        print("\nText:")

        print(
            chunk["text"]
        )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main() 


# if __name__ == "__main__":
#     test_oversized_subtitle()       