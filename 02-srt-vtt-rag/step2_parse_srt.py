from pathlib import Path

# ---------------------------------------------------------
# STEP 1: Locate the SRT file
# ---------------------------------------------------------

srt_path = Path("data/01_native-components-vs-core-components_epm.srt")

# ---------------------------------------------------------
# STEP 2: Check whether the file exists
# ---------------------------------------------------------

if not srt_path.exists():
    print(f"SRT file not found: {srt_path}")
    exit()


# ---------------------------------------------------------
# STEP 3: Load the complete SRT file
# ---------------------------------------------------------

srt_text = srt_path.read_text(
    encoding="utf-8",
    errors="replace"
)

# ---------------------------------------------------------
# STEP 4: Split the SRT into subtitle blocks
# ---------------------------------------------------------

# In an SRT file, subtitle blocks are separated by
# blank lines.
#
# Example:
#
# 1
# 00:00:01,000 --> 00:00:04,000
# Hello
#
# 2
# 00:00:04,500 --> 00:00:08,000
# Welcome
#
# The "\n\n" represents the blank line between blocks.

blocks = srt_text.strip().split("\n\n")


print(f"Total subtitle blocks found: {len(blocks)}")



# ---------------------------------------------------------
# STEP 5: Create a list to store parsed subtitles
# ---------------------------------------------------------

subtitles = []

# ---------------------------------------------------------
# STEP 6: Parse every subtitle block
# ---------------------------------------------------------

for block in blocks:

    # print("\n-----------------------------")
    # print("RAW BLOCK")
    # print("-----------------------------")
    # print(block)

    # Split one subtitle block into individual lines.
    # lines = block.splitlines()

    # print("\n-----------------------------")
    # print("LINES")
    # print("-----------------------------")

    # print(lines)


    #  # Convert the block into individual lines.
    # lines = block.splitlines()

    #  # First line = subtitle number
    # subtitle_number = lines[0]

    # # Second line = timestamp information
    # timestamp = lines[1]

    # # Everything after the timestamp = subtitle text
    # subtitle_text = " ".join(lines[2:])

    # print("\n-----------------------------")
    # print("PARSED SUBTITLE")
    # print("-----------------------------")

    # print("Number   :", subtitle_number)
    # print("Time     :", timestamp)
    # print("Text     :", subtitle_text)


    # Split the subtitle block into lines.
    lines = block.splitlines()

    # Basic validation.
    # We need at least:
    #   line 0 -> subtitle number
    #   line 1 -> timestamp
    #   line 2 -> text
    if len(lines) < 3:
        continueuy

    # First line = subtitle number
    subtitle_number = int(lines[0])

    # Second line = timestamp
    timestamp = lines[1]

    # Split start and end timestamps.
    start_time, end_time = timestamp.split(" --> ")

    # Subtitle text.
    # Remaining lines = subtitle text.
    # We join them because subtitle text can span
    # multiple lines.
    # Joining lines[2:] handles multi-line subtitles.
    subtitle_text = " ".join(lines[2:])

    # Clean the subtitle before storing it.# Clean formatting/noise.
    subtitle_text = clean_text(subtitle_text)

    print("\n-----------------------------")
    print("PARSED SUBTITLE")
    print("-----------------------------")

    print("Number :", subtitle_number)
    print("Start  :", start_time)
    print("End    :", end_time)
    print("Text   :", subtitle_text)


    # Create structured subtitle object.
    subtitle = {
        "index": subtitle_number,
        "start": start_time,
        "end": end_time,
        "text": subtitle_text
    }
     # Store the subtitle.
    subtitles.append(subtitle)


# ---------------------------------------------------------
# STEP 7: Display the parsed result
# ---------------------------------------------------------
print(f"Total subtitles parsed: {len(subtitles)}")

for subtitle in subtitles:

    print("\n-----------------------------")

    print("Index :", subtitle["index"])
    print("Start :", subtitle["start"])
    print("End   :", subtitle["end"])
    print("Text  :", subtitle["text"])