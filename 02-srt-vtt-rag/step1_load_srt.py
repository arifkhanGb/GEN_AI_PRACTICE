from pathlib import Path

# ---------------------------------------------------------
# STEP 1: Locate the SRT file
# ---------------------------------------------------------

# Path to our SRT file.
# Path() is better than manually building strings such as:
# "data/sample.srt"
#
# because pathlib works cleanly across operating systems.
srt_path = Path("data/01_native-components-vs-core-components_epm.srt")

# ---------------------------------------------------------
# STEP 2: Check whether the file actually exists
# ---------------------------------------------------------

if not srt_path.exists():
    print(f"SRT file not found: {srt_path}")
    exit()

# ---------------------------------------------------------
# STEP 3: Read the complete SRT file
# ---------------------------------------------------------

# UTF-8 is important because subtitles can contain:
# - English
# - Hindi
# - Chinese
# - Arabic
# - emojis
# - special characters
#
# errors="replace" prevents the program from crashing if
# the subtitle file contains an invalid character.
srt_text = srt_path.read_text(
    encoding="utf-8",
    errors="replace"
)

# ---------------------------------------------------------
# STEP 4: Inspect what we loaded
# ---------------------------------------------------------

print("SRT file loaded successfully!")
print("--------------------------------")

# Display the complete raw SRT content.
print(type(srt_text))
# print(srt_text)