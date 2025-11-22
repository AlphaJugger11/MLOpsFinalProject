import os
import re
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
DATASET_DIR = BASE_DIR / "dataset"

barber_dir = DATASET_DIR / "barber"
race1_dir = barber_dir / "Race 1"
race2_dir = barber_dir / "Race 2"

# Create directories
race1_dir.mkdir(exist_ok=True)
race2_dir.mkdir(exist_ok=True)

# Regex patterns (case insensitive)
race1_patterns = [
    r"race[\s_]?1",     # race 1, race_1, Race1
    r"\br1\b",          # r1 as a separate token
    r"^r1_",            # starts with r1_
]

race2_patterns = [
    r"race[\s_]?2",
    r"\br2\b",
    r"^r2_",
]

def matches_any_pattern(filename, patterns):
    filename_lower = filename.lower()
    return any(re.search(p, filename_lower) for p in patterns)

# Sort files
for file in os.listdir(barber_dir):
    if file == ".DS_Store":
        continue

    file_path = barber_dir / file
    if not file_path.is_file():
        continue

    # Match Race 1
    if matches_any_pattern(file, race1_patterns):
        shutil.move(str(file_path), str(race1_dir / file))
        continue

    # Match Race 2
    if matches_any_pattern(file, race2_patterns):
        shutil.move(str(file_path), str(race2_dir / file))
        continue
