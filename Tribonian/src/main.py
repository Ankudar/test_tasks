import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from file_utils import convert_files, get_files
from summarizer_hf import Summarizer

load_dotenv()

with open("./Tribonian/urls.txt") as f:
    GDRIVE_URLS = [line.strip() for line in f if line.strip()]

DATA_INCOME_FOLDER = Path(os.environ["DATA_INCOME_FOLDER"])
DATA_PREPARED_FOLDER = Path(os.environ["DATA_PREPARED_FOLDER"])

DATA_INCOME_FOLDER.mkdir(parents=True, exist_ok=True)
DATA_PREPARED_FOLDER.mkdir(parents=True, exist_ok=True)


# get_files(GDRIVE_URLS, DATA_INCOME_FOLDER)
# convert_files(DATA_INCOME_FOLDER, DATA_PREPARED_FOLDER)

# device=0 — GPU, device=-1 — CPU
summarizer = Summarizer(device=0)
summary_text = summarizer.summarize_folder(DATA_PREPARED_FOLDER)
print("\n=== Summary ===\n")
print(summary_text)

with open("./Tribonian/result.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)
