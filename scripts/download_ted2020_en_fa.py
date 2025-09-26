# scripts/download_ted2020_en_fa.py
import zipfile
import os
import urllib.request
from pathlib import Path

RAW_DIR = Path("data/raw/TED2020")
RAW_DIR.mkdir(parents=True, exist_ok=True)

ZIP_PATH = RAW_DIR / "en-fa.txt.zip"
FA_PATH  = RAW_DIR / "TED2020.en-fa.fa"
EN_PATH  = RAW_DIR / "TED2020.en-fa.en"

ZIP_URL  = "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-fa.txt.zip"

def main():
    if not (FA_PATH.exists() and EN_PATH.exists()):
        if not ZIP_PATH.exists():
            print("Downloading TED2020 en-fa dataset...")
            urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
    else:
        print("Dataset already exists, skip downloading.")

    print(f"EN: {EN_PATH.exists()}  FA: {FA_PATH.exists()}")
    print(f"Files at: {RAW_DIR.resolve()}")

if __name__ == "__main__":
    main()
