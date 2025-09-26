#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Builds data/processed/{train,dev,test}.tsv from TED2020 en-fa.

What it does:
  1) Ensures raw files exist (downloads + extracts if needed)
  2) Loads aligned EN/FA lines
  3) Cleans/normalizes/tokenizes both sides (simple, fast)
  4) Filters by length and length ratio (defaults: 2..50, ratio 3)
  5) Splits into train/dev/test (defaults: 0.90/0.05/0.05)
  6) Writes TSVs with columns: src, tgt

Run:
  python scripts/build_ted2020_dataset.py --direction en2fa
  python scripts/build_ted2020_dataset.py --direction fa2en

Dependencies:
  pip install hazm persian-tools word2number pandas
"""

import csv
import os
import re
import zipfile
import urllib.request
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from hazm import Normalizer, word_tokenize
from persian_tools import digits
from word2number import w2n


# -------------------------------
# Paths & constants
# -------------------------------
REPO_ROOT_CANDIDATES = [
    Path(__file__).resolve().parents[2],  # scripts/ -> repo root (common)
    Path(__file__).resolve().parents[1],  # just in case
    Path.cwd(),                           # current dir as fallback
]

RAW_SUBDIR = Path("data/raw/TED2020")
PROC_DIR   = Path("data/processed")
ZIP_NAME   = "en-fa.txt.zip"
ZIP_URL    = "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-fa.txt.zip"
EN_NAME    = "TED2020.en-fa.en"
FA_NAME    = "TED2020.en-fa.fa"

# -------------------------------
# Utilities
# -------------------------------
def find_repo_root() -> Path:
    for cand in REPO_ROOT_CANDIDATES:
        if (cand / "data").exists():
            return cand
    # Last resort: go up until we see "data" or stop at filesystem root
    cur = Path(__file__).resolve().parent
    for _ in range(5):
        if (cur / "data").exists():
            return cur
        cur = cur.parent
    return Path(__file__).resolve().parent

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def download_and_extract(zip_url: str, zip_path: Path, extract_dir: Path) -> None:
    if not zip_path.exists():
        print(f"[*] Downloading: {zip_url}")
        urllib.request.urlretrieve(zip_url, zip_path)
    print(f"[*] Extracting: {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

def resolve_raw_paths(repo_root: Path) -> Tuple[Path, Path, Path]:
    raw_dir = repo_root / RAW_SUBDIR
    zip_path = raw_dir / ZIP_NAME
    en_path = raw_dir / EN_NAME
    fa_path = raw_dir / FA_NAME

    ensure_dir(raw_dir)

    # If files missing, try to download/extract
    if not (en_path.exists() and fa_path.exists()):
        print("[*] Raw EN/FA not found; will download & extract.")
        download_and_extract(ZIP_URL, zip_path, raw_dir)

    # Final check (also allow rglob fallback)
    if not en_path.exists():
        cand = next((raw_dir.rglob(EN_NAME)), None)
        if cand: en_path = cand
    if not fa_path.exists():
        cand = next((raw_dir.rglob(FA_NAME)), None)
        if cand: fa_path = cand

    if not (en_path.exists() and fa_path.exists()):
        raise FileNotFoundError(
            f"Could not find {EN_NAME} or {FA_NAME} under {raw_dir}. "
            f"Ensure network access or place the files manually."
        )

    return en_path, fa_path, raw_dir


# -------------------------------
# Cleaning / tokenization
# -------------------------------
_EN_APOST = {"’": "'", "‘": "'", "“": '"', "”": '"'}
_en_norm = Normalizer()  # unused for EN, but kept for symmetry
_fa_norm = Normalizer()

def _replace_number_words_en(text: str) -> str:
    def convert(match):
        word = match.group(0)
        try:
            return str(w2n.word_to_num(word))
        except ValueError:
            return word
    return re.sub(r"\b[a-z]+\b", convert, text)

def preprocess_en(s: str) -> List[str]:
    s = re.sub(r"[\t\r\n]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    for bad, good in _EN_APOST.items():
        s = s.replace(bad, good)
    s = re.sub(r"\([^)]*\)", "", s)  # (Laughter), (Applause), etc.
    s = re.sub(r"--+", " ", s)
    s = re.sub(r"\.{3,}", " ", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\.\,\?\!]", "", s)  # keep basic punctuation
    s = _replace_number_words_en(s)
    return s.split()

def preprocess_fa(s: str) -> List[str]:
    s = _fa_norm.normalize(s)
    s = re.sub(r"[\t\r\n]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"--+", " ", s)
    s = re.sub(r"\.{3,}", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = digits.convert_to_en(s)              # unify digits
    s = re.sub(r"[^\w\s\.\,\?\!]", "", s)    # keep basic punctuation
    return word_tokenize(s)

def filter_by_length_ratio(
    df: pd.DataFrame,
    src_col_tok: str,
    tgt_col_tok: str,
    min_len: int = 2,
    max_len: int = 50,
    ratio: float = 3.0,
) -> pd.DataFrame:
    out = df.copy()
    out = out[out[src_col_tok].str.len().between(min_len, max_len)]
    out = out[out[tgt_col_tok].str.len().between(min_len, max_len)]
    out = out[out[tgt_col_tok].str.len() != 0]
    frac = out[src_col_tok].str.len() / out[tgt_col_tok].str.len()
    out = out[frac.between(1.0/ratio, ratio)]
    return out.reset_index(drop=True)

# -------------------------------
# Pipeline
# -------------------------------
def build_pairs_from_raw(en_path: Path, fa_path: Path, max_lines: Optional[int]) -> pd.DataFrame:
    with en_path.open(encoding="utf-8") as f_en, fa_path.open(encoding="utf-8") as f_fa:
        pairs = []
        for i, (en, fa) in enumerate(zip(f_en, f_fa)):
            if max_lines is not None and i >= max_lines:
                break
            en, fa = en.strip(), fa.strip()
            if en and fa:
                pairs.append((en, fa))
    df = pd.DataFrame(pairs, columns=["en", "fa"])
    return df

def preprocess_pairs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["en_tok"] = out["en"].apply(preprocess_en)
    out["fa_tok"] = out["fa"].apply(preprocess_fa)
    return out

def export_splits(df_tok: pd.DataFrame, out_dir: Path, direction: str,
                  train_ratio: float, dev_ratio: float, seed: int) -> Tuple[Path, Path, Path]:
    ensure_dir(out_dir)

    if direction == "en2fa":
        df_tok["src"] = df_tok["en_tok"].apply(lambda x: " ".join(x))
        df_tok["tgt"] = df_tok["fa_tok"].apply(lambda x: " ".join(x))
        src_col_tok, tgt_col_tok = "en_tok", "fa_tok"
    elif direction == "fa2en":
        df_tok["src"] = df_tok["fa_tok"].apply(lambda x: " ".join(x))
        df_tok["tgt"] = df_tok["en_tok"].apply(lambda x: " ".join(x))
        src_col_tok, tgt_col_tok = "fa_tok", "en_tok"
    else:
        raise ValueError("direction must be 'en2fa' or 'fa2en'")

    # Filter by lengths/ratio on chosen direction
    df_tok = filter_by_length_ratio(df_tok, src_col_tok, tgt_col_tok)

    df_final = df_tok[["src", "tgt"]].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df_final)
    n_train = int(n * train_ratio)
    n_dev   = int(n * dev_ratio)
    train = df_final.iloc[:n_train]
    dev   = df_final.iloc[n_train:n_train+n_dev]
    test  = df_final.iloc[n_train+n_dev:]

    train_path = out_dir / "train.tsv"
    dev_path   = out_dir / "dev.tsv"
    test_path  = out_dir / "test.tsv"

    train.to_csv(train_path, sep="\t", index=False)
    dev.to_csv(dev_path,     sep="\t", index=False)
    test.to_csv(test_path,   sep="\t", index=False)

    return train_path, dev_path, test_path

def validate_tsvs(paths: List[Path]) -> None:
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p, sep="\t", dtype=str)
        if list(df.columns) != ["src", "tgt"]:
            raise ValueError(f"{p}: columns must be exactly ['src','tgt'], got {list(df.columns)}")
        if df["src"].isna().any() or df["tgt"].isna().any():
            raise ValueError(f"{p}: contains NaN rows")
        if (df["src"].str.contains("\t").any() or df["tgt"].str.contains("\t").any()):
            raise ValueError(f"{p}: contains TABs inside text, replace with spaces")
        print(f"[ok] {p} rows={len(df)}")

# -------------------------------
# CLI
# -------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build TED2020 train/dev/test TSVs with cleaning")
    ap.add_argument("--direction", choices=["en2fa","fa2en"], default="en2fa",
                    help="Translation direction: src→tgt")
    ap.add_argument("--max_lines", type=int, default=None,
                    help="Limit number of aligned lines from raw (for quick tests)")
    ap.add_argument("--train_ratio", type=float, default=0.90)
    ap.add_argument("--dev_ratio",   type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=str(PROC_DIR), help="Output directory for TSVs")
    args = ap.parse_args()

    repo_root = find_repo_root()
    os.chdir(repo_root)  # make all relative paths consistent
    print(f"[*] Repo root: {repo_root}")

    en_path, fa_path, raw_dir = resolve_raw_paths(repo_root)
    print(f"[*] Raw files:\n    EN: {en_path}\n    FA: {fa_path}")

    print("[*] Loading aligned pairs...")
    df = build_pairs_from_raw(en_path, fa_path, max_lines=args.max_lines)
    print(f"    pairs: {len(df)}")

    print("[*] Preprocessing (EN/FA)...")
    df_tok = preprocess_pairs(df)

    print("[*] Exporting splits...")
    train_p, dev_p, test_p = export_splits(
        df_tok=df_tok,
        out_dir=Path(args.out_dir),
        direction=args.direction,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )

    print("[*] Validating outputs...")
    validate_tsvs([train_p, dev_p, test_p])

    print("\nDone. You can now train/test with these in configs/model_seq2seq.yml:")
    print(f"data:\n  train_tsv: {train_p}\n  dev_tsv:   {dev_p}\n  test_tsv:  {test_p}\n  src_col: \"src\"\n  tgt_col: \"tgt\"")


if __name__ == "__main__":
    main()
