# src/data/subtitle_dataset.py
import os
import csv
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# NLTK (optional, for en POS if you enable it)
import nltk
# Uncomment first run if needed:
# nltk.download("averaged_perceptron_tagger")
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

from hazm import Normalizer, word_tokenize, POSTagger
from persian_tools import digits
from word2number import w2n

RAW_DIR = Path("data/raw/TED2020")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

EN_RAW = RAW_DIR / "TED2020.en-fa.en"
FA_RAW = RAW_DIR / "TED2020.en-fa.fa"

CLEAN_CACHE = RAW_DIR / "cleaned_talks.csv"  # optional cache
PAIR_CACHE  = RAW_DIR / "talks.csv"          # raw pair cache


class Subtitle:
    def __init__(self, fa_path: Path = FA_RAW, en_path: Path = EN_RAW):
        self.en_path = Path(en_path)
        self.fa_path = Path(fa_path)
        self.fa_normalizer = Normalizer()
        self.df: Optional[pd.DataFrame] = None

    # ---------- IO ----------
    def _load_to_df(self, max_lines: Optional[int] = None) -> pd.DataFrame:
        if PAIR_CACHE.exists():
            print("CSV cache found, loading:", PAIR_CACHE)
            self.df = pd.read_csv(PAIR_CACHE, dtype=str, encoding="utf-8", quoting=csv.QUOTE_ALL)
            self.df = self.df.dropna(subset=["en", "fa"]).reset_index(drop=True)
        else:
            if not (self.en_path.exists() and self.fa_path.exists()):
                raise FileNotFoundError(
                    f"Missing raw files. Expected:\n  {self.en_path}\n  {self.fa_path}\n"
                    "Run: python scripts/download_ted2020_en_fa.py"
                )
            print("Building CSV from raw files...")
            with open(self.en_path, encoding="utf-8") as en, open(self.fa_path, encoding="utf-8") as fa:
                data = []
                for i, (en_line, fa_line) in enumerate(zip(en, fa)):
                    if max_lines and i >= max_lines:
                        break
                    en_line = en_line.strip()
                    fa_line = fa_line.strip()
                    if en_line and fa_line:
                        data.append((en_line, fa_line))
            self.df = pd.DataFrame(data, columns=["en", "fa"])
            self._save_df(PAIR_CACHE)
            print("Raw pairs saved:", PAIR_CACHE)
        return self.df

    def _save_df(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    # ---------- EN ----------
    def _tokenize_en(self, text: str):
        # keep simple whitespace tokenization to match your original design
        return text.split()

    def _replace_number_words(self, text: str) -> str:
        def convert(match):
            word = match.group(0)
            try:
                return str(w2n.word_to_num(word))
            except ValueError:
                return word
        return re.sub(r"\b[a-z]+\b", convert, text)

    def _preprocess_en(self, text: str):
        text = re.sub(r"[\t\r\n]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"--+", " ", text)
        text = re.sub(r"\.{3,}", " ", text)
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\.\,\?\!]", "", text)
        text = self._replace_number_words(text)
        return self._tokenize_en(text)

    # ---------- FA ----------
    def _tokenize_fa(self, text: str):
        return word_tokenize(text)

    def _preprocess_fa(self, text: str):
        text = self.fa_normalizer.normalize(text)
        text = re.sub(r"[\t\r\n]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"--+", " ", text)
        text = re.sub(r"\.{3,}", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        text = digits.convert_to_en(text)
        text = re.sub(r"[^\w\s\.\,\?\!]", "", text)
        return self._tokenize_fa(text)

    # ---------- Filtering ----------
    def clean_length_ratio(self, max_len=50, min_len=2, ratio=3) -> pd.DataFrame:
        df = self.df.copy()
        df = df[df["en_tokens"].str.len().between(min_len, max_len)]
        df = df[df["fa_tokens"].str.len().between(min_len, max_len)]
        # avoid division by zero
        df = df[(df["fa_tokens"].str.len() != 0)]
        df = df[(df["en_tokens"].str.len() / df["fa_tokens"].str.len()).between(1/ratio, ratio)]
        self.df = df.reset_index(drop=True)
        return self.df

    # ---------- POS (optional) ----------
    def pos_tag_en(self):
        import nltk
        self.df["en_pos"] = self.df["en_tokens"].apply(lambda x: nltk.pos_tag(x))
        return self.df

    def pos_tag_fa(self, model_path: str = "resources/postagger.model"):
        # You must ensure this model exists; otherwise, comment POS out.
        tagger = POSTagger(model=model_path)
        self.df["fa_pos"] = self.df["fa_tokens"].apply(lambda x: tagger.tag(x))
        return self.df

    # ---------- Batch Preprocess ----------
    def _preprocess_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        batch_df = batch_df.copy()
        batch_df["en_tokens"] = batch_df["en"].apply(self._preprocess_en)
        batch_df["fa_tokens"] = batch_df["fa"].apply(self._preprocess_fa)
        return batch_df

    def preprocess(self, batch_size=50000, max_lines: Optional[int] = None, use_cache=True) -> pd.DataFrame:
        if use_cache and CLEAN_CACHE.exists():
            print("Loading cleaned cache:", CLEAN_CACHE)
            self.df = pd.read_csv(CLEAN_CACHE, dtype=str, encoding="utf-8", quoting=csv.QUOTE_ALL)
            # ensure token lists are lists if you reload later; skip for now to keep simple
            return self.df

        self._load_to_df(max_lines=max_lines)
        processed_batches = []
        for i in range(0, len(self.df), batch_size):
            print(f"Processing batch {i}..")
            batch = self.df.iloc[i:i+batch_size]
            processed_batch = self._preprocess_batch(batch)
            processed_batches.append(processed_batch)

        self.df = pd.concat(processed_batches, ignore_index=True)
        # Save a “clean” CSV (tokens are Python lists if saved naively; keeping simple df here)
        self._save_df(CLEAN_CACHE)
        print("Cleaned pairs cached at:", CLEAN_CACHE)
        return self.df

    # ---------- Export to seq2seq TSV ----------
    def export_splits(self, train_ratio=0.9, dev_ratio=0.05, seed=42) -> Tuple[Path, Path, Path]:
        """Create train/dev/test TSVs (src=en, tgt=fa) under data/processed/."""
        df = self.df.copy()
        # build plain strings back from tokens for src/tgt
        df["src"] = df["en_tokens"].apply(lambda toks: " ".join(toks))
        df["tgt"] = df["fa_tokens"].apply(lambda toks: " ".join(toks))

        df = df[["src", "tgt"]].sample(frac=1.0, random_state=seed).reset_index(drop=True)

        n = len(df)
        n_train = int(n * train_ratio)
        n_dev   = int(n * dev_ratio)
        train_df = df.iloc[:n_train]
        dev_df   = df.iloc[n_train:n_train+n_dev]
        test_df  = df.iloc[n_train+n_dev:]

        out_train = PROC_DIR / "train.tsv"
        out_dev   = PROC_DIR / "dev.tsv"
        out_test  = PROC_DIR / "test.tsv"

        train_df.to_csv(out_train, sep="\t", index=False)
        dev_df.to_csv(out_dev, sep="\t", index=False)
        test_df.to_csv(out_test, sep="\t", index=False)

        print("Saved splits:")
        print("  ", out_train, len(train_df))
        print("  ", out_dev,   len(dev_df))
        print("  ", out_test,  len(test_df))
        return out_train, out_dev, out_test


def build_ted2020_splits(max_lines: Optional[int] = 100_000):
    """End-to-end convenience function."""
    sub = Subtitle()
    sub.preprocess(batch_size=50_000, max_lines=max_lines, use_cache=False)
    sub.clean_length_ratio(max_len=50, min_len=2, ratio=3)
    return sub.export_splits()


if __name__ == "__main__":
    # Example run:
    build_ted2020_splits(max_lines=100_000)
