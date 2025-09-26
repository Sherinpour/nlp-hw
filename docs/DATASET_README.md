# Adding the *Special* Dataset to This Repo

There are a few safe ways to include your dataset depending on size and privacy.

## A) Small & shareable (<100 MB total)
Commit it directly under `data/`:
```
data/
  raw/           # untouched originals
  processed/     # train/dev/test TSV with columns: src, tgt
    train.tsv
    dev.tsv
    test.tsv
```

## B) Bigger files (hundreds of MB to a few GB)
Use **Git LFS** so `git clone` stays fast but files live *inside* the repo:

```bash
# one-time setup in this repo
git lfs install
git lfs track "*.tsv" "*.csv" "*.json" "*.zip"

# commit the tracking rules
git add .gitattributes
git commit -m "chore: track data files via Git LFS"

# add your data
git add data/processed/train.tsv data/processed/dev.tsv data/processed/test.tsv
git commit -m "data: add special dataset (LFS)"
git push
```

## C) Private or too large to host
Keep the dataset out of Git, but provide a **downloader**:
- Put a script at `scripts/download_data.py` (or a Makefile target) that fetches the data
- Add checksums and a usage blurb in `README.md`
- Add `data/` to `.gitignore` except `data/processed/*.tsv` if you want to ship only the splits

Example `.gitignore` snippet:
```
data/*
!data/processed/
!data/processed/*.tsv
```

---

## Where the EDA Looks
Both the notebook `EDA_seq2seq.ipynb` and the CLI script `eda.py` expect:
```
data/processed/train.tsv
data/processed/dev.tsv
data/processed/test.tsv
```
with columns `src` and `tgt`. Adjust paths/column names inside if yours differ.
