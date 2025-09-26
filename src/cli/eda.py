#!/usr/bin/env python3
"""Headless EDA for seq2seq TSV data (src/tgt).

Usage:
  python eda.py --train data/processed/train.tsv --dev data/processed/dev.tsv --test data/processed/test.tsv --out artifacts/eda
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def read_tsv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p, sep='\t', dtype=str).fillna('')

def to_tokens(s: str):
    return s.split()

def add_lengths(df: pd.DataFrame, src_col='src', tgt_col='tgt') -> pd.DataFrame:
    out = df.copy()
    out['src_char_len'] = out[src_col].str.len()
    out['tgt_char_len'] = out[tgt_col].str.len()
    out['src_tok_len']  = out[src_col].apply(lambda x: len(to_tokens(x)))
    out['tgt_tok_len']  = out[tgt_col].apply(lambda x: len(to_tokens(x)))
    return out

def plot_hist(series, title, outdir: Path, fname: str, bins=50):
    plt.figure()
    series.hist(bins=bins)
    plt.title(title)
    plt.xlabel('Length')
    plt.ylabel('Count')
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / fname, bbox_inches='tight')
    plt.close()

def vocab_counter(df: pd.DataFrame, col: str) -> Counter:
    c = Counter()
    for s in df[col].tolist():
        c.update(s.split())
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--dev', type=Path, required=True)
    ap.add_argument('--test', type=Path, required=True)
    ap.add_argument('--src_col', default='src')
    ap.add_argument('--tgt_col', default='tgt')
    ap.add_argument('--out', type=Path, default=Path('artifacts/eda'))
    args = ap.parse_args()

    train = read_tsv(args.train)
    dev   = read_tsv(args.dev)
    test  = read_tsv(args.test)

    for name, df in [('train', train), ('dev', dev), ('test', test)]:
        if args.src_col not in df.columns or args.tgt_col not in df.columns:
            raise ValueError(f"{name}.tsv must have columns '{args.src_col}' and '{args.tgt_col}' – got {list(df.columns)}")

    # sizes
    report = []
    for name, df in [('train', train), ('dev', dev), ('test', test)]:
        report.append({
            'split': name,
            'rows': len(df),
            'null_src': int(df[args.src_col].isnull().sum()),
            'null_tgt': int(df[args.tgt_col].isnull().sum()),
            'empty_src': int((df[args.src_col] == '').sum()),
            'empty_tgt': int((df[args.tgt_col] == '').sum()),
        })
    sizes_df = pd.DataFrame(report)
    args.out.mkdir(parents=True, exist_ok=True)
    sizes_df.to_csv(args.out / 'sizes.csv', index=False)

    # lengths & plots
    train_len = add_lengths(train, args.src_col, args.tgt_col)
    plot_hist(train_len['src_tok_len'], 'Train: Source token length', args.out, 'train_src_tok_len.png')
    plot_hist(train_len['tgt_tok_len'], 'Train: Target token length', args.out, 'train_tgt_tok_len.png')
    plot_hist(train_len['src_char_len'], 'Train: Source char length', args.out, 'train_src_char_len.png')
    plot_hist(train_len['tgt_char_len'], 'Train: Target char length', args.out, 'train_tgt_char_len.png')

    # vocab
    src_v = vocab_counter(train, args.src_col)
    tgt_v = vocab_counter(train, args.tgt_col)
    pd.DataFrame(src_v.most_common(200), columns=['token','count']).to_csv(args.out / 'top_src_tokens.csv', index=False)
    pd.DataFrame(tgt_v.most_common(200), columns=['token','count']).to_csv(args.out / 'top_tgt_tokens.csv', index=False)

    # examples
    ex = train.sample(min(10, len(train)), random_state=42)[[args.src_col, args.tgt_col]]
    ex.to_csv(args.out / 'random_examples.csv', index=False)

    print(f"EDA reports saved to: {args.out}")

if __name__ == '__main__':
    main()
