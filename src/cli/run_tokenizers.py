import argparse
import os
from pathlib import Path

from src.utils.io import load_yaml

def _p(x: str) -> Path:
    return Path(x).expanduser().resolve()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=str(_p(Path(__file__).parents[2] / "configs" / "tokenization.yml")))
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    samples = cfg["comparison"]["samples"]

    if not args.compare:
        print("Use --compare to print tokens for sample sentences from the config.")
        return

    # === WordPiece ===
    print("== WordPiece ==")
    from src.tokenization.wordpiece_demo import tokenize_wordpiece
    wp_name = cfg["wordpiece"]["pretrained_model_name"]
    for s in samples:
        print(s, "->", tokenize_wordpiece(s, wp_name))

    # === BPE (HF tokenizers) ===
    print("\n== BPE ==")
    from src.tokenization.bpe_demo import train_bpe, tokenize_bpe
    bpe_train = cfg["bpe"]["train"]
    bpe_use = cfg["bpe"].get("use", {})  # might be missing
    bpe_path = _p(bpe_use.get("tokenizer_path", bpe_train["save_path"]))

    if not bpe_path.is_file():
        # ensure output dir exists
        bpe_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_files = [str(_p(p)) for p in bpe_train["corpus_files"]]
        train_bpe(
            corpus_files=corpus_files,
            vocab_size=bpe_train["vocab_size"],
            min_frequency=bpe_train["min_frequency"],
            special_tokens=bpe_train["special_tokens"],
            save_path=str(bpe_path),
        )
    for s in samples:
        print(s, "->", tokenize_bpe(s, str(bpe_path)))

    # === SentencePiece ===
    print("\n== SentencePiece ==")
    from src.tokenization.sentencepiece_demo import train_spm, tokenize_spm
    spm_train = cfg["sentencepiece"]["train"]
    spm_use = cfg["sentencepiece"].get("use", {})  # might be missing
    # derive model_file if absent
    spm_model = _p(spm_use.get("model_file", f"{spm_train['model_prefix']}.model"))

    if not spm_model.is_file():
        in_file = _p(spm_train["input_file"])
        if not in_file.is_file():
            raise FileNotFoundError(f"SentencePiece input file not found: {in_file}")
        model_prefix = _p(spm_train["model_prefix"])
        model_prefix.parent.mkdir(parents=True, exist_ok=True)
        # keep your current train_spm signature
        train_spm(
            str(in_file),
            str(model_prefix),
            spm_train["vocab_size"],
            spm_train["model_type"],
            spm_train["character_coverage"],
        )
        spm_model = _p(f"{model_prefix}.model")

    for s in samples:
        print(s, "->", tokenize_spm(s, str(spm_model)))

if __name__ == "__main__":
    main()
