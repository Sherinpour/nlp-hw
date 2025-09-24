from pathlib import Path
import re
import sentencepiece as spm

def train_spm(
    input_file: str,
    model_prefix="artifacts/vocab/spm",
    vocab_size=8000,
    model_type="unigram",
    character_coverage=1.0,
    min_vocab_fallback=32,
):
    in_path = Path(input_file).expanduser().resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"SentencePiece input file not found: {in_path}")

    out_prefix = Path(model_prefix).expanduser()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    def _args(vsize: int) -> str:
        return (
            f"--input={in_path} "
            f"--model_prefix={out_prefix} "
            f"--vocab_size={vsize} "
            f"--model_type={model_type} "
            f"--character_coverage={character_coverage}"
        )

    try:
        spm.SentencePieceTrainer.Train(_args(vocab_size))
    except RuntimeError as e:
        msg = str(e)
        # Looks like: "Vocabulary size too high ... set it to a value <= 103"
        if "Vocabulary size too high" in msg:
            m = re.search(r"<=\s*(\d+)", msg)
            if m:
                max_allowed = int(m.group(1))
                new_size = max(min_vocab_fallback, min(max_allowed, vocab_size))
                print(f"[sentencepiece] Lowering vocab_size {vocab_size} → {new_size} (max allowed {max_allowed})")
                spm.SentencePieceTrainer.Train(_args(new_size))
            else:
                raise
        else:
            raise

    return f"{out_prefix}.model"

def tokenize_spm(text: str, model_file="artifacts/vocab/spm.model"):
    sp = spm.SentencePieceProcessor(model_file=model_file)
    return sp.encode(text, out_type=str)
