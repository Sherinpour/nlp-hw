from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

def _to_paths(files):
    return [str(Path(f).expanduser().resolve()) for f in files]

def train_bpe(
    corpus_files,
    vocab_size=8000,
    min_frequency=2,
    special_tokens=None,
    save_path="artifacts/vocab/bpe_tok.json",
):
    special_tokens = special_tokens or ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    # 1) Ensure output directory exists
    save_path = Path(save_path).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 2) Normalize & sanity-check corpus files
    corpus_files = _to_paths(corpus_files)
    for f in corpus_files:
        p = Path(f)
        if not p.is_file():
            raise FileNotFoundError(f"Corpus file not found: {p}")

    # 3) Train
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )
    tok.train(files=corpus_files, trainer=trainer)

    # 4) Save tokenizer
    tok.save(str(save_path))
    return str(save_path)

def tokenize_bpe(text: str, tok_path="artifacts/vocab/bpe_tok.json"):
    tok_path = str(Path(tok_path).expanduser().resolve())
    p = Path(tok_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Tokenizer file not found: {p}\n"
            f"→ Train it first, e.g.: train_bpe(['data/processed/train.txt'], save_path='{p}')"
        )
    tok = Tokenizer.from_file(tok_path)
    return tok.encode(text).tokens
