from collections import Counter
from typing import Dict, List
from src.utils.io import read_tsv, save_json
from src.utils.io import load_yaml

def tokenize_simple(s: str) -> List[str]:
    return s.split()

def build_vocab(cfg_model="configs/model_seq2seq.yml") -> Dict[str, Dict[str,int]]:
    cfg = load_yaml(cfg_model)
    train = read_tsv(cfg["data"]["train_tsv"])
    sp = cfg["tokenization"]["special_tokens"]
    specials = [sp["pad"], sp["sos"], sp["eos"], sp["unk"]]

    src_counter, tgt_counter = Counter(), Counter()
    for r in train:
        src_tokens = tokenize_simple(r["src"])
        tgt_tokens = tokenize_simple(r["tgt"])
        src_counter.update(src_tokens)
        tgt_counter.update(tgt_tokens)

    def make_vocab(counter: Counter, max_size: int) -> Dict[str, int]:
        vocab = {}
        for i, tok in enumerate(specials):
            vocab[tok] = i
        idx = len(specials)
        for tok, _ in counter.most_common(max_size - len(specials)):
            if tok in vocab:
                continue
            vocab[tok] = idx; idx += 1
        return vocab

    src_vocab = make_vocab(src_counter, cfg["vocab"]["vocab_size_src"])
    tgt_vocab = make_vocab(tgt_counter, cfg["vocab"]["vocab_size_tgt"])
    save_json(cfg["vocab"]["save_to"], {"src": src_vocab, "tgt": tgt_vocab})
    return {"src": src_vocab, "tgt": tgt_vocab}

if __name__ == "__main__":
    build_vocab()
