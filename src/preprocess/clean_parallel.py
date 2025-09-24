import random
from typing import List, Dict
from src.utils.io import read_tsv, write_tsv, load_yaml
from src.utils.text_norm_fa import basic_clean_pair

def _filter_len_ok(s: str, min_len: int, max_len: int) -> bool:
    L = len((s or "").split())
    return (L >= min_len) and (L <= max_len)

def clean_parallel(cfg_path="configs/preprocess.yml") -> None:
    cfg = load_yaml(cfg_path)
    inp = cfg["io"]["input_tsv"]
    out = cfg["io"]["output_tsv"]

    rows = read_tsv(inp)
    cleaned: List[Dict[str, str]] = []
    for r in rows:
        src, tgt = basic_clean_pair(r["src"], r["tgt"])
        if not (_filter_len_ok(src, cfg["filters"]["min_len"], cfg["filters"]["max_len"]) and
                _filter_len_ok(tgt, cfg["filters"]["min_len"], cfg["filters"]["max_len"])):
            continue
        if cfg["cleanup"]["drop_if_identical_src_tgt"] and src == tgt:
            continue
        cleaned.append({"src": src, "tgt": tgt})

    if cfg["cleanup"]["drop_duplicates"]:
        seen = set()
        uniq = []
        for r in cleaned:
            key = (r["src"], r["tgt"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        cleaned = uniq

    if cfg["splits"]["create_splits"]:
        seed = cfg["splits"]["shuffle_seed"]
        random.Random(seed).shuffle(cleaned)
        n = len(cleaned)
        a, b = cfg["splits"]["ratios"][:2]
        n_train = int(n * a)
        n_dev = int(n * b)
        train = cleaned[:n_train]
        dev = cleaned[n_train:n_train+n_dev]
        test = cleaned[n_train+n_dev:]
        write_tsv(cfg["io"]["output_tsv"], train, ["src","tgt"])
        write_tsv(cfg["splits"]["dev_out"], dev, ["src","tgt"])
        write_tsv(cfg["splits"]["test_out"], test, ["src","tgt"])
    else:
        write_tsv(out, cleaned, ["src","tgt"])

if __name__ == "__main__":
    clean_parallel()
