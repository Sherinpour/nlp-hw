import os
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.utils.freeze_utils import freeze_all_except_last
from src.utils.io import load_yaml, read_tsv, load_json
from src.utils.io import ensure_dir
from src.utils.seed import set_seed
from src.utils.metrics import corpus_bleu
from src.models.seq2seq_lstm import Encoder, Decoder, Seq2Seq

class ParallelDataset(Dataset):
    def __init__(self, rows: List[Dict[str,str]], src_vocab: Dict[str,int], tgt_vocab: Dict[str,int],
                 max_len_src: int, max_len_tgt: int, sp: Dict[str,str]):
        self.src_ids = [self.encode(r["src"], src_vocab, max_len_src, sp) for r in rows]
        self.tgt_ids = [self.encode_tgt(r["tgt"], tgt_vocab, max_len_tgt, sp) for r in rows]

    @staticmethod
    def encode(text: str, vocab: Dict[str,int], max_len: int, sp: Dict[str,str]) -> List[int]:
        toks = text.split()[:max_len-2]
        ids = [vocab.get(sp["sos"])] + [vocab.get(t, vocab.get(sp["unk"])) for t in toks] + [vocab.get(sp["eos"])]
        ids = ids + [vocab.get(sp["pad"])] * (max_len - len(ids))
        return ids[:max_len]

    @staticmethod
    def encode_tgt(text: str, vocab: Dict[str,int], max_len: int, sp: Dict[str,str]) -> List[int]:
        return ParallelDataset.encode(text, vocab, max_len, sp)

    def __len__(self): return len(self.src_ids)
    def __getitem__(self, i):
        return torch.tensor(self.src_ids[i]), torch.tensor(self.tgt_ids[i])

def make_loader(tsv_path: str, cfg, src_vocab, tgt_vocab, batch_size: int, shuffle: bool) -> DataLoader:
    rows = read_tsv(tsv_path)
    ds = ParallelDataset(
        rows, src_vocab, tgt_vocab,
        cfg["tokenization"]["max_len_src"], cfg["tokenization"]["max_len_tgt"],
        cfg["tokenization"]["special_tokens"]
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def train_epoch(model, loader, crit, opt, device):
    model.train()
    total = 0.0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        opt.zero_grad()
        logits = model(src, tgt)  # (B, T-1, V)
        gold = tgt[:, 1:]
        loss = crit(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    model.eval()
    total = 0.0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt)
        gold = tgt[:, 1:]
        loss = crit(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
        total += loss.item()
    return total / max(1, len(loader))

def ids_to_tokens(ids: List[int], inv_vocab: Dict[int,str], eos_tok: str) -> List[str]:
    toks = []
    for i in ids:
        t = inv_vocab.get(int(i), "<unk>")
        if t == eos_tok:
            break
        toks.append(t)
    return toks

@torch.no_grad()
def greedy_decode(model, src_batch, max_len, sos_idx, eos_idx):
    model.eval()
    h, c = model.enc(src_batch)
    B = src_batch.size(0)
    inp = torch.full((B,1), sos_idx, dtype=torch.long, device=src_batch.device)
    outputs = []
    for _ in range(max_len):
        logits, h, c = model.dec(inp, h, c)
        next_tok = logits.argmax(-1)
        outputs.append(next_tok)
        inp = next_tok
    return torch.cat(outputs, dim=1)  # (B, T)

def main(config_path="configs/model_seq2seq.yml"):
    cfg = load_yaml(config_path)
    set_seed(cfg["experiment"]["seed"])
    ensure_dir(cfg["experiment"]["save_dir"])
    ensure_dir(cfg["experiment"]["log_dir"])

    # replace current device line with:
    if cfg["training"]["device"] == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif cfg["training"]["device"] == "mps":
        device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:  # "auto" or anything else
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # load vocab
    vocabs = load_json(cfg["vocab"]["save_to"])
    src_vocab, tgt_vocab = vocabs["src"], vocabs["tgt"]
    inv_tgt = {v:k for k,v in tgt_vocab.items()}

    sp = cfg["tokenization"]["special_tokens"]
    pad_idx, sos_idx, eos_idx, unk_idx = (src_vocab[sp["pad"]], src_vocab[sp["sos"]], src_vocab[sp["eos"]], src_vocab[sp["unk"]])

    # model
    enc = Encoder(vocab_size=len(src_vocab),
                  emb_dim=cfg["model"]["encoder"]["emb_dim"],
                  hid_dim=cfg["model"]["encoder"]["hidden_dim"],
                  pad_idx=pad_idx,
                  num_layers=cfg["model"]["encoder"]["num_layers"],
                  dropout=cfg["model"]["encoder"]["dropout"])
    dec = Decoder(vocab_size=len(tgt_vocab),
                  emb_dim=cfg["model"]["decoder"]["emb_dim"],
                  hid_dim=cfg["model"]["decoder"]["hidden_dim"],
                  pad_idx=pad_idx,
                  num_layers=cfg["model"]["decoder"]["num_layers"],
                  dropout=cfg["model"]["decoder"]["dropout"])
    model = Seq2Seq(enc, dec, sos_idx=sos_idx, eos_idx=eos_idx, teacher_forcing=cfg["model"]["teacher_forcing"]).to(device)
    freeze_cfg = cfg.get("training", {}).get("freeze", {})
    if freeze_cfg.get("enable", True):
        model = freeze_all_except_last(
            model,
            likely_heads=freeze_cfg.get("head_names"),
            last_n_fallback=freeze_cfg.get("last_n_fallback", 1),
            enable_gc=freeze_cfg.get("gradient_checkpointing", True),
        )
    # data
    train_loader = make_loader(cfg["data"]["train_tsv"], cfg, src_vocab, tgt_vocab, cfg["training"]["batch_size"], True)
    dev_loader   = make_loader(cfg["data"]["dev_tsv"],   cfg, src_vocab, tgt_vocab, cfg["training"]["batch_size"], False)

    # training setup
    crit = nn.CrossEntropyLoss(ignore_index=pad_idx)
    if cfg["training"]["optimizer"]["name"].lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["optimizer"]["lr"], weight_decay=cfg["training"]["optimizer"]["weight_decay"])
    else:
        opt = torch.optim.SGD(model.parameters(), lr=cfg["training"]["optimizer"]["lr"], momentum=0.9)

    best = 1e9
    for epoch in range(1, cfg["training"]["epochs"]+1):
        tr = train_epoch(model, train_loader, crit, opt, device)
        dv = eval_epoch(model, dev_loader, crit, device)
        print(f"[epoch {epoch}] train_loss={tr:.4f} dev_loss={dv:.4f}")
        if dv < best:
            best = dv
            torch.save(model.state_dict(), os.path.join(cfg["experiment"]["save_dir"], "best.pt"))

    # quick BLEU on dev (greedy)
    dev_rows = read_tsv(cfg["data"]["dev_tsv"])
    dev_loader_eval = make_loader(cfg["data"]["dev_tsv"], cfg, src_vocab, tgt_vocab, batch_size=32, shuffle=False)
    hyp_tokens, ref_tokens = [], []
    for src, tgt in dev_loader_eval:
        src = src.to(device)
        out = greedy_decode(model, src, cfg["evaluation"]["greedy_max_len"], sos_idx, eos_idx).cpu().tolist()
        for i, tgt_ids in enumerate(tgt.tolist()):
            hyp = ids_to_tokens(out[i], inv_tgt, cfg["tokenization"]["special_tokens"]["eos"])
            ref = ids_to_tokens(tgt_ids[1:], inv_tgt, cfg["tokenization"]["special_tokens"]["eos"])  # exclude sos
            hyp_tokens.append(hyp)
            ref_tokens.append([ref])
    bleu = corpus_bleu(ref_tokens, hyp_tokens)
    print(f"Dev BLEU (greedy): {bleu:.2f}")

if __name__ == "__main__":
    main()
