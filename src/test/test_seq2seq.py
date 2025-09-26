import os
import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn

from src.utils.io import load_yaml, load_json, ensure_dir
from src.utils.seed import set_seed
from src.utils.metrics import corpus_bleu
from src.train.train_seq2seq import (
    make_loader,           # reuse your dataset/loader
    ids_to_tokens,         # id->token utility
    greedy_decode,         # greedy decoder (expects sos_idx/eos_idx args)
    eval_epoch,            # loss on a loader
)
from src.models.seq2seq_lstm import Encoder, Decoder, Seq2Seq


def _select_device(cfg_device: str) -> torch.device:
    if cfg_device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif cfg_device == "mps":
        return torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:  # "auto" or anything else
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


@torch.no_grad()
def evaluate_greedy(
    model: torch.nn.Module,
    loader,
    inv_tgt_vocab: Dict[int, str],
    eos_token: str,
    max_len: int,
    device: torch.device,
    sos_idx: int,
    eos_idx: int,
):
    model.eval()
    hyp_tokens, ref_tokens = [], []
    all_preds_text: List[str] = []

    for src, tgt in loader:
        src = src.to(device)
        out = greedy_decode(model, src, max_len, sos_idx, eos_idx).cpu().tolist()
        tgt = tgt.cpu().tolist()

        for i, tgt_ids in enumerate(tgt):
            hyp = ids_to_tokens(out[i], inv_tgt_vocab, eos_token)
            ref = ids_to_tokens(tgt_ids[1:], inv_tgt_vocab, eos_token)  # drop <s>

            hyp_tokens.append(hyp)
            ref_tokens.append([ref])
            all_preds_text.append(" ".join(hyp))

    bleu = corpus_bleu(ref_tokens, hyp_tokens)
    return bleu, all_preds_text


def main(
    config_path: str = "configs/model_seq2seq.yml",
    ckpt_path: str = "artifacts/checkpoints/seq2seq_lstm_baseline/best.pt",
    out_pred_path: Optional[str] = None,
):
    # ---- Config / seed / dirs ----
    cfg = load_yaml(config_path)
    set_seed(cfg["experiment"]["seed"])

    save_dir = Path(cfg["experiment"]["save_dir"])
    log_dir  = Path(cfg["experiment"]["log_dir"])
    ensure_dir(save_dir)
    ensure_dir(log_dir)
    pred_dir = save_dir / "pred"
    ensure_dir(pred_dir)

    if out_pred_path is None:
        out_pred_path = str(pred_dir / "test.pred.txt")

    # ---- Device ----
    device = _select_device(cfg.get("training", {}).get("device", "auto"))
    print(f"[*] Using device: {device}")

    # ---- Vocab & specials ----
    vocabs = load_json(cfg["vocab"]["save_to"])
    src_vocab, tgt_vocab = vocabs["src"], vocabs["tgt"]
    inv_tgt = {v: k for k, v in tgt_vocab.items()}  # id -> token

    sp = cfg["tokenization"]["special_tokens"]
    pad_idx_src = src_vocab[sp["pad"]]      # follow training choice
    sos_idx     = src_vocab[sp["sos"]]      # match your training
    eos_idx     = src_vocab[sp["eos"]]      # match your training

    # ---- Model (same as training) ----
    enc = Encoder(
        vocab_size=len(src_vocab),
        emb_dim=cfg["model"]["encoder"]["emb_dim"],
        hid_dim=cfg["model"]["encoder"]["hidden_dim"],
        pad_idx=pad_idx_src,
        num_layers=cfg["model"]["encoder"]["num_layers"],
        dropout=cfg["model"]["encoder"]["dropout"],
    )
    dec = Decoder(
        vocab_size=len(tgt_vocab),
        emb_dim=cfg["model"]["decoder"]["emb_dim"],
        hid_dim=cfg["model"]["decoder"]["hidden_dim"],
        pad_idx=pad_idx_src,
        num_layers=cfg["model"]["decoder"]["num_layers"],
        dropout=cfg["model"]["decoder"]["dropout"],
    )
    model = Seq2Seq(
        enc, dec,
        sos_idx=sos_idx,
        eos_idx=eos_idx,
        teacher_forcing=cfg["model"]["teacher_forcing"],
    ).to(device)

    # ---- Load checkpoint ----
    print(f"[*] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))

    # ---- Test loader ----
    test_tsv = cfg["data"]["test_tsv"]
    if not os.path.exists(test_tsv):
        raise FileNotFoundError(f"Missing test TSV: {test_tsv}")
    test_loader = make_loader(
        test_tsv, cfg, src_vocab, tgt_vocab,
        batch_size=cfg["evaluation"].get("batch_size", 32),
        shuffle=False,
    )

    # ---- Test loss ----
    crit = nn.CrossEntropyLoss(ignore_index=pad_idx_src)
    test_loss = eval_epoch(model, test_loader, crit, device)
    print(f"[*] Test loss: {test_loss:.4f}")

    # ---- Greedy BLEU + predictions ----
    greedy_max_len = cfg["evaluation"]["greedy_max_len"]
    bleu, preds_text = evaluate_greedy(
        model=model,
        loader=test_loader,
        inv_tgt_vocab=inv_tgt,
        eos_token=sp["eos"],
        max_len=greedy_max_len,
        device=device,
        sos_idx=sos_idx,
        eos_idx=eos_idx,
    )
    print(f"[*] Test BLEU (greedy): {bleu:.2f}")

    # ---- Save outputs ----
    with open(out_pred_path, "w", encoding="utf-8") as f:
        for line in preds_text:
            f.write(line.strip() + "\n")
    print(f"[✔] Predictions -> {out_pred_path}")

    metrics_path = save_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"test_loss": float(test_loss), "bleu_greedy": float(bleu)},
            f, ensure_ascii=False, indent=2
        )
    print(f"[✔] Metrics     -> {metrics_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_seq2seq.yml")
    ap.add_argument("--ckpt",   required=True, help="Path to trained checkpoint (e.g., artifacts/checkpoints/.../best.pt)")
    ap.add_argument("--out_pred", default=None, help="Optional path for predictions file")
    args = ap.parse_args()

    main(config_path=args.config, ckpt_path=args.ckpt, out_pred_path=args.out_pred)
