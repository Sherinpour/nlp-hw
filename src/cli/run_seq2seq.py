import argparse
from src.preprocess.clean_parallel import clean_parallel
from src.preprocess.build_vocab import build_vocab
from src.train.train_seq2seq import main as train_main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep_cfg", default="configs/preprocess.yml")
    ap.add_argument("--model_cfg", default="configs/model_seq2seq.yml")
    ap.add_argument("--skip_clean", action="store_true")
    ap.add_argument("--skip_vocab", action="store_true")
    ap.add_argument("--test_only", action="store_true",help = "Evaluate on test set using a trained checkpoint (no training).")
    ap.add_argument("--ckpt", default="", help="Path to trained checkpoint for --test_only")

    args = ap.parse_args()

    if not args.skip_clean:
        print("[*] cleaning parallel data…")
        clean_parallel(args.prep_cfg)
    if not args.skip_vocab:
        print("[*] building vocab…")
        build_vocab(args.model_cfg)
    print("[*] training…")
    train_main(args.model_cfg)

    # test-only fast exit
    if args.test_only:
        from src.train.train_seq2seq import eval_on_test
    if not args.ckpt:
        raise SystemExit("Provide --ckpt path to a trained model for --test_only.")
    eval_on_test(args.model_cfg, args.ckpt)

    raise SystemExit(0)

if __name__ == "__main__":
    main()
