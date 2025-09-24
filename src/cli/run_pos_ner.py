import argparse
from src.utils.io import load_yaml
from src.linguistics.pos_tagging import pos_tag
from src.linguistics.ner import ner_tag

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/pos_ner.yml")
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    print("== POS ==")
    print(pos_tag(args.text, args.cfg))
    print("\n== NER ==")
    print(ner_tag(args.text, args.cfg))

if __name__ == "__main__":
    main()
