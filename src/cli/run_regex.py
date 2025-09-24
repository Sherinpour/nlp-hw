import argparse
from src.regex.fa_dates import find_dates
from src.regex.abbreviations import find_abbr
from src.regex.html_attrs import extract_ids_classes
from src.regex.json_detect import looks_like_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, help="input text or html or json")
    ap.add_argument("--mode", choices=["dates","abbr","html","json"], required=True)
    ap.add_argument("--cfg", default="configs/regex.yml")
    args = ap.parse_args()

    if args.mode == "dates":
        print(find_dates(args.text, cfg_path=args.cfg))
    elif args.mode == "abbr":
        print(find_abbr(args.text, cfg_path=args.cfg))
    elif args.mode == "html":
        ids, classes = extract_ids_classes(args.text, cfg_path=args.cfg)
        print("ids:", ids)
        print("classes:", classes)
    elif args.mode == "json":
        print(looks_like_json(args.text, cfg_path=args.cfg))

if __name__ == "__main__":
    main()
