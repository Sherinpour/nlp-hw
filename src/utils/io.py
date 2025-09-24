import os
import csv
import json
from typing import Dict, Any, Iterable, List
import yaml

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_tsv(path: str) -> List[Dict[str, str]]:
    with open(path, newline='', encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def write_tsv(path: str, rows: Iterable[Dict[str, str]], fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline='', encoding="utf-8") as g:
        wr = csv.DictWriter(g, fieldnames=fieldnames, delimiter="\t")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def write_lines(path: str, lines: Iterable[str]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
