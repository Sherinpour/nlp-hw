import re
from typing import List, Tuple
from src.utils.io import load_yaml

def extract_ids_classes(html: str, cfg_path="configs/regex.yml") -> Tuple[List[str], List[str]]:
    cfg = load_yaml(cfg_path)
    id_re = re.compile(cfg["html"]["id_regex"])
    class_re = re.compile(cfg["html"]["class_regex"])
    ids = id_re.findall(html)
    classes = []
    for c in class_re.findall(html):
        classes.extend(c.split())
    return ids, classes
