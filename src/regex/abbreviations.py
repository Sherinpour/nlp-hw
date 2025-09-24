import re
from typing import List
from src.utils.io import load_yaml

def compile_abbr_pattern(cfg_path="configs/regex.yml") -> re.Pattern:
    cfg = load_yaml(cfg_path)
    toks = cfg["abbreviations"]["tokens"]
    union = "|".join(toks)
    pat = cfg["abbreviations"]["pattern_template"].replace("__TOKENS__", union)
    return re.compile(pat)

def find_abbr(text: str, cfg_path="configs/regex.yml") -> List[str]:
    pat = compile_abbr_pattern(cfg_path)
    return [m.group() for m in pat.finditer(text)]
