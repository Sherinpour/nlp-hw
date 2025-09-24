import re
from typing import List, Tuple, Pattern
from src.utils.io import load_yaml

def compile_date_patterns(cfg_path="configs/regex.yml") -> List[Pattern]:
    cfg = load_yaml(cfg_path)
    months = cfg["dates"]["months_regex"]
    words = cfg["dates"]["words_day"]
    ymd_sep = cfg["dates"]["ymd_sep"]
    d_mon_y = cfg["dates"]["d_mon_y"].replace("__MONTHS__", months)
    spoken  = cfg["dates"]["spoken"].replace("__MONTHS__", months).replace("__WORDS_DAY__", words)
    return [re.compile(p) for p in (ymd_sep, d_mon_y, spoken)]

def find_dates(text: str, cfg_path="configs/regex.yml") -> List[Tuple[str, Tuple[int,int]]]:
    spans = []
    for pat in compile_date_patterns(cfg_path):
        for m in pat.finditer(text):
            spans.append((m.group(), m.span()))
    return spans
