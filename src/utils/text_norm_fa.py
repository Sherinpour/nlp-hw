import re

ARABIC_TO_PERSIAN = str.maketrans({"ي":"ی","ك":"ک"})
PUNCT_SPACES = re.compile(r"\s+")

def normalize_fa(text:str):
    text = text.strip().translate(ARABIC_TO_PERSIAN)
    text = re.sub(r"\u200c{2,}", "\u200c", text)  # نرمال‌سازی نیم‌فاصله
    text = PUNCT_SPACES.sub(" ", text)
    return text

def basic_clean_pair(src:str, tgt:str):
    return normalize_fa(src), normalize_fa(tgt)
