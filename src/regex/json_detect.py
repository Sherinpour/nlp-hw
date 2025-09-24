import re, json
from src.utils.io import load_yaml

def looks_like_json(text: str, cfg_path="configs/regex.yml") -> bool:
    cfg = load_yaml(cfg_path)
    boundary = cfg["json"]["loose_boundary"]
    validate = cfg["json"].get("validate_with_json_loads", True)
    if not re.compile(boundary, re.DOTALL).match(text or ""):
        return False
    if not validate:
        return True
    try:
        json.loads(text)
        return True
    except Exception:
        return False
