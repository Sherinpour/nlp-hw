from transformers import AutoTokenizer

def tokenize_wordpiece(text: str, model_name: str = "bert-base-multilingual-cased", add_special_tokens=False):
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = tok(text, add_special_tokens=add_special_tokens)
    return tok.convert_ids_to_tokens(enc["input_ids"])
