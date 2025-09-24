from typing import List, Tuple
from src.utils.io import load_yaml

def ner_tag(sentence: str, cfg_path="configs/pos_ner.yml") -> List[Tuple[str, str]]:
    cfg = load_yaml(cfg_path)
    eng = cfg["ner"]["engine"]
    tokens = sentence.split()

    if eng == "stanza":
        import stanza
        st = cfg["ner"]["stanza"]
        nlp = stanza.Pipeline(lang=st["lang"], processors=st["processors"], use_gpu=st.get("use_gpu", False))
        doc = nlp(sentence)
        tags = [(t, "O") for t in tokens]
        for ent in doc.ents:
            # naive projection over whitespace tokens
            for i, t in enumerate(tokens):
                if t in ent.text.split():
                    tags[i] = (t, ent.type)
        return tags

    if eng == "transformers":
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        mname = cfg["ner"]["transformers"]["model_name"]
        device = cfg["ner"]["transformers"].get("device", "auto")
        aggr = cfg["ner"]["transformers"].get("aggregation_strategy", "simple")
        nlp = pipeline("token-classification", model=AutoModelForTokenClassification.from_pretrained(mname),
                       tokenizer=AutoTokenizer.from_pretrained(mname),
                       aggregation_strategy=aggr, device=device)
        out = nlp(sentence)
        # convert spans to token-level BIO-ish (simplified)
        tags = [(t, "O") for t in tokens]
        for ent in out:
            for i, t in enumerate(tokens):
                if t in ent["word"].split():
                    tags[i] = (t, ent["entity_group"])
        return tags

    # placeholder
    return [(t, "O") for t in tokens]
