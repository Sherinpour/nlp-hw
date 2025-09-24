from typing import List, Tuple
from src.utils.io import load_yaml

def pos_tag(sentence: str, cfg_path="configs/pos_ner.yml") -> List[Tuple[str, str]]:
    cfg = load_yaml(cfg_path)
    engine = cfg["pos"]["engine"]
    if engine == "hazm":
        from hazm import POSTagger, word_tokenize
        tagger = POSTagger(model=cfg["pos"]["hazm"]["model_path"])
        tokens = word_tokenize(sentence)
        return list(zip(tokens, tagger.tag(tokens)))
    elif engine == "stanza":
        import stanza
        stcfg = cfg["pos"]["stanza"]
        nlp = stanza.Pipeline(lang=stcfg["lang"], processors=stcfg["processors"], use_gpu=stcfg.get("use_gpu", False))
        doc = nlp(sentence)
        pairs = []
        for s in doc.sentences:
            for w in s.words:
                pairs.append((w.text, w.xpos or w.upos or "X"))
        return pairs
    else:
        # fallback: whitespace split with "X"
        toks = sentence.split()
        return [(t, "X") for t in toks]
