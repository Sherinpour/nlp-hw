from typing import List
import math

def _ngrams(tokens: List[str], n: int):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def corpus_bleu(list_of_references, hypotheses, max_n=4, smooth=1):
    # simple BLEU for quick sanity checks (not sacrebleu)
    weights = [1.0/max_n]*max_n
    p_ns = []
    for n in range(1, max_n+1):
        match, total = 0, 0
        for refs, hyp in zip(list_of_references, hypotheses):
            hyp_ngrams = _ngrams(hyp, n)
            total += max(len(hyp_ngrams), 1)
            ref_counts = {}
            for ref in refs:
                for ng in _ngrams(ref, n):
                    ref_counts[ng] = max(ref_counts.get(ng, 0), 1 + ref_counts.get(ng, 0))
            # clip counts
            for ng in hyp_ngrams:
                match += min(hyp_ngrams.count(ng), ref_counts.get(ng, 0))
        p_n = (match + smooth) / (total + smooth)
        p_ns.append(p_n)
    bp = 1.0  # skip brevity penalty for simplicity
    score = bp * math.exp(sum(w * math.log(p) for w, p in zip(weights, p_ns)))
    return score * 100.0
