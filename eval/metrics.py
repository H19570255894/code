# eval/metrics.py
from typing import Set, Dict

def prf_jaccard(pred: Set[int], gold: Set[int]) -> Dict[str, float]:
    inter = len(pred & gold)
    if not pred:
        precision = 0.0
    else:
        precision = inter / len(pred)
    if not gold:
        recall = 0.0
    else:
        recall = inter / len(gold)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    union = len(pred | gold)
    jacc = 0.0 if union == 0 else inter / union
    return dict(P=precision, R=recall, F1=f1, Jaccard=jacc)
