# eval_metrics.py
import numpy as np
from sklearn.metrics import roc_curve, auc

def cosine_similarity(a, b):
    return np.sum(a*b, axis=1)

def far_frr_at_threshold(scores, labels, thr):
    # labels: 1=same, 0=different ; scores: higher=more similar
    preds_same = (scores >= thr).astype(int)
    # confusion:
    TP = np.sum((preds_same==1) & (labels==1))
    FP = np.sum((preds_same==1) & (labels==0))
    TN = np.sum((preds_same==0) & (labels==0))
    FN = np.sum((preds_same==0) & (labels==1))
    FAR = FP / max(FP+TN, 1)  # impostor accept rate
    FRR = FN / max(TP+FN, 1)  # genuine reject rate
    return FAR, FRR, TP, FP, TN, FN

def sweep(scores, labels, n=1000):
    thrs = np.linspace(scores.min()-1e-6, scores.max()+1e-6, n)
    fars, frrs = [], []
    for t in thrs:
        FAR, FRR, *_ = far_frr_at_threshold(scores, labels, t)
        fars.append(FAR); frrs.append(FRR)
    fars, frrs = np.array(fars), np.array(frrs)
    # EER = FAR == FRR (closest point)
    idx = np.argmin(np.abs(fars - frrs))
    eer = (fars[idx] + frrs[idx]) / 2.0
    eer_thr = thrs[idx]
    # ROC using sklearn: need y_score=similarity
    fpr, tpr, roc_thrs = roc_curve(labels, scores)  # positive class = same
    roc_auc = auc(fpr, tpr)
    return {
        "thrs": thrs, "fars": fars, "frrs": frrs,
        "eer": float(eer), "eer_thr": float(eer_thr),
        "fpr": fpr, "tpr": tpr, "roc_thrs": roc_thrs, "auc": float(roc_auc)
    }
