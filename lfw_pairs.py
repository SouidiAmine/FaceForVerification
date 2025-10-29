# lfw_pairs.py
import numpy as np
from sklearn.datasets import fetch_lfw_pairs

def load_lfw_pairs(split="test", people_min=0):
    data = fetch_lfw_pairs(subset=split, color=True, resize=0.5)  # shape: (n, 2, h, w, 3)
    X1 = data.pairs[:,0]
    X2 = data.pairs[:,1]
    y  = data.target.astype(np.int32)  # 1 = same, 0 = different
    return X1, X2, y