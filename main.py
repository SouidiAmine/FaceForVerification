# evaluate_lfw.py
import numpy as np
from tqdm import tqdm
from lfw_pairs import load_lfw_pairs
from models_facenet import FaceNetEmbedder
from models_arcface import ArcFaceEmbedder
from eval_metrics import cosine_similarity, sweep
import cv2
import matplotlib.pyplot as plt
from att_pairs import load_att_pairs as load_lfw_pairs
from models_vggface import VGGFaceEmbedder


def prep(img):
    # sklearn returns RGB float [0..255]? Often uint8; ensure uint8 BGR for ArcFace, RGB for FaceNet handled inside
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img

def extract_pairs_embeddings(embedder, X1, X2):
    e1, e2 = [], []
    for a,b in tqdm(zip(X1, X2), total=len(X1)):
        img1 = prep(a)
        img2 = prep(b)
        # For ArcFace embedder expects BGR; for FaceNet we passed BGR and convert internally.
        e1.append(embedder.embed(cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)))
        e2.append(embedder.embed(cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)))
    return np.vstack(e1), np.vstack(e2)

def run_eval(model_name="facenet"):
    dataset_path = r"C:\Users\SERAJ\.cache\kagglehub\datasets\kasikrit\att-database-of-faces\versions\2"
    X1, X2, y = load_lfw_pairs(dataset_path)
    # X1, X2, y = load_lfw_pairs("test")
    if model_name == "facenet":
        embedder = FaceNetEmbedder()
    elif model_name == "vggface":
        embedder = VGGFaceEmbedder()
    elif model_name == "arcface":
        embedder = ArcFaceEmbedder()
    else:
        raise ValueError("model_name must be 'facenet' or 'arcface'.")

    E1, E2 = extract_pairs_embeddings(embedder, X1, X2)
    scores = cosine_similarity(E1, E2)
    res = sweep(scores, y, n=2000)
    print(f"[{model_name}] AUC={res['auc']:.4f} | EER={res['eer']:.4f} @ thr={res['eer_thr']:.4f}")

    # Plot ROC
    plt.figure()
    plt.plot(res["fpr"], res["tpr"], label=f"{model_name} (AUC={res['auc']:.3f})")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"roc_{model_name}.png", dpi=180)

    # Plot FAR/FRR vs threshold (and mark EER)
    plt.figure()
    plt.plot(res["thrs"], res["fars"], label="FAR")
    plt.plot(res["thrs"], res["frrs"], label="FRR")
    plt.axvline(res["eer_thr"], linestyle="--")
    plt.title(f"FAR/FRR vs Threshold - {model_name}")
    plt.xlabel("Threshold (cosine similarity)")
    plt.ylabel("Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"far_frr_{model_name}.png", dpi=180)

    return res

if __name__ == "__main__":
    
    r2 = run_eval("vggface")
    r1 = run_eval("facenet")
