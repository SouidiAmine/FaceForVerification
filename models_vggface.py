# fichier: models_vggface.py
import numpy as np
from deepface import DeepFace

class VGGFaceEmbedder:
    def __init__(self, model_name="VGG-Face"):
        self.model_name = model_name

    def embed(self, img):
        # DeepFace retourne un vecteur d'embedding à partir d'une image RGB
        embedding = DeepFace.represent(img, model_name=self.model_name, enforce_detection=False)[0]["embedding"]
        emb = np.array(embedding)
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        return emb
