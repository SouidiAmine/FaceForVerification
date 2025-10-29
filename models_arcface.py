# models_arcface.py
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

class ArcFaceEmbedder:
    def __init__(self, ctx_id=0):
        self.app = FaceAnalysis(name='buffalo_l')  # good default
        self.app.prepare(ctx_id=ctx_id if cv2.cuda.getCudaEnabledDeviceCount() > 0 else -1)

    def embed(self, img):
        # img: BGR uint8
        faces = self.app.get(img)
        if len(faces) == 0:
            # fallback: center crop if no face detected (or return zeros)
            h, w = img.shape[:2]
            y0,x0 = max(0,h//4), max(0,w//4)
            crop = img[y0:3*h//4, x0:3*w//4]
            faces = self.app.get(crop)
            if len(faces)==0:
                return np.zeros(512, dtype=np.float32)
            f = faces[0]
        else:
            f = faces[0]
        emb = f.normed_embedding  # already L2-normalized
        return emb.astype(np.float32)
