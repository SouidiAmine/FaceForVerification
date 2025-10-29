# models_facenet.py
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceNetEmbedder:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    @staticmethod
    def _preprocess(img):
        # img: HxWx3 uint8 (RGB expected by FaceNet)
        if img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb = img
        rgb = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(rgb).permute(2,0,1).float() / 255.0
        # Standardization used in facenet-pytorch examples
        mean = t.mean([1,2], keepdim=True)
        std  = t.std([1,2], keepdim=True).clamp(min=1e-6)
        t = (t - mean)/std
        return t

    def embed(self, img):
        t = self._preprocess(img)[None].to(device)
        with torch.no_grad():
            emb = self.model(t).cpu().numpy().squeeze()
        # L2-normalize
        return emb / (np.linalg.norm(emb) + 1e-12)
