# fichier: att_pairs.py
import os
import cv2
import numpy as np
import random

def load_att_faces(root):
    """Loads AT&T (ORL) face dataset from the given root path."""
    images, labels = [], []
    for person_dir in sorted(os.listdir(root)):
        person_path = os.path.join(root, person_dir)
        if not os.path.isdir(person_path):
            continue
        label = int(person_dir[1:])  # 's1' -> 1
        for file in os.listdir(person_path):
            if file.endswith(".pgm"):
                img = cv2.imread(os.path.join(person_path, file), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (160, 160))  # match FaceNet input size
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # make 3 channels
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)
def make_att_pairs(images, labels, n_same=200, n_diff=200, seed=42):
    """Generate same-person and different-person pairs."""
    random.seed(seed)
    np.random.seed(seed)
    pairs_X1, pairs_X2, pairs_y = [], [], []

    # Group images by label
    people = {}
    for img, label in zip(images, labels):
        people.setdefault(label, []).append(img)

    # Positive pairs
    for label, imgs in people.items():
        if len(imgs) < 2:
            continue
        combos = [(imgs[i], imgs[j]) for i in range(len(imgs)) for j in range(i+1, len(imgs))]
        random.shuffle(combos)
        for a, b in combos[:min(n_same // len(people), len(combos))]:
            pairs_X1.append(a)
            pairs_X2.append(b)
            pairs_y.append(1)

    # Negative pairs
    all_labels = list(people.keys())
    for _ in range(n_diff):
        l1, l2 = random.sample(all_labels, 2)
        a = random.choice(people[l1])
        b = random.choice(people[l2])
        pairs_X1.append(a)
        pairs_X2.append(b)
        pairs_y.append(0)

    return np.array(pairs_X1), np.array(pairs_X2), np.array(pairs_y)


def load_att_pairs(root, n_same=200, n_diff=200):
    """High-level function returning pairs like fetch_lfw_pairs()."""
    images, labels = load_att_faces(root)
    X1, X2, y = make_att_pairs(images, labels, n_same, n_diff)
    print(f"✅ Generated {len(y)} pairs ({y.sum()} same, {len(y)-y.sum()} different)")
    return X1, X2, y

