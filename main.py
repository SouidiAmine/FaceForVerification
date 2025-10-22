from sklearn.datasets import fetch_lfw_people

lfw = fetch_lfw_people(color=True, resize=1.0, min_faces_per_person=2)

print(lfw.images.shape)   # (nombre_images, hauteur, largeur, 3)
print(lfw.target_names[:5])