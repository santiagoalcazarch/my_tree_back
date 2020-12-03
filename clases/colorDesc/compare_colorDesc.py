import json
import cv2
from math import *
import numpy as np
from prod.ColorDescriptor import ColorDescriptor

def euclidean_distance(x,y):
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

with open('colorDes.json') as f:
  data = json.load(f)

clientImage = cv2.imread("./test.jpg")

cd = ColorDescriptor((8, 12, 3))
features = cd.describe( clientImage )
features = [float(f) for f in features]

best = None
best_data = None
print("Comparing...")
for carga in data:
  distance = euclidean_distance(normalize(features), normalize(carga['vector']))
  if best is None or distance < best:
    best = distance
    best_data = carga
print(1-best)
print(best_data["imagePath"])
