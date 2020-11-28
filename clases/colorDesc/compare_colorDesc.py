import json
import cv2
from math import *
from prod.ColorDescriptor import ColorDescriptor

def euclidean_distance(x,y):
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

with open('colorDes.json') as f:
  data = json.load(f)

clientImage = cv2.imread("../../archive/leafsnap-dataset/dataset/images/field/abies_concolor/12995307070714.jpg")

cd = ColorDescriptor((8, 12, 3))
features = cd.describe( clientImage )
features = [float(f) for f in features]

for carga in data:
    print( str(euclidean_distance( features, carga['vector'] )) + " - " + carga['imagePath'] )
