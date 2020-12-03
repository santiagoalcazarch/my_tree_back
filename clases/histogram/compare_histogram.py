import json
import cv2
import numpy as np
from math import *

def histogram_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

with open('hist.json') as f:
	data = json.load(f)

clientImage = cv2.imread("test.jpg")
hist, bins = np.histogram(clientImage.ravel(),256,[0,256])
hist = [ float(f) for f in hist ]

compare_hist = []

for carga in data:
	hist_int = histogram_intersection( hist, carga['vector'] )
	compare_hist.append( { 'value': hist_int, 'imagePath': carga['imagePath'] } )

newlist = sorted(compare_hist, key=lambda k: k['value']) 
print(newlist[len(newlist) - 1]["value"])
spplited = newlist[0]["imagePath"].split("/")
print(spplited[len(spplited) - 2])
