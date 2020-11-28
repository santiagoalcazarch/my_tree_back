import numpy as np
import glob
import cv2
import json

# open the output index file for writing
output = open("hist.json", "w")

write = []

# use glob to grab the image paths and loop over them
for imagePath in glob.glob("../../archive/leafsnap-dataset/dataset/images/field/abies_concolor/*.jpg"):
	# extract the image ID (i.e. the unique filename) from the image
	# path and load the image itself
	imageID = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	hist, bins = np.histogram(image.ravel(),256,[0,256])

	hist = [ float(f) for f in hist ]
	write.append({ 'imagePath': imageID, 'vector': hist })

st = json.dumps( write )
output.write( st )

# close the index file
output.close()
