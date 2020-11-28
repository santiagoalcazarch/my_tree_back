from prod.ColorDescriptor import ColorDescriptor
import argparse
import glob
import cv2
import json

# from clases.Sift import Sift

# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))

# open the output index file for writing
output = open("colorDes.json", "w")

write = []

# use glob to grab the image paths and loop over them
for imagePath in glob.glob("../../archive/leafsnap-dataset/dataset/images/field/abies_concolor/*.jpg"):
	# extract the image ID (i.e. the unique filename) from the image
	# path and load the image itself
	imageID = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)

	# sift = Sift()
	# grey = sift.to_gray( imgtest )
	# kp = sift.sift_detect( grey )

	# features = sift.keypointList( image )
	features = cd.describe( image )
	features = [float(f) for f in features]
	write.append({ 'imagePath': imageID, 'vector': features })

st = json.dumps( write )
output.write( st )

# close the index file
output.close()
