from ColorDescriptor import ColorDescriptor
import glob
import cv2
import json

# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))

# open the output index file for writing
output = open("colorDesc.json", "w")

# JSON array to write
write = []

for folder in glob.glob("../../../archive/leafsnap-dataset/dataset/images/field/*"):
	
	# use glob to grab the image paths and loop over them
	for imagePath in glob.glob(f"{folder}/*.jpg"):
		
		# extract the image ID (i.e. the unique filename) from the image
		# path and load the image itself
		imageID = imagePath[imagePath.rfind("/") + 1:]
		image = cv2.imread(imagePath)
		
		# describe the image
		features = cd.describe(image)
		# write the features to file
		
		features = cd.describe( image )
		features = [float(f) for f in features]
		write.append({ 'imagePath': imageID, 'vector': features })

st = json.dumps( write )
output.write( st )

# close the index file
output.close()