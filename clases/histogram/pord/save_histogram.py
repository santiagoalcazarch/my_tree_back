import glob
import cv2
import json
import numpy as np

# open the output index file for writing
output = open("histo.json", "w")

# JSON array to write
write = []

n_files = len(glob.glob("./archive/leafsnap-dataset/dataset/images/field/*/*.jpg"))
print(n_files)
current_file = 1
for imagePath in glob.glob("./archive/leafsnap-dataset/dataset/images/field/*/*.jpg"):
	# extract the image ID (i.e. the unique filename) from the image
	# path and load the image itself
	print(str(current_file / n_files * 100), end='\r', flush=True)
	imageID = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)

	hist, bins = np.histogram(image.ravel(),256,[0,256])

	hist = [ float(f) for f in hist ]
	write.append({ 'imagePath': imageID, 'vector': hist })
	current_file += 1
st = json.dumps( write )
output.write( st )

# close the index file
output.close()