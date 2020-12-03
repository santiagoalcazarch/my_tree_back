import numpy as np
import glob
import cv2
import json

# open the output index file for writing
output = open("hist.json", "w")

write = []

print("Reading images...")
n_files = len(glob.glob("./archive/leafsnap-dataset/dataset/images/field/*/*.jpg"))
current_file = 1
# use glob to grab the image paths and loop over them
for imagePath in glob.glob("./archive/leafsnap-dataset/dataset/images/field/*/*.jpg"):
	print(str(current_file / n_files * 100), end='\r', flush=True)
	image = cv2.imread(imagePath)
	hist, bins = np.histogram(image.ravel(),256,[0,256])

	hist = [ float(f) for f in hist ]
	write.append({ 'imagePath': imagePath, 'vector': hist })
	current_file += 1
st = json.dumps( write )
output.write( st )

# close the index file
output.close()
print("\nDONE!")
