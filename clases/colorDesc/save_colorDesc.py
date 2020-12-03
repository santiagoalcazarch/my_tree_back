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

print("Reading images...")
# use glob to grab the image paths and loop over them
n_files = len(glob.glob("./archive/leafsnap-dataset/dataset/images/field/*/*.jpg"))
current_file = 1
for imagePath in glob.glob("./archive/leafsnap-dataset/dataset/images/field/*/*.jpg"):
	print(str(current_file / n_files * 100), end='\r', flush=True)
	image = cv2.imread(imagePath)
	features = cd.describe( image )
	features = [float(f) for f in features]
	write.append({ 'imagePath': imagePath, 'vector': features })
	current_file += 1
st = json.dumps( write )
output.write( st )

# close the index file
output.close()
print("\nDONE!")
