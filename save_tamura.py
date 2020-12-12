from src.mytree_back.my_tree_api.models import Tamura
import glob
import cv2
import json


# open the output index file for writing
output = open("tamura.json", "w")

# JSON array to write
write = []
total = len(glob.glob("./archive/leafsnap-dataset/dataset/images/field/*/*.jpg"))
count = 1
for imagePath in glob.glob("./archive/leafsnap-dataset/dataset/images/field/*/*.jpg"):
    count += 1
    # extract the image ID (i.e. the unique filename) from the image
    # path and load the image itself
    imageID = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    
    # describe the image
    features = Tamura().describe(image)
    features = [float(f) for f in features]
    # write the features to file
    write.append({ 'imagePath': imagePath, 'vector': features })
    print(str(count / total * 100) + " %", end="\r", flush=True)

st = json.dumps( write )
output.write( st )

# close the index file
output.close()