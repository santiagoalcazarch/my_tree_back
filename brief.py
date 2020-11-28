import numpy as np
import cv2


# ESTA MIERDA NO FUNCIONA JUEPUTA

try:
    
    img = cv2.imread('test.jpg',0)

    # Initiate STAR detector
    orb = cv2.ORB()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    print("hola")

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    print(kp)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)

    cv2.imshow(img2)

except expression as identifier:
    print(identifier)


print(brief.getInt('bytes'))
print(des.shape)