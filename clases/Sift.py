import numpy as np
import cv2 as cv

class Sift:
    
    def to_gray(self, color_img):
        gray = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
        return gray

    def sift_detect(self, gray_img):
        sift = cv.SIFT_create()
        # desc is the SIFT descriptors, they're 128-dimensional vectors that we can use for our final features
        keypoints = sift.detect( gray_img, None )
        return keypoints

    def keypointVector( self, keypoint ):
        vector = []
        #vector.append(keypoint.pt)
        vector.append(keypoint.size)
        vector.append(keypoint.angle)
        vector.append(keypoint.response)
        vector.append(keypoint.octave)
        vector.append(keypoint.class_id)
        return vector

    def keypointList( self, img ):       
        gray = self.to_gray( img )
        kps = self.sift_detect( gray )
        return [ self.keypointVector(f) for f in kps]

    def getKeypoint( self, point ):
        kp = []
        temp = cv.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5]) 
        kp.append(temp)
        return kp


if __name__ == '__main__':
    siftClass = Sift()
    siftClass.test()