import json
import cv2
import numpy as np
from math import *
import imutils

class Histogram:
    def histogram_intersection(self, hist_1, hist_2):
        minima = np.minimum(hist_1, hist_2)
        intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
        return intersection

    def run(self):
        with open('hist.json') as f:
            data = json.load(f)

        clientImage = cv2.imread("image.jpg")
        hist, bins = np.histogram(clientImage.ravel(),256,[0,256])
        hist = [ float(f) for f in hist ]

        compare_hist = []

        for carga in data:
            hist_int = self.histogram_intersection( hist, carga['vector'] )
            compare_hist.append( { 'value': hist_int, 'imagePath': carga['imagePath'] } )

        newlist = sorted(compare_hist, key=lambda k: k['value']) 
        value = newlist[len(newlist) - 1]["value"]
        spplited = newlist[len(newlist) - 1]["imagePath"].split("/")
        kind = spplited[len(spplited) - 2]
        return (value, kind)

class ColorDescriptor:
    def __init__(self, bins):
        # store the number of bins for the 3D histogram
        self.bins = bins

    def euclidean_distance(self, x, y):
        return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

    def normalize(self, v):
        norm=np.linalg.norm(v, ord=1)
        if norm==0:
            norm=np.finfo(v.dtype).eps
        return v/norm

    def run(self):
        with open('colorDes.json') as f:
            data = json.load(f)

        clientImage = cv2.imread("image.jpg")

        features = self.describe( clientImage )
        features = [float(f) for f in features]

        best = None
        best_data = None
        for carga in data:
            distance = self.euclidean_distance(self.normalize(features), self.normalize(carga['vector']))
            if best is None or distance < best:
                best = distance
                best_data = carga
        value = 1-best
        spplited = best_data["imagePath"].split("/")
        kind = spplited[len(spplited) - 2]
        return (value,kind)

    def describe(self, image):
        # convert the image to the HSV color space and initialize
        # the features used to quantify the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        segments = [(0, cX, 0, cY), (cX, w, 0, cY),
                     (cX, w, cY, h), (0, cX, cY, h)]
        # construct an elliptical mask representing the center of the
        # image
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # construct a mask for each corner of the image, subtracting
            # the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)
            # extract a color histogram from the image, then update the
            # feature vector
            hist = self.histogram(image, cornerMask)
            features.extend(hist)
        # extract a color histogram from the elliptical region and
        # update the feature vector
        hist = self.histogram(image, ellipMask)
        features.extend(hist)
        # return the feature vector
        return features

    def histogram(self, image, mask):
        # extract a 3D color histogram from the masked region of the
        # image, using the supplied number of bins per channel
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
        # normalize the histogram if we are using OpenCV 2.4
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()
        # otherwise handle for OpenCV 3+
        else:
            hist = cv2.normalize(hist, hist).flatten()
        # return the histogram
        return hist

class Tamura:
    def coarseness(self, image, kmax):
        image = np.array(image, dtype = 'int64')
        w = image.shape[0]
        h = image.shape[1]
        kmax = kmax if (np.power(2,kmax) < w) else int(np.log(w) / np.log(2))
        kmax = kmax if (np.power(2,kmax) < h) else int(np.log(h) / np.log(2))
        average_gray = np.zeros([kmax,w,h])
        horizon = np.zeros([kmax,w,h])
        vertical = np.zeros([kmax,w,h])
        Sbest = np.zeros([w,h])

        for k in range(kmax):
            window = np.power(2,k)
            for wi in range(w)[window:(w-window)]:
                for hi in range(h)[window:(h-window)]:
                    average_gray[k][wi][hi] = np.sum(image[wi-window:wi+window, hi-window:hi+window])
            for wi in range(w)[window:(w-window-1)]:
                for hi in range(h)[window:(h-window-1)]:
                    horizon[k][wi][hi] = average_gray[k][wi+window][hi] - average_gray[k][wi-window][hi]
                    vertical[k][wi][hi] = average_gray[k][wi][hi+window] - average_gray[k][wi][hi-window]
            horizon[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))
            vertical[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))

        for wi in range(w):
            for hi in range(h):
                h_max = np.max(horizon[:,wi,hi])
                h_max_index = np.argmax(horizon[:,wi,hi])
                v_max = np.max(vertical[:,wi,hi])
                v_max_index = np.argmax(vertical[:,wi,hi])
                index = h_max_index if (h_max > v_max) else v_max_index
                Sbest[wi][hi] = np.power(2,index)

        fcrs = np.mean(Sbest)
        return fcrs


    def contrast(self, image):
        image = np.array(image)
        image = np.reshape(image, (1, image.shape[0]*image.shape[1]))
        m4 = np.mean(np.power(image - np.mean(image),4))
        v = np.var(image)
        std = np.power(v, 0.5)
        alfa4 = m4 / np.power(v,2)
        fcon = std / np.power(alfa4, 0.25)
        return fcon

    def directionality(self, image):
        image = np.array(image, dtype = 'int64')
        h = image.shape[0]
        w = image.shape[1]
        convH = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        convV = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        deltaH = np.zeros([h,w])
        deltaV = np.zeros([h,w])
        theta = np.zeros([h,w])

        # calc for deltaH
        for hi in range(h)[1:h-1]:
            for wi in range(w)[1:w-1]:
                deltaH[hi][wi] = np.sum(np.multiply(image[hi-1:hi+2, wi-1:wi+2], convH))
        for wi in range(w)[1:w-1]:
            deltaH[0][wi] = image[0][wi+1] - image[0][wi]
            deltaH[h-1][wi] = image[h-1][wi+1] - image[h-1][wi]
        for hi in range(h):
            deltaH[hi][0] = image[hi][1] - image[hi][0]
            deltaH[hi][w-1] = image[hi][w-1] - image[hi][w-2]

        # calc for deltaV
        for hi in range(h)[1:h-1]:
            for wi in range(w)[1:w-1]:
                deltaV[hi][wi] = np.sum(np.multiply(image[hi-1:hi+2, wi-1:wi+2], convV))
        for wi in range(w):
            deltaV[0][wi] = image[1][wi] - image[0][wi]
            deltaV[h-1][wi] = image[h-1][wi] - image[h-2][wi]
        for hi in range(h)[1:h-1]:
            deltaV[hi][0] = image[hi+1][0] - image[hi][0]
            deltaV[hi][w-1] = image[hi+1][w-1] - image[hi][w-1]

        deltaG = (np.absolute(deltaH) + np.absolute(deltaV)) / 2.0
        deltaG_vec = np.reshape(deltaG, (deltaG.shape[0] * deltaG.shape[1]))

        # calc the theta
        for hi in range(h):
            for wi in range(w):
                if (deltaH[hi][wi] == 0 and deltaV[hi][wi] == 0):
                    theta[hi][wi] = 0;
                elif(deltaH[hi][wi] == 0):
                    theta[hi][wi] = np.pi
                else:
                    theta[hi][wi] = np.arctan(deltaV[hi][wi] / deltaH[hi][wi]) + np.pi / 2.0
        theta_vec = np.reshape(theta, (theta.shape[0] * theta.shape[1]))

        n = 16
        t = 12
        cnt = 0
        hd = np.zeros(n)
        dlen = deltaG_vec.shape[0]
        for ni in range(n):
            for k in range(dlen):
                if((deltaG_vec[k] >= t) and (theta_vec[k] >= (2*ni-1) * np.pi / (2 * n)) and (theta_vec[k] < (2*ni+1) * np.pi / (2 * n))):
                    hd[ni] += 1
        hd = hd / np.mean(hd)
        hd_max_index = np.argmax(hd)
        fdir = 0
        for ni in range(n):
            fdir += np.power((ni - hd_max_index), 2) * hd[ni]
        return fdir

    def normalize(self, v):
        norm=np.linalg.norm(v, ord=1)
        if norm==0:
            norm=np.finfo(v.dtype).eps
        return v/norm

    def euclidean_distance(self, x, y):
        return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

    def describe(self, img, resize=True):
        if resize is True:
            scale_percent = 1 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        else:
            resized = img
        coar = self.coarseness(resized, 3)
        return self.normalize([self.directionality(resized), coar])

    def run(self):
        with open('tamura.json') as f:
            data = json.load(f)

        clientImage = cv2.imread("image.jpg")

        features = self.describe(clientImage)
        features = [float(f) for f in features]

        best = None
        best_data = None
        for carga in data:
            distance = self.euclidean_distance(self.normalize(features), self.normalize(carga['vector']))
            if best is None or distance < best:
                best = distance
                best_data = carga
        value = 1-best
        spplited = best_data["imagePath"].split("/")
        kind = spplited[len(spplited) - 2]

        return (value, kind)