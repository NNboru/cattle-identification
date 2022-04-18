import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

# hyperparameters
BIAS   = 0
REP    = 4
THRESH = 180
BLUR   = 5 # should be odd value
SHAPE  = 1 # 0=contour, 1=elipse, 2=point

# load
img = cv2.imread('2_mask.jpg',0)

if 1: #clahe
    clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize=(32,32))
    for i in range(1,REP+1):
        img = clahe.apply(img) + BIAS
im1 = img.copy()

if 1: # threshold
    _,img = cv2.threshold(img,THRESH,255,cv2.THRESH_BINARY)
im2 = img.copy()


kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5,5))

img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel,iterations = 1)
im3 = img.copy()
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel,iterations = 1)
im4 = img.copy()

im7 = np.concatenate((im1,im2),axis=1)
im8 = np.concatenate((im4,im3),axis=1)
im9 = np.concatenate((im7,im8),axis=0)
plt.imshow(im9,cmap='gray')
plt.tight_layout()
plt.show()
