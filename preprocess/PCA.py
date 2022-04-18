import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys


SHAPE  = 512

# load
img1 = cv2.imread('comp 31.jpg',0)
img1 = cv2.resize(img1,(SHAPE,SHAPE))
img2 = cv2.imread('comp 32.jpg',0)
img2 = cv2.resize(img2,(SHAPE,SHAPE))
img3 = cv2.imread('comp 33.jpg',0)
img3 = cv2.resize(img3,(SHAPE,SHAPE))
imgs=[img1,img2,img3]

mean, eigvec = cv2.PCACompute(imgs, mean=None)



