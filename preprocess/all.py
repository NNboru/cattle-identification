import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

# hyperparameters
BIAS   = 8
REP    = 3
THRESH = 200
BLUR   = 3 # should be odd value
SHAPE  = 0 # 0=contour, 1=elipse, 2=point

# load
img = cv2.imread('2_mask.jpg',0)

def show():
    plt.imshow(img,cmap='gray')
    plt.show()


if 1: #clahe
    clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize=(32,32))
    for i in range(1,REP+1):
        img = clahe.apply(img) + BIAS
im1 = img.copy()

if 1: # threshold
    _,img = cv2.threshold(img,THRESH,255,cv2.THRESH_BINARY)
im2 = img.copy()

if 1: #blur
    img = cv2.medianBlur(img,BLUR)
im3 = img.copy()

if 1: # Canny
    img = cv2.Canny(img, 30, 200)
im4 = img.copy()

# Contours
contours, hierarchy = cv2.findContours(img,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
im5 = np.zeros(img.shape)
cv2.drawContours(im5, contours, -1, 255, -1)

if 1: # filter by size
    print("Number of Contours found = " + str(len(contours)))
    carr = []
    for i,c in enumerate(contours):
        a = cv2.minAreaRect(c)
        area = a[1][0]*a[1][1]*1
        if 200>len(c)>20 and area>20:
            c = np.concatenate((c[-1:],c))
            carr.append(c)
    contours = carr

# Draw contours
img = np.zeros(img.shape)
if SHAPE==0:
    cv2.drawContours(img, contours, -1, 255, -1)
elif SHAPE==1:
    for c in contours:
        a = cv2.fitEllipse(c)
        cv2.ellipse(img, a, 255, -1)
im6 = img.copy()

if 0:
    imgs = [im1,im2,im3,im4,im5,im6]
    for i in range(1,7):
        plt.subplot(2,3,i)
        plt.imshow(imgs[i-1],cmap='gray')
        plt.axis(False)
    plt.tight_layout()
    plt.show()
else:
    im7 = np.concatenate((im1,im2,im3),axis=1)
    im8 = np.concatenate((im6,im5,im4),axis=1)
    im9 = np.concatenate((im7,im8),axis=0)
    plt.imshow(im9,cmap='gray')
    plt.tight_layout()
    plt.show()
    cv2.imwrite('all_compare.jpg',im9)
    

#cv2.imwrite('pre 3_3.jpg',im)

