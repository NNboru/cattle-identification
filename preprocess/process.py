import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

# hyperparameters
BIAS   = 8
REP    = 3
THRESH = 200
BLUR   = 3 # should be odd value
SHAPE  = 1 # 0=contour, 1=elipse, 2=point

def show():
    plt.imshow(img,cmap='gray')
    plt.show()

def process_clahe(img):
    if 1: #clahe
        clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize=(32,32))
        for i in range(REP):
            img = clahe.apply(img) + BIAS
    return img

def process_thresh(img):
    img = process_clahe(img)
    _,img = cv2.threshold(img,THRESH,255,cv2.THRESH_BINARY)
    return img
    
def process_blur(img):
    img = process_thresh(img)
    img = cv2.medianBlur(img, BLUR)
    return img

def process_canny(img):
    img = process_blur(img)
    img = cv2.Canny(img, 30, 200)
    return img
    
def process_con(img):
    img = process_canny(img)
    contours, hierarchy = cv2.findContours(img,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if 1: # filter by size
        print("Number of Contours found = " + str(len(contours)))
        carr = []
        for i,c in enumerate(contours):
            a = cv2.minAreaRect(c)
            area = a[1][0]*a[1][1]*1000
            if 200>len(c)>20 and area>100:
                c = np.concatenate((c[-1:],c))
                carr.append(c)
        contours = carr

    # Draw contours
    im = np.zeros(img.shape)
    if SHAPE==0:
        cv2.drawContours(im, contours, -1, 255, -1)
    else:
        for c in contours:
            a = cv2.fitEllipse(c)
            cv2.ellipse(im, a, 255, -1)
    return img

process = process_blur

if __name__=='__main__':
    # load
    img = cv2.imread('3_2.jpg',0)
    im = process_blur(img)
    #cv2.imwrite('pre 3_2.jpg',im)
    plt.imshow(im,cmap='gray')
    plt.show()
