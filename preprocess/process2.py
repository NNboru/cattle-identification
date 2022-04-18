import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

# hyperparameters
BIAS   = 8
REP    = 4
THRESH = 190
COMP   = 100

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
    
def process_morpho(img):
    img = process_thresh(img)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5,5))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel,iterations = 1)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel,iterations = 1)
    return img

def process_compo(img):
    img = process_morpho(img)
    num_labels, img = cv2.connectedComponents(img)
    #print(num_labels)

    cnt = np.zeros((num_labels,))
    for i in img.ravel():
        cnt[i]+=1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if cnt[img[i,j]]<COMP:
                img[i,j]=0
    #print(len(np.unique(img)))
    img[img>0]=255
    img = img.astype(np.uint8)
    return img

process = process_compo

if __name__=='__main__':
    # load
    img = cv2.imread('3_2.jpg',0)
    im = process_compo(img)
    #cv2.imwrite('pre 3_2.jpg',im)
    plt.imshow(im,cmap='gray')
    plt.show()
