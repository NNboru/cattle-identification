import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

# hyperparameters
BIAS   = 8
REP    = 4
THRESH = 190
COMP   = 100
SHAPE  = 512

# load
img = cv2.imread('3_3.jpg',0)
print(img.shape)
img = cv2.resize(img,(SHAPE,SHAPE))
COMP = min(img.shape)//10

def show(img=img):
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

if 1: #morpho
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5,5))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel,iterations = 1)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel,iterations = 1)
    im3 = img.copy()


if 1: #components
    def imshow_components(labels):
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        labeled_img[label_hue==0] = 0
        return labeled_img
    
    num_labels, img = cv2.connectedComponents(img)
    print(num_labels)

    cnt = np.zeros((num_labels,))
    for i in img.ravel():
        cnt[i]+=1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if cnt[img[i,j]]<COMP:
                img[i,j]=0
    print(len(np.unique(img)))
    if 0:
        img = imshow_components(img)
    else:
        img[img>0]=255
        img = img.astype(np.uint8)
        
    im4 = img.copy()


if 0:
    imgs = [im1,im2,im3,im4,im5,im6]
    for i in range(1,7):
        plt.subplot(2,3,i)
        plt.imshow(imgs[i-1],cmap='gray')
        plt.axis(False)
    plt.tight_layout()
    plt.show()
else:
    im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)
    im3 = cv2.cvtColor(im3, cv2.COLOR_GRAY2BGR)
    if len(im4.shape)==2:
        im4 = cv2.cvtColor(im4, cv2.COLOR_GRAY2BGR)
    im7 = np.concatenate((im1,im2),axis=1)
    im8 = np.concatenate((im4,im3),axis=1)
    im9 = np.concatenate((im7,im8),axis=0)
    plt.imshow(im9,cmap='gray')
    plt.tight_layout()
    plt.show()
    #cv2.imwrite('all_compare2.jpg',im9)
    

#cv2.imwrite('comp 33.jpg',im4)

