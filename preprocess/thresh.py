import cv2
cv=cv2
import numpy as np
from matplotlib import pyplot as plt

img_bw = cv.imread('1_mask.jpg',0)

if 1: #clahe
    img = img_bw.copy()
    clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize=(12,12))
    for i in range(1,4):
        img = clahe.apply(img) + 2
        ret,th = cv.threshold(img,200,255,cv.THRESH_BINARY)
        #cv2.imwrite('th16 '+str(i)+'.jpg',th)
    print('clahe done')
if 0: #blur
    img = cv.medianBlur(img,2)

ret,th1 = cv.threshold(img,200,255,cv.THRESH_BINARY)

plt.imshow(th1,cmap='gray')
plt.show()

