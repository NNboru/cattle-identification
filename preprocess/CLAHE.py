import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# Reading the image from the present directory
image = cv2.imread("2_mask.jpg")
# Resizing
#image = cv2.resize(image, (500, 600))

# The initial processing of the image
# image = cv2.medianBlur(image, 3)
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Showing
#cv2.imshow("CLAHE image", final_img)
#plt.imshow(final_img,cmap='gray')
#plt.show()

'''
# diff clipLimit
val = 1
for i in range(10):
    val*=2
    clahe = cv2.createCLAHE(clipLimit = val)
    final_img = clahe.apply(image_bw) + 10
    cv2.imwrite('pre '+str(val)+'.jpg',final_img)
'''

img = image_bw.copy()
clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize=(32,32))
for i in range(1,16):
    img = clahe.apply(img) + 10
    #_,im = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
    cv2.imwrite('rep2 '+str(i)+'.jpg',img)
