import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

# Let's load a simple image with 3 black squares
image = cv2.imread('th2.jpg')
cv2.waitKey(0)

# Grayscale
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if 1: #blur
    img = cv2.medianBlur(img,11)

if 1: # Find Canny edges
    img = cv2.Canny(img, 30, 200)
    #plt.imshow(img,cmap='gray')
    #plt.show()

# Finding Contours
contours, hierarchy = cv2.findContours(img,
	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#cv2.imshow('Canny Edges After Contouring', img)
#cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))

carr = []
for i,c in enumerate(contours):
    if 200>len(c)>20:
        carr.append(c)

im = np.zeros(image.shape)
# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(im, carr, -1, (255, 255, 255), -1)
print(1)
plt.imshow(im,cmap='gray')
plt.show()
