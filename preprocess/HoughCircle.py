import numpy as np
import cv2
import matplotlib.pyplot as plt


img_bw = cv2.imread('1_mask.jpg',0)

img = img_bw.copy()
clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize=(8,8))
for i in range(1,6):
    img = clahe.apply(img) + 20

print('clahe done')
img = 255-img
#img = cv.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,30,
                            param1=300,param2=25,minRadius=1,maxRadius=40)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    tmp=cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    tmp=cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
