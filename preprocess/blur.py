import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys


# load
image = cv2.imread('th1.jpg',0)
img=image.copy()
    
s=12
im = img.copy()
h,w=img.shape
for i in range(s,h-s,3):
    for j in range(s,w-s,3):
        cnt=0
        for k in range(1,s+1):
            cnt+=np.count_nonzero(img[i-k,j-(s-k):j+(s-k)+1]==255)
            cnt+=np.count_nonzero(img[i+k,j-(s-k):j+(s-k)+1]==255)
        cnt+= np.count_nonzero(img[i,j-s:j+s+1]==255)
        total = 2*s*(s+1)+1
        if int(.6*total)<cnt:
            for k in range(1,s+1):
                im[i-k,j-(s-k):j+(s-k)+1]=255
                im[i+k,j-(s-k):j+(s-k)+1]=255
            im[i,j-s:j+s+1]=255

  
plt.subplot(121)
plt.imshow(image,cmap='gray')
plt.subplot(122)
plt.imshow(im,cmap='gray')
plt.tight_layout()
plt.show()
sys.exit()



for i in range(41): #blur
    img = cv2.medianBlur(img,5)
    cv2.imwrite(f'blur/rep {i}.jpg',img)
