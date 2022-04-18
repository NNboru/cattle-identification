import numpy as np
import cv2
from matplotlib import pyplot as plt

img_names = ['1_mask.jpg','2_mask.jpg']

imgs=[]
for img in img_names:
    im = cv2.imread(img,0)
    im2 = cv2.resize(im,(512,512))
    imgs.append(im2)

print('creating new template - ')
avg_img = np.zeros((512,512))
for i in range(512):
    for j in range(512):
        avg_img[i][j]+=im[i][j]

avg_img//=len(imgs)

cv2.imwrite('avg_img.jpg',avg_img)
plt.imshow(avg_img)
plt.show()
    

