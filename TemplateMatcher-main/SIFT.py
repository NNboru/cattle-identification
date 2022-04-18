import numpy as np
import cv2
import sys
from time import time

def resize(img):
    SIZE=512
    s = img.shape
    val = min(s[0]/SIZE,s[1]/SIZE)
    img = cv2.resize(img,(int(s[1]/val),int(s[0]/val)))
    return img

q_img = cv2.imread('1_1.jpg',0)
t_img = cv2.imread('1_2.jpg',0)
if 1:
    q_img=resize(q_img)
    t_img=resize(t_img)
print(q_img.shape,t_img.shape)

t1=time()
# Initialize the ORB detector algorithm
orb = cv2.ORB_create()
orb = cv2.SIFT_create()

# Now detect the keypoints and compute the descriptors
qkey, qdes = orb.detectAndCompute(q_img,None)
tkey, tdes = orb.detectAndCompute(t_img,None)
t2=time()
print('sift time: ',round(t2-t1,2))

# Initialize the Matcher for matching
matcher = cv2.BFMatcher()
if 0:
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

# Do the matching
if 0: #radiusMatch
    matchess = matcher.radiusMatch(qdes,tdes,maxDistance=120)
    matches=[]
    for i in matchess:
        matches.extend(i)
else: #match
    matchess = matcher.match(qdes,tdes)
    matches=[]
    for i in matchess:
        if i.distance<140:
            matches.append(i)

t3=time()
print('match time: ',round(t3-t2,2), len(matches))
# draw the matches to the final image
final_img = cv2.drawMatches(q_img, qkey,
t_img, tkey, matches,None)

final_img = cv2.resize(final_img, (1000,650))

# Show the final image
cv2.imshow("Matches", final_img)
cv2.waitKey(5000)
