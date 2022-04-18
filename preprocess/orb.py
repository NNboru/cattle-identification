import numpy as np
import cv2
import sys
from process import *

query_img = cv2.imread('2_mask.jpg')
train_img = cv2.imread('3_1.JPG')
# Convert it to grayscale
query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

if 1:
    query_img_bw = process_blur(query_img_bw)
    train_img_bw = process_blur(train_img_bw)

if 0:
    query_img_bw = 255 - query_img_bw
    train_img_bw = 255 - train_img_bw

    
# Initialize the ORB detector algorithm
orb = cv2.ORB_create()
orb = cv2.SIFT_create()

# Now detect the keypoints and compute the descriptors
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)

# Initialize the Matcher for matching
if 1:
    matcher = cv2.BFMatcher()
else:
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

# Do the matching
if 0: #radiusMatch
    matchess = matcher.radiusMatch(queryDescriptors,trainDescriptors,maxDistance=80)
    matches=[]
    for i in matchess:
        matches.extend(i)
else: #match
    matchess = matcher.match(queryDescriptors,trainDescriptors)
    matches=[]
    for i in matchess:
        if i.distance<100:
            matches.append(i)

# draw the matches to the final image
final_img = cv2.drawMatches(query_img, queryKeypoints,
train_img, trainKeypoints, matches,None)

final_img = cv2.resize(final_img, (1000,650))

# Show the final image
cv2.imshow("Matches", final_img)
cv2.waitKey(5000)
