import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt
import os
from glob import glob
import pandas as pd
from time import time

MIN_MATCH_COUNT = 4

def get_matched_coordinates(temp_img, map_img, name=''):
    """
    Gets template and map image and returns matched coordinates in map image

    Parameters
    ----------
    temp_img: image
        image to be used as template

    map_img: image 
        image to be searched in

    Returns
    ---------
    ndarray
        an array that contains matched coordinates

    """

    # initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(temp_img, None)
    kp2, des2 = sift.detectAndCompute(map_img, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # find matches by knn which calculates point distance in 128 dim
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.4*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        #print((src_pts, dst_pts))
        # find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = temp_img.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                          [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)  # matched coordinates

        map_img = cv2.polylines(
            map_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        raise 'not found'
        print("Not enough matches are found - %d/%d" %
              (len(good), MIN_MATCH_COUNT))
        matchesMask = None
    '''
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    # draw template and map image, matches, and keypoints
    img3 = cv2.drawMatches(temp_img, kp1, map_img, kp2,
                           good, None, **draw_params)

    # if --show argument used, then show result image
    #if args.show:
    #    plt.imshow(img3, 'gray'), plt.show()

    # result image path
    cv2.imwrite('result '+name+'.png', img3)

    return dst'''


def comp_temp(x):
	if x[13]!='_':
		return int(x[12:14])
	else:
		return int(x[12])
def comp_map(x):
	if x[12]==' ':
		return int(x[15:])
	else:
		return int(x[12:])
	    
# for matching diff class images
if "I am main":
    t1 = time()
    #### create dataframe for xlsx file
    df = pd.DataFrame(columns=list(range(1,65)))
    
    #### to read map images from directory
    temp_img_gray = cv2.imread('1_mask.jpg',0)
    temp_img_gray = cv2.resize(temp_img_gray,(512,512))
    # equalize histograms
    temp_img_eq = cv2.equalizeHist(temp_img_gray)
    
    for folder in sorted(glob('../dataset/*'),key=comp_map):
        label = os.path.basename(os.path.abspath(folder))
        sides = glob('../dataset/'+label + '/*')
        sides = list(map( os.path.basename, sides) )
        name = ''
        for side in sides:
            if side=='M': name='M'
            elif side.startswith('Muzzel'):
                name = side
        if name=='':
            print('error in folder : ',folder)
            break

        imgs = glob('../dataset/'+label + '/' + name + '/*')
        imgs.sort(key=lambda x:int(x.split('-')[3]))
        count = 0
        #print('matching template - ',temp_name)
        column = []
        for img in imgs:
            map_img_gray = cv2.imread(img,0)
            map_img_eq = cv2.equalizeHist(map_img_gray)
            try:
                get_matched_coordinates(temp_img_eq, map_img_eq)
                print(count,'matched!! with - ',label,count)
                column.append(1)
            except:
                print('gave error -',label,count+1)
                column.append(0)
            count+=1
        column.extend([pd.NA]*(64-count))
        df.loc[label] = column
        df.to_excel('results_5001-mask_matching.xlsx')
        print('Running from : ',round((time()-t1)/60,1),'min')

    




