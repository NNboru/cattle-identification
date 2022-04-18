import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt
import os
from glob import glob
import pandas as pd
from time import time
from pandas import ExcelWriter

MIN_MATCH_COUNT = 5
PATH = r'E:/muzzle/'
DPATH = PATH + 'dataset_yolo/'
EXCEL = 'results_sift_matching_muzzle_yolo'
SIZE=512
'''
Dataset in folder DPATH
masks in folder "mask"
'''
def get_matched_coordinates(temp_img, map_img, name=''):

    # initiate SIFT/SURF detector
    sift = cv2.SIFT_create()
    #sift = cv2.xfeatures2d.SURF_create(50000)

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
        if m.distance < 0.6*n.distance:
            good.append(m)

    return len(good)

def comp_map(x):
    x = os.path.basename(x)
    if x[1]==' ':
        return int(x[4:])
    else:
        return int(x[1:])
def comp_name(x):
    x = os.path.basename(x)
    l=x.split('-')
    try:
        if len(l)<4:
            return int(l[-1][-3:])
        return int(l[3])
    except:
        return 404
    return int(l[3])
def comp_mask(x):
    x=x.rstrip('_M.jpg')
    return comp_map(x)


t1 = time()
#### create dataframe for xlsx file
df = pd.DataFrame(columns=list(range(1,101)))
glob_mask = sorted(glob(PATH+r'mask/*'),key=comp_mask)
glob_data = sorted(glob(DPATH+'/*'),key=comp_map)
print('Total masks =',len(glob_mask),'Total classes =',len(glob_data))

failed = []
for mask_path in glob_mask:
    mask_name = os.path.basename(os.path.abspath(mask_path)).split('_')[-2]
    print('\n\nStarting with mask : ',mask_name)
    
    #### to read map images from directory
    temp_img_gray = cv2.imread(mask_path,0)
    s = temp_img_gray.shape
    val = min(s[0]/SIZE,s[1]/SIZE)
    temp_img_gray = cv2.resize(temp_img_gray,(int(s[1]/val),int(s[0]/val)))
    #print('template size :',s,'=>',temp_img_gray.shape)
    # equalize histograms
    temp_img_eq = cv2.equalizeHist(temp_img_gray)
    
    if 1:
        label = mask_name
        #print('\rMatching with label : ',label, flush=True)
        imgs = glob(DPATH+label + '/*')
        imgs.sort(key=comp_name)
        count = 0
        #print('matching template - ',temp_name)
        column = []
        for img in imgs:
            #print('\rMatching with img : ',img)
            map_img_gray = cv2.imread(img,0)
            s = map_img_gray.shape
            val = min(s[0]/SIZE,s[1]/SIZE)
            map_img_gray = cv2.resize(map_img_gray,(int(s[1]/val),int(s[0]/val)))
            map_img_eq = cv2.equalizeHist(map_img_gray)
            try:
                good = get_matched_coordinates(temp_img_eq, map_img_eq)
                if good >= MIN_MATCH_COUNT:
                    print(count+1,'matched! with - ',label,good)
                    column.append(1)
                else:
                    print(count+1,'fail matched! -',label,good)
                    failed.append((label,count+1))
                    column.append(0)
            except Exception as e:
                print('gave error -',label,count+1,e)
                column.append(0)
            count+=1
        column.extend([pd.NA]*(100-count))
        df.loc[label] = column

    print('\rRunning from :',round((time()-t1)/60,2),'min                ', flush=True)

df.to_excel(f'./{EXCEL}.xlsx')
    




