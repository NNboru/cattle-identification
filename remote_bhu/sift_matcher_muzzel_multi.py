import numpy as np
import cv2
#from matplotlib import pyplot as plt
import os
from glob import glob
import pandas as pd
#from pandas import ExcelWriter
import multiprocessing
from time import time, sleep
t1 = time()

MIN_MATCH_COUNT = 4
PATH = r"E:\muzzle\remote_bhu\\"
#PATH = r"E:\muzzle\\"
DPATH = 'dataset'
'''
Dataset in folder "dataset"
masks in folder "mask"
'''
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
    print('1:',time()-t1)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(temp_img, None)
    kp2, des2 = sift.detectAndCompute(map_img, None)
    print('2:',time()-t1)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # find matches by knn which calculates point distance in 128 dim
    matches = flann.knnMatch(des1, des2, k=2)
    print('3:',time()-t1)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
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
        print('--end--')

        h, w = temp_img.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                          [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)  # matched coordinates

        map_img = cv2.polylines(
            map_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        raise Exception('Not enough matches are found')


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


def one_mask_to_all(mask_path, glob_data):
    global t1,time
    mask_name = os.path.basename(os.path.abspath(mask_path)).split('_')[-2]
    print('Starting with mask : ',mask_name,flush=True)
    #### create dataframe for xlsx file
    df = pd.DataFrame(columns=list(range(1,101)))
    
    #### to read map images from directory
    temp_img_gray = cv2.imread(mask_path,0)
    s = temp_img_gray.shape
    SIZE=512
    val = min(s[0]/SIZE,s[1]/SIZE)
    temp_img_gray = cv2.resize(temp_img_gray,(int(s[1]/val),int(s[0]/val)))
    #print('template size :',s,'=>',temp_img_gray.shape)
    # equalize histograms
    temp_img_eq = cv2.equalizeHist(temp_img_gray)

    for folder in glob_data:
        label = os.path.basename(os.path.abspath(folder))
        sides = glob(os.path.join(PATH,DPATH,label)+ r'/*')
        sides = list(map( os.path.basename, sides) )
        name = ''
        for side in sides:
            if side=='M': name='M'
            elif side.lower().startswith('muzzel'):
                name = side
            elif side.lower().startswith('muzel'):
                name = side
        if name=='':
            print('\rError in : '+folder, '"F" folder not found')
            print('Ignoring "'+folder+'" and moving ahead')
            continue
                  
        #print('\rMatching with label : ',label,end='', flush=True)
        imgs = glob(os.path.join(PATH,DPATH,label,name) + r'/*')
        imgs.sort(key=comp_name)
        count = 0
        #print('matching template - ',temp_name)
        column = []
        SIZE = 1024
        for img in imgs:
            img2 = cv2.imread(img,0)
            s = img2.shape
            val = min(s[0]/SIZE,s[1]/SIZE)
            map_img_gray = cv2.resize(img2,(int(s[1]/val),int(s[0]/val)))
            print(map_img_gray.shape)
            map_img_eq = cv2.equalizeHist(map_img_gray)
            try:
                get_matched_coordinates(temp_img_eq, map_img_eq)
                print('running', round((time()-t1)/60,2),'min',flush=True)
                print(count,'matched!! with - ',label,count+1)
                column.append(1)
            except Exception as e:
                print('gave error -',label,count+1,e)
                print('running', round((time()-t1)/60,2),'min',flush=True)
                column.append(0)
            count+=1
        column.extend([pd.NA]*(100-count))
        df.loc[label] = column

    #df.to_excel(PATH + r'results_mask_matching_muzzel/'+mask_name+r'.xlsx')


if __name__=='__main__':
    N_TASKS = 1
    t1 = time()
    if not os.path.exists(PATH + 'results_mask_matching_muzzel/'):
        os.mkdir(PATH + 'results_mask_matching_muzzel/')
    
    glob_mask = sorted(glob(PATH+r'mask/*'),key=comp_mask)
    glob_data = sorted(glob(PATH+DPATH+r'/*'),key=comp_map)
    print('Total masks =',len(glob_mask),'Total classes =',len(glob_data))

    pros_list = []
    
    cnt_done=0
    while cnt_done!=len(glob_mask):
        # remove completed process
        n=len(pros_list)
        while n>0:
            n-=1
            c=0
            if pros_list[n].is_alive()==False:
                print(pros_list[n].name, end=', ')
                del pros_list[n]
                c+=1
            if c:
                print('completed in', round((time()-t1)/60,2),'min',flush=True)

        # add more process if space
        while N_TASKS!=len(pros_list):
            if cnt_done==len(glob_mask):
                break
            mask_path = glob_mask[cnt_done]
            mask_name = os.path.basename(os.path.abspath(mask_path)).split('_')[-2]
            pro = multiprocessing.Process(target=one_mask_to_all, args=(mask_path, glob_data), name=mask_name)
            pros_list.append(pro)
            pro.start()
            cnt_done+=1

        sleep(10)
        
    for i in pros_list:
        i.join()

    print('Completed in', round((time()-t1)/60,2),'min',flush=True)




