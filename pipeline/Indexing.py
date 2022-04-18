import numpy as np
import cv2
from matplotlib import pyplot as plt
from glob import glob
import os
from time import time

INDEX_SHAPE=512
M_PATH = "E:/muzzle/pipeline/mean_image_dataset/"

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
def comp_map(x):
    x = os.path.basename(x)
    if x[1]==' ':
        return int(x[4:])
    else:
        return int(x[1:])
def comp_mask(x):
    x=x.rstrip('_m.jpg')
    return comp_map(x)

    
imgs = sorted(glob(M_PATH + '**'),key=comp_mask)
index_cols = [os.path.basename(os.path.abspath(f))[:-4] for f in imgs]

# reading all mean images
t1=time()
imgs = [cv2.imread(p,0) for p in imgs]
imgs = [cv2.resize(p,(INDEX_SHAPE,INDEX_SHAPE)).ravel() for p in imgs]
mean_imgs = np.array(imgs)
t2=time()
print('mean-images reading time:',t2-t1)

# calculating distance from 1 test image
def indexing_top5(img,num=5):
    def cal_top5(row):
        inf=row.max()
        l=np.zeros((num,),dtype=int)
        for i in range(num):
            l[i]=row.argmin()
            row[l[i]]=inf
        l = [index_cols[i].rstrip('_m') for i in l]
        return l
    img = cv2.resize(img,(INDEX_SHAPE,INDEX_SHAPE)).ravel()
    dis_arr = np.sqrt(((mean_imgs-img)**2).sum(axis=-1)).astype(int)
    return cal_top5(dis_arr)

if __name__ == '__main__':
    img_path = 'E:\muzzle\dataset_yolo\C7/M-HF-4-7-C-C7.jpg'
    t = time()
    img = cv2.imread(img_path,0)
    top5 = indexing_top5(img)
    print(top5)
    print('indexing time : ',round(time()-t,3))


