import numpy as np
import cv2
#from matplotlib import pyplot as plt
import os
from os.path import abspath, basename, splitext
from glob import glob
import pandas as pd

fname = "results_sift_matching_muzzle_yolo.xlsx"
def read_label(fname):
    df = pd.read_excel(fname)
    df.index = df['Unnamed: 0'].to_numpy()
    df.drop('Unnamed: 0',axis=1,inplace=True)
    return df

df = read_label(fname)
allr=[]
not_matched=[]
for th in range(10,101,10):
    TP=0
    totalp=0
    non=[]
    for ind in df.index:
        if ind in ['C31','C - 5021']:
            continue
        a = np.count_nonzero(df.loc[ind]==1)-1
        b = np.count_nonzero(df.loc[ind]==0)
        if (a*100)/(a+b) >=th:
            TP+=1
        else:
            non.append(ind)
        totalp+=1

    recall = TP/totalp
    allr.append(round(recall*100,3))
    not_matched.append(non)
print(allr)

#### image-based-recall = 1853/1896 = 97.73%
#### class-based-recall at different thresholds - 
# [100.0, 99.432, 99.432, 99.432, 98.864, 98.295, 97.159, 93.75, 92.045, 90.341]
# number of classes not matched = [0, 1, 1, 1, 2, 3, 5, 11, 14, 17] out of 176


# C31 & C-5021 are ignored. So total 176 classes
# The image from which template is croped is not considered.
# time taken per image is 2.5-3sec.
