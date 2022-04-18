import numpy as np
import cv2
#from matplotlib import pyplot as plt
import os
from os.path import abspath, basename, splitext
from glob import glob
import pandas as pd

PATH = r"./results_sift_matching_muzzel_remote/"

def read_label(fname):
    df = pd.read_excel(PATH + fname + '.xlsx')
    df.index = df['Unnamed: 0'].to_numpy()
    df.drop('Unnamed: 0',axis=1,inplace=True)
    return df



globe = glob(PATH + '/**')
globe.sort(key=lambda x:(splitext(basename(x))[0]))
l=[]
totalp=0
totaln=0
TP=0
FP=0
TN=0
FN=0
cntp=[]
cntn=[]
all_th=[]
THRESH = .2
for fpath in globe:
    fname = splitext(basename(fpath))[0]
    if fname in ['C31','C - 5021']:
        continue
    #if fname == 'C - 5076':continue
    print(fname, end=',')
    df = read_label(fname)
    totalp += 1
    totaln += len(df)-1
    a = np.count_nonzero(df.loc[fname]==1)-1
    b = np.count_nonzero(df.loc[fname]==0)
    all_th.append(a/(a+b))
    if a/(a+b) >=THRESH:
        a,b=1,0
        TP+=1
    else:
        a,b=0,1
        FN+=1
    acc = np.count_nonzero(df==1,axis=1)/df.count(axis=1)                
    FP += np.count_nonzero(acc>=THRESH) - a
    TN += np.count_nonzero(acc <THRESH) - b
    if b: cntp.append(fname)
    if np.count_nonzero(acc>=THRESH) - a > 0: cntn.append(fname)

totalp = len(globe)
print()
print('TP :', TP)
print('FP :', FP)
print('TN :', TN)
print('FN :', FN)

acc = (TP+TN)/(totalp+totaln)
recall = TP/totalp
print('accuracy :', acc)
print('recall :', recall)

#### image-based-recall = 95.586%
#### class-based-recall at different thresholds - 
# [99.432, 99.432, 99.432, 98.864, 97.159, 94.318, 93.75, 90.341, 89.205, 88.068]
# number of classes not matched = [1, 1, 1, 2, 5, 10, 11, 17, 19, 21] out of 176




