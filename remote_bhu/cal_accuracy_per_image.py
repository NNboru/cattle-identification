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
tot_img=1856        #df.count().sum()
TP=0
FP=0
TN=0
FN=0
cntp=0
cntn=0
for fpath in globe:
    fname = splitext(basename(fpath))[0]
    if fname in ['C31','C - 5021']:
        continue
    #if fname == 'C - 5076':continue #coz no front data in 5076
    print(fname, end=',')
    df = read_label(fname)
    totalp += df.loc[fname].count()-1
    totaln += tot_img - df.loc[fname].count()
    a = np.count_nonzero(df.loc[fname]==1)-1
    b = np.count_nonzero(df.loc[fname]==0)
    TP += a
    FN += b
    FP += np.count_nonzero(df==1) - a
    TN += np.count_nonzero(df==0) - b
    if a: cntp+=1
    if np.count_nonzero(df==1) > a: cntn+=1

print()
print('TP :', TP)
print('FP :', FP)
print('TN :', TN)
print('FN :', FN)

acc = (TP+TN)/(totalp+totaln)
recall = TP/totalp
print('accuracy :', acc)
print('recall :', recall)



