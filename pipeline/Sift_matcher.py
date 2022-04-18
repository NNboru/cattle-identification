import numpy as np
import cv2
from time import time
import matplotlib.pyplot as plt

SIZE=512
THRESH_KEY_DIS=200
WINDOW=30

# Initialize the SIFT detector algorithm
sift = cv2.SIFT_create()
matcher = cv2.BFMatcher()

def BFmatch(qdes,tdes):
    matcher = cv2.BFMatcher()
    matchess = matcher.match(qdes,tdes)
    cnt=0
    for i in matchess:
        if i.distance<THRESH_KEY_DIS:
            cnt+=1
    return cnt

def ang_matcher(a,b,c,d,win=WINDOW//2):
    n=360//win
    bins1 = [[] for _ in range(n)]
    bins2 = [[] for _ in range(n)]
    for i in range(len(c)):
        ang = int(c[i].angle)//win
        bins1[ang].append(a[i])
    for i in range(len(d)):
        ang = int(d[i].angle)//win
        bins2[ang].append(b[i])
    bins1=tuple(map(np.array,bins1))
    bins2=tuple(map(np.array,bins2))
    matcher = cv2.BFMatcher()
    cnt=0
    #matchess=[]
    for i in range(n):
        tmp = []
        a=bins2[i]
        b=bins2[(i-1)%n]
        c=bins2[(i+1)%n]
        if len(a): tmp.append(a)
        if len(b): tmp.append(b)
        if len(c): tmp.append(c)
        if len(tmp):
            bin2 = np.concatenate(tmp)
            matchess = matcher.match(bins1[i],bin2)
            for j in matchess:
                if j.distance < THRESH_KEY_DIS:
                    cnt += 1
    return cnt

def resize_aspect(img):
    s = img.shape
    val = min(s[0]/SIZE,s[1]/SIZE)
    img = cv2.resize(img,(int(s[1]/val),int(s[0]/val)))
    return img

def SIFT_matcher(q_img,t_img):
    if 1:
        q_img=resize_aspect(q_img)
        t_img=resize_aspect(t_img)
    # SIFT
    qkey, qdes = sift.detectAndCompute(q_img,None)
    tkey, tdes = sift.detectAndCompute(t_img,None)

    # angMatcher
    cnt = ang_matcher(qdes,tdes,qkey,tkey)
    # BFmatch
    #cnt = BFmatch(qdes,tdes)
    return cnt


if __name__ == '__main__':
    img1 = r'E:\muzzle\pipeline\muzzle_dataset\C1\MUZ_1.jpg'
    img2 = r'E:\muzzle\pipeline\muzzle_dataset\C1\MUZ_8.jpg'
    t=time()

    #read
    q_img = cv2.imread(img1,0)
    t_img = cv2.imread(img2,0)

    cnt = SIFT_matcher(q_img,t_img)
    
    print('SIFT + match time: ',round(time()-t,3),'matches =', cnt)

    





