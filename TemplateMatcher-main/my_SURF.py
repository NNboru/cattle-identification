import numpy as np
import cv2
import sys
from time import time
from bisect import bisect

SIZE=512
THRESH=0.12
TH_F=.5

def match(a,b):
	cnt=0
	s1=a.sum(axis=-1).astype(int)
	s2=b.sum(axis=-1).astype(int)
	s2=list(zip(s2,np.arange(s2.shape[0])))
	s2.sort()
	s2,s2_1 = zip(*s2)
	b2=b[list(s2_1)]
	span=160
	for i,val in enumerate(s1):
		l,r=bisect(s2,val-span),bisect(s2,val+span)
		if r>l and ((b2[l:r]-a[i])**2).sum(axis=-1).min()<14400:
			cnt+=1
	return cnt
def match2(a,b):
	cnt=0
	s1=a.sum(axis=-1).astype(int)
	s2=b.sum(axis=-1).astype(int)
	for i,val in enumerate(s1):
		fil = b[np.absolute(s2-val)<160]
		if ((fil-a[i])**2).sum(axis=-1).min()<14400:
			cnt+=1
	return cnt
def match3(a,b,c,d,th1=160,th2=8):
	cnt=0
	s1=a.sum(axis=-1).astype(int)
	s2=b.sum(axis=-1).astype(int)
	s3=np.array([i.angle for i in c])
	s4=np.array([i.angle for i in d])
	for i,val in enumerate(s1):
		tmp=np.absolute(s4-s3[i])%(360-th2)
		fil = b[(np.absolute(s2-val)<th1) & (tmp<th2)]
		if len(fil) and ((fil-a[i])).sum(axis=-1).min()<14400:
			cnt+=1
	return cnt
def match4(a,b,c,d,th1=160,th2=8,th3=120**2):
	cnt=0
	s1=a.sum(axis=-1).astype(int)
	s2=b.sum(axis=-1).astype(int)
	s3=np.array([i.angle for i in c])
	s4=np.array([i.angle for i in d])
	s2=list(zip(s2,np.arange(s2.shape[0])))
	s2.sort()
	s2,s2_1 = zip(*s2)
	b2=b[list(s2_1)]
	s4=s4[list(s2_1)]
	for i,val in enumerate(s1):
		l,r=bisect(s2,val-th1),bisect(s2,val+th1)
		tmp1=np.absolute(s4[l:r]-s3[i])%(360-th2)
		tmp=b2[l:r][tmp1<th2]
		if len(tmp) and ((tmp-a[i])**2).sum(axis=-1).min()<th3:
			cnt+=1
	return cnt

def resize(img):
    SIZE=512
    s = img.shape
    val = min(s[0]/SIZE,s[1]/SIZE)
    img = cv2.resize(img,(int(s[1]/val),int(s[0]/val)))
    return img

def BFmatch():
    if f:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        matcher = cv2.BFMatcher()
    matchess = matcher.match(qdes,tdes)
    cnt=0
    for i in matchess:
        if i.distance<THRESH:
            cnt+=1
    return cnt

def Radiusmatch():
    if f:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        matcher = cv2.BFMatcher()
    matchess = matcher.radiusMatch(qdes,tdes,maxDistance=THRESH)
    cnt=0
    for i in matchess:
        cnt+=len(i)
    return cnt

def flannKnnMatch():
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(qdes,tdes, k=2)
        cnt=0
        for m, n in matches:
                if m.distance < TH_F*n.distance:
                        cnt+=1
        return cnt


q_img = cv2.imread('1_1.jpg',0)
t_img = cv2.imread('1_2.jpg',0)
print(q_img.shape,t_img.shape)
if 1:
    q_img=resize(q_img)
    t_img=resize(t_img)
print(q_img.shape,t_img.shape)

t1=time()
# Initialize the ORB detector algorithm
f=0
if f:
    orb = cv2.ORB_create()
else:
    orb = cv2.xfeatures2d.SURF_create(100,upright=False)

# Now detect the keypoints and compute the descriptors
qkey, qdes = orb.detectAndCompute(q_img,None)
tkey, tdes = orb.detectAndCompute(t_img,None)
t2=time()
print('surf time: ',round(t2-t1,2))

#flann
if not f:
        cnt=flannKnnMatch()
        t3=time()
        print('flann match time: ',round(t3-t2,2),'matches =', cnt)
t3=time()

#radius
cnt=Radiusmatch()
t4=time()
print('radius match time: ',round(t4-t3,2),'matches =', cnt)

#radius
cnt=BFmatch()
t5=time()
print('BF match time: ',round(t5-t4,2),'matches =', cnt)

#mymatch
cnt = match4(qdes,tdes,qkey,tkey,th3=THRESH**2)
t6=time()
print('my match time: ',round(t6-t5,2),'matches =', cnt)









