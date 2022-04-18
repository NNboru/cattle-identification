import numpy as np
import cv2
import sys
from time import time
from bisect import bisect
import matplotlib.pyplot as plt

SIZE=512
THRESH=120
TH_F=.4
WINDOW=30

img1 = '2_1.jpg'
img2 = '2_2.jpg'

def process_clahe(img):
    clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize=(32,32))
    img = clahe.apply(img)
    return img

def match1(a,b):
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
def match4(a,b,c,d,th1=160,th2=8,th3=THRESH**2):
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
def match5(a,b,c,d,win=WINDOW):
    s1=np.array([i.angle for i in c])
    s2=np.array([i.angle for i in d])
    s1=list(zip(s1,np.arange(s1.shape[0])))
    s2=list(zip(s2,np.arange(s2.shape[0])))
    s1.sort()
    s2.sort()
    s1,s1_1 = zip(*s1)
    a2=a[list(s1_1)]
    s2,s2_1 = zip(*s2)
    b2=b[list(s2_1)]
    matcher = cv2.BFMatcher()
    cnt=0
    win=360//win
    len1=len(a)//win
    len2=len(b)//win
    st1,st2=len1//2,len2//2
    matchess=[]
    for i in range(win*2-1):
        matchess = matcher.match(a2[st1*i:st1*i+len1], b2[st2*i:st2*i+len2])
        for j in matchess:
            if j.distance < THRESH:
                cnt += 1
    return cnt
def match6(a,b,c,d,win=WINDOW):
    st=win//2
    n=360//st
    bins1 = [[] for _ in range(n)]
    bins2 = [[] for _ in range(n)]
    for i in range(len(c)):
        ang = int(c[i].angle)
        bins1[ang//st].append(a[i])
        if ang//st:
            bins1[ang//st-1].append(a[i])
    for i in range(len(d)):
        ang = int(d[i].angle)
        bins2[ang//st].append(b[i])
        if ang//st:
            bins2[ang//st-1].append(b[i])
    bins1=tuple(map(np.array,bins1))
    bins2=tuple(map(np.array,bins2))
    matcher = cv2.BFMatcher()
    cnt=0
    #matchess=[]
    for i in range(n):
        if len(bins1[i]) and len(bins2[i]):
            matchess = matcher.match(bins1[i],bins2[i])
            for j in matchess:
                if j.distance < THRESH:
                    cnt += 1
    return cnt
def match7(a,b,c,d,win=WINDOW//2):
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
        bin2 = np.concatenate((bins2[i],bins2[(i-1)%n],bins2[(i+1)%n]))
        if len(bin2):
            matchess = matcher.match(bins1[i],bin2)
            for j in matchess:
                if j.distance < THRESH:
                    cnt += 1
    return cnt
def match(a,b,c,d):
    bins1 = [[] for _ in range(60)]
    bins2 = [[] for _ in range(60)]
    for i in range(len(c)):
        ang = int(c[i].angle)//6
        bins1[ang].append(a[i])
        if ang:
            bins1[ang-1].append(a[i])
    for i in range(len(d)):
        ang = int(d[i].angle)//6
        bins2[ang].append(b[i])
        if ang:
            bins2[ang-1].append(b[i])
    bins1=tuple(map(np.array,bins1))
    bins2=tuple(map(np.array,bins2))
    cnt=0
    #matchess=[]
    for i in range(60):
        if len(bins1[i]) and len(bins2[i]):
            matchess = matcher.match(bins1[i],bins2[i])
            for j in matchess:
                if j.distance < THRESH:
                    cnt += 1
    return cnt


def resize(img):
    s = img.shape
    val = min(s[0]/SIZE,s[1]/SIZE)
    img = cv2.resize(img,(int(s[1]/val),int(s[0]/val)))
    return img

def BFmatch():
    if f:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
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
        '''index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(None, None)
        matches = matcher.knnMatch(qdes,tdes, k=2)'''
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        matches = matcher.knnMatch(qdes,tdes, 2)
        cnt=0
        for m, n in matches:
                if m.distance < TH_F*n.distance:
                        cnt+=1
        return cnt


q_img = cv2.imread(img1,0)
t_img = cv2.imread(img2,0)
print(q_img.shape,t_img.shape)
if 1:
    q_img=resize(q_img)
    t_img=resize(t_img)
if 0:
    q_img=process_clahe(q_img)
    t_img=process_clahe(t_img)

print(q_img.shape,t_img.shape)

t1=time()
# Initialize the ORB detector algorithm
f=0
if f:
    orb = cv2.ORB_create()
else:
    orb = cv2.SIFT_create()

# Now detect the keypoints and compute the descriptors
qkey, qdes = orb.detectAndCompute(q_img,None)
tkey, tdes = orb.detectAndCompute(t_img,None)
t2=time()
print('sift time: ',round(t2-t1,2))

#flann
if not f:
        cnt=flannKnnMatch()
        t3=time()
        print('flann match time: ',round(t3-t2,3),'matches =', cnt)
t3=time()

#radius
cnt=Radiusmatch()
t4=time()
print('radius match time: ',round(t4-t3,3),'matches =', cnt)

#radius
cnt=BFmatch()
t5=time()
print('BF match time: ',round(t5-t4,3),'matches =', cnt)

matcher = cv2.BFMatcher()
#mymatch
for i in range(1):
    cnt = match7(qdes,tdes,qkey,tkey)
    print('my match time: ',round(time()-t5,3),'matches =', cnt)
    t5=time()



'''
def draw():
    a=cv2.drawKeypoints(q_img,qkey,q_img)
    plt.imshow(a)
    plt.show()
draw()
'''



