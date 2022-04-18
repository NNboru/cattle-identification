'''
takes one query image and pridicts its class. (yolo+index+sift)
'''

from Object_Detection import *
from Indexing import *
from Sift_matcher import *


D_PATH = r'E:\muzzle\pipeline\muzzle_dataset'
THRESH_NUM_KEY_MATCH = 10

def query_pipeline(img, show=True):
    result ={'match':False}
    
    # localization
    t = time()
    muzzle = YOLOV3_localizer(img)
    time_yolo = round(time()-t,3)
    if show: print('yolo time =', time_yolo)
    if type(muzzle)==np.ndarray:
        if show:
            plt.imshow(muzzle)
            #plt.show()
    else:
        result['error']='no muzzel found'
        return result
        

    # indexing
    t = time()
    muzzle_bw = cv2.cvtColor(muzzle, cv2.COLOR_BGR2GRAY)
    top5 = indexing_top5(muzzle_bw)
    time_indexing = round(time()-t,3)
    if show:
        print('\nindexing time =', time_indexing)
        print('top 5 classes = ', ', '.join(top5))

    # SIFT & matching
    
    for label in top5:
        if show: print('\nSIFT matching with:',label)
        imgs = glob(os.path.join(D_PATH,label) + r'/*')
        imgs.sort(key=comp_name)
        # selecting 8 images
        imgs = imgs[:8]
    
        total_match = 0
        total = 0
        for im in imgs:
            t_img = cv2.imread(im)
            cnt = SIFT_matcher(muzzle,t_img)
            print(label,"cnt :",cnt)
            if cnt >= THRESH_NUM_KEY_MATCH:
                total_match+=1
                if show: print('matched')
            else:
                if show: print('not matched')

            if total_match >= 3:
                # matched!
                result ={'match':True, 'prediction':label}
                return result

    result['error'] = 'none matched at indexing' + ','.join(top5)
    return result     # no match found
        

if __name__ == '__main__':
    print('\nRunning pipeline ->\n')

    img_path = 'E:\muzzle\dataset\C8163\M\M-J-5-7-C-C8163.jpeg'

    t=time()
    img = cv2.imread(img_path)
    result = query_pipeline(img)
    if result['match']:
        print('matched with: ', result['prediction'])
    else:
        print('no match found', result['error'])
    print('\ntotal time taken: ', round(time()-t,3))
    



    
    
