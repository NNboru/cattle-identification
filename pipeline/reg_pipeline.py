'''
input: array of query images for registration.
output: generates mean image and saves it.
'''

from Object_Detection import *

def reg_pipeline(imgs, label, show=True):
    SHAPE = 512
    print(f'\nRunning reg-pipeline for {label} ->')
    
    ### localization
    if show: print('localizing...')
    muz_arr = []
    ind=1
    for img in imgs:
        muzzle = YOLOV3_localizer(img)
        if type(muzzle)==np.ndarray:
            muzzle = cv2.cvtColor(muzzle, cv2.COLOR_BGR2GRAY)
            muzzle = cv2.resize(muzzle,(SHAPE,SHAPE))
            muz_arr.append(muzzle)
        else:
            print('localize error: ', label, ind)
        ind+=1
    #time_yolo = round(time()-t,3)
    #if show: print('yolo time =', time_yolo)

    ### mean image generation
    if show: print('creating new template...')
    mean_img = np.zeros((SHAPE,SHAPE))
    for muz in muz_arr:
        mean_img+=muz
    mean_img= mean_img/len(muz_arr)

    return muz_arr, mean_img     # done
        

if __name__ == '__main__':
    
    folder = r'E:\muzzle\dataset\C9\M\\'

    img_paths = glob(folder + '/**')
    img_paths.sort(reverse=0)
    img_paths = img_paths[:6]
    imgs = [cv2.imread(img) for img in img_paths]
    
    #t=time()
    result = reg_pipeline(imgs, 'C9')
    if type(result)==np.ndarray:
        print('registered')
    else:
        print('not registered')

    #cv2.imwrite(mean_loc + f'/{label}_m.jpg',avg_img)
    #print('\ntotal time taken: ', round(time()-t,3))
    



    
    
