'''
reads all dataset and create mean images for all classes
'''

from reg_pipeline import *
import os

PATH = r"E:\muzzle\dataset\\"
#PATH = r"E:\muzzle\\"
MEAN_PATH = r'./mean_image_dataset/'
YPATH = './muzzle_dataset/'
NUM_MEAN_IMG = 8

def comp_map(x):
    x = os.path.basename(x)
    try:
        if x[1]==' ':
            return int(x[4:])
        else:
            return int(x[1:])
    except:
        return 404

def comp_name(x):
    x = os.path.basename(x)
    l=x.split('-')
    try:
        if len(l)<4:
            return int(l[-1][-3:])
        return int(l[3])
    except:
        return 404


tmp = glob(MEAN_PATH+r'/*')
all_mean_path = [os.path.basename(i).rstrip('_m.jpg') for i in tmp]

less_imgs = []

t=time()
glob_data = sorted(glob(PATH+r'/*'),key=comp_map)
for folder in glob_data:
        label = os.path.basename(os.path.abspath(folder))
        if 'skip if its there':
            flag=0
            for i in all_mean_path:
                if label in i:
                    flag=1
            if flag:
                print(label, 'already there')
                continue
        
        new_path = PATH + label
        sides = glob(new_path + '/*')
        while len(sides)==1:
            new_path += '/' + os.path.basename(sides[0])
            sides = glob(new_path + '/*')
        sides = list(map( os.path.basename, sides) )
        name = ''
        for side in sides:
            if side.lower().startswith('m'):
                name = side
        if name=='':
            print('\rError in : '+folder, '"M" folder not found')
            print('Ignoring "'+folder+'" and moving ahead')
            continue

        img_paths = glob(os.path.join(new_path,name) + r'/*')
        if len(img_paths) < 9 or (label in ['C31','C - 5021']):
            less_imgs.append(label)
            continue
        img_paths.sort(key=comp_name)

        img_paths = img_paths[:NUM_MEAN_IMG]
        imgs = [cv2.imread(img) for img in img_paths]

        muz_arr, mean_img = reg_pipeline(imgs, label , show=False)

        # saving croped muzzle's
        LAB_PATH = os.path.join(YPATH,label)
        os.mkdir(LAB_PATH)
        ind=1
        for muz in muz_arr:
                IMG_PATH = os.path.join(LAB_PATH,'MUZ_'+str(ind)+'.jpg')
                cv2.imwrite(IMG_PATH,muz)
                ind+=1

        # saving mean-image
        cv2.imwrite(MEAN_PATH + f'/{label}_m.jpg',mean_img)
        
        print('\ntotal time taken: ', round(time()-t,3))

print('classes with images less than 8: ', *less_imgs)    
print('classes skipped: ', 'C31','C - 5021')



        
