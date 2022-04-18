'''
reads all dataset and create mean images for all classes
'''

from query_pipeline import *
import os

PATH = r"E:\muzzle\dataset\\"
#PATH = r"E:\muzzle\\"
MEAN_PATH = r'./mean_images'
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


less_imgs = []

correct = 0
total = 0
t=time()
glob_data = sorted(glob(PATH+r'/*'),key=comp_map)
for folder in glob_data:
        label = os.path.basename(os.path.abspath(folder))
        
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

        img_path = img_paths[NUM_MEAN_IMG]
        img = cv2.imread(img_path)

        result = query_pipeline(img, show=False)
        if result['match']:
            print(label,'matched with: ', result['prediction'])
            if label == result['prediction']:
                correct+=1
        else:
            print(label, 'no match found', result['error'])
        total+=1

        print('\ntotal time taken: ', round(time()-t,3))

print('classes with images less than 8: ', *less_imgs)    
print('classes skipped: ', 'C31','C - 5021')

print('accuracy : ', correct, total, correct/total*100)



        
