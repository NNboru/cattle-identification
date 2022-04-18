from glob import glob
import cv2
import os
import shutil
import zipfile
'''
with zipfile.ZipFile(PATH+'Images_inpaint_resized.zip', 'r') as zip_ref:
    zip_ref.extractall(PATH)
'''
#if os.path.exists('dataset_muzzle/'):
#    raise "dataset folder exists"

#os.mkdir('dataset_muzzle/')
PATH = 'dataset/'

for folder in glob(PATH + '/**/'):
    label = os.path.basename(os.path.abspath(folder))
    new_path = PATH + label
    sides = glob(new_path + '/*')
    while len(sides)==1:
        new_path += '/' + os.path.basename(sides[0])
        sides = glob(new_path + '/*')
    sides = list(map( os.path.basename, sides) )
    name = ''

    # for Left
    '''
    for side in sides:
        if side=='L': name='L'
        elif side.lower().startswith('left face'):
            name = side
        elif side.lower().startswith('left ffce'):
            name = side
        elif side.lower().startswith('left fcae'):
            name = side
    '''
    # for full Left
    '''
    for side in sides:
        if side=='FL': name='FL'
        elif side.lower().startswith('l-full'):
            name = side
        elif side.lower().startswith('f-full'):
            name = side
        elif side.lower().startswith('left full'):
            name = side
        elif side.lower().startswith('left said full'):
            name = side
    '''
    # for full Right
    '''
    if label in ['C - 5076', 'C - 6008']: # no full right data in 5076
        continue
    for side in sides:
        if side=='FR': name='FR'
        elif side.lower().startswith('right said full'):
            name = side
        elif side.lower().startswith('r-full'):
            name = side
        elif side.lower().startswith('r -full'):
            name = side
        elif side.lower().startswith('right full'):
            name = side
        elif side.lower().startswith('right  full'):
            name = side
    '''
    # for right
    '''
    if label=='C - 5076': # no right data in 5076
        continue
    for side in sides:
        if side=='R': name='R'
        elif side.lower().startswith('right face'):
            name = side
        elif side.lower().startswith('right fcae'):
            name = side
        elif side.lower().startswith('r-face'):
            name = side
    '''
    # for muzzle
    
    for side in sides:
        if side.lower().startswith('m'):
            name = side
    '''
        elif side.lower().startswith('muzzel'):
            name = side
        elif side.lower().startswith('muzel'):
            name = side
        elif side.lower().startswith('muzill'):
            name = side
        elif side.lower().startswith('muzzile'):
            name = side
        elif side.lower().startswith('muzzle'):
            name = side
        elif side.lower().startswith('muzzil'):
            name = side
        elif side.lower().startswith('mouth'):
            name = side
    '''

    # for front
    '''
    if label=='C - 5076': # no front data in 5076
        continue
    for side in sides:
        if side=='F': name='F'
        elif side.lower().startswith('fornt'):
            name = side
        elif side.lower().startswith('front'):
            name = side
        elif side.lower().startswith('forent'):
            name = side
    '''
    if name=='':
        print('error in folder : ',folder)
        break
'''
    os.mkdir('dataset_muzzle/'+label)
    images = glob('dataset/'+label + '/' + name + '/*')
    cnt=1
    for im in images:
        im = cv2.imread(im)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (size,size), interpolation=cv2.INTER_AREA)
        cv2.imwrite('dataset_muzzle/'+label+'/'+str(cnt)+'.jpg', im)
        cnt+=1
'''     

    
