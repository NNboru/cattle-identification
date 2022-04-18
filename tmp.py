from glob import glob
import cv2
import os
import shutil
import zipfile

PATH = 'dataset_2/'

cnt=[]
for folder in glob(PATH + '/**/'):
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
    if name=='':
        print('error in folder : ',folder)
        break
    
    m= len(glob(new_path + '/'+name+'/**'))
    cnt.append(m)
    if m<10:
        print(m, new_path)

'''
cnty=[]
path = 'dataset_yolo'
path = r'E:\muzzle\agronomy\300-Cattle-source/'
for folder in sorted(glob(path+'/*')):
    label = os.path.basename(os.path.abspath(folder))
    m= len(glob(path+label + '/**'))
    cnty.append(m)
'''
