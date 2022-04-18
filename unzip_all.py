
import zipfile
import glob,os

PATH = './dataset_3/'
SAVE_PATH = './dataset/'

all_zips = glob.glob(PATH + '**.zip')
for folder in all_zips:
    with zipfile.ZipFile(folder, 'r') as zip_ref:
        label = os.path.basename(os.path.abspath(folder)).rstrip('.zip')
        os.mkdir(SAVE_PATH + label)
        zip_ref.extractall(SAVE_PATH + label)
        print('unzipped :',label)
