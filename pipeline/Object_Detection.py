import cv2
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import sys
from time import time

t=time()
net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing_remove1.cfg')
print('yolo-model loading time =',round(time()-t,3))

def pool(img):
    size=max(img.shape)
    X = np.zeros((size,size,3),dtype=np.uint8)
    s = img.shape
    z = abs(s[0]-s[1])
    #x = np.random.randint(0,z+1)
    x=z//2
    if s[0]<s[1]:
        X[x:x+s[0]] = img
    else:
        X[:,x:x+s[1]] = img
    return X

def YOLOV3_localizer(img):
    SIZE=416
    s = img.shape[:2]
    W = (max(s)*SIZE)//min(s)
    W+= 32-W%32
    img = pool(img)    # pooling failed in some cases like C5024 img: 4,5
    height, width, _ = img.shape
    img2 = cv2.resize(img,(W,W))
    #print(img2.shape)
    blob = cv2.dnn.blobFromImage(img2, 1/255, swapRB=True, crop=False)
    
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            #confidence = str(round(confidences[i],2))
            crop = img[y:y+h, x:x+w]
            return crop
            
            
    return False  # not found
    

if __name__ == '__main__':
    #imgs = glob("test_images/*.jpg")
    imgs = [ r'E:\muzzle\dataset\C - 5024\M\M-HF-5-4-C-C5024.JPG' ]
    imgs.sort()
    for img_path in imgs:
        t = time()
        img = cv2.imread(img_path)
        crop = YOLOV3_localizer(img)
        print('yolo time =',round(time()-t,3))
        if type(crop)==np.ndarray:
            plt.imshow(crop)
            plt.show()
        else:
            print('no muzzel found')









    
