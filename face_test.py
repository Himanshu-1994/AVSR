import sys
import os
import dlib
import glob
from skimage import io
import cv2
from scipy import misc
import numpy as np
from scipy.fftpack import dct, idct


def dct2 (block):
  return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')


predictor_path = './trained_features/shape_predictor_68_face_landmarks.dat'
faces_folder_path = './face_frames/face_data/'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):

    print("Processing file: {}".format(f))
    img = io.imread(f)
    dets = detector(img, 1)

    print("Number of faces detected: {}".format(len(dets)))

    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, dets[0])
    print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
		                                  shape.part(1)))
	
	 
    xlist = []
    ylist = []
    for i in range(48,68): 
#Store X and Y coordinates in two lists
        xlist.append(float(shape.part(i).x))
        ylist.append(float(shape.part(i).y))
	
 
    xmin = np.int(min(xlist))
    xmax = np.int(max(xlist))
        
    ymin = np.int(min(ylist))
    ymax = np.int(max(ylist))

    #c1 = [xlistxmin),ylist(ymin)]
    #c2 = [xlist(xmax),ylist(ymax)]
    
    img1 = img
    cv2.rectangle(img, (xmin,ymin), (xmax,ymax) , (255,0,0), 2)
	   
   # for i in range(48,68):
   #     cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) 
    
    img_crop = img1[ymin:ymax,xmin:xmax]
    gray_img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    img_mouth = np.array(misc.imresize(gray_img,(64,64)),dtype='float')/255.0
  
    dct_img = dct2(img_mouth)

    misc.imshow(img)
    misc.imshow(img_crop)
    misc.imshow(img_mouth)



