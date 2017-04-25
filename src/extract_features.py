import sys
import os
import dlib
import glob
from skimage import io
import cv2
from scipy import misc
import numpy as np
from scipy.fftpack import dct, idct
import h5py
from python_speech_features import mfcc
from scipy.io.wavfile import read, write

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


predictor_path = '../shape_predictor_68_face_landmarks.dat'
data_path = '../../grid_corpus/'
#faces_folder_path = './face_frames/face_data/'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()


def get_feature(audio,F):
    c = mfcc(audio, F, winlen=0.01, winstep=0.005, numcep=40, nfilt=80, nfft=4096, lowfreq=80, highfreq=8000)
    c = np.concatenate((np.zeros((len(c),2)), c, np.zeros((len(c),2))), axis=1)
    delta1 = (c[:,2:]-c[:,0:-2])/2
    delta2 = (delta1[:,2:]-delta1[:,0:-2])/2
    feature = np.concatenate((c[:,2:-2], delta1[:,1:-1], delta2), axis=1)
    return feature


def detect(img):

    status = 0
    dets = detector(img, 1)

  #  print("Number of faces detected: {}".format(len(dets)))

    # Get the landmarks/parts for the face in box d.
    if len(dets) ==0:
        print ('error, dets = 0' )
        return 0,status

    shape = predictor(img, dets[0])

    xlist = []
    ylist = []
    for i in range(48, 68):      # 48,68 contains the lip features
        # Store X and Y coordinates in two lists
        xlist.append(float(shape.part(i).x))
        ylist.append(float(shape.part(i).y))

    xmin = np.int(min(xlist))
    xmax = np.int(max(xlist))

    ymin = np.int(min(ylist))
    ymax = np.int(max(ylist))

    img1 = img
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    img_crop = img1[ymin:ymax, xmin:xmax]
    gray_img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    img_mouth = np.array(misc.imresize(gray_img, (64, 64)), dtype='float') / 255.0

    dct_img = dct2(img_mouth)
    dcts = dct_img.reshape(64*64)
    inds = np.argpartition(abs(dcts),-500)[-500:]
    select_dct = dcts[inds]
    sorted_dct = select_dct[np.argsort(select_dct)]
    sorted_dct = np.fliplr([sorted_dct])[0]

    status = 1
    return sorted_dct,status


for i in range(33):

    badfiles = []
    audio_list = np.sort(glob.glob(data_path + 's' + str(i + 1) + '/*.wav'))
    align_list = np.sort(glob.glob(data_path + 's' + str(i + 1) + '/align/*.align'))
    video_list = np.sort(glob.glob(data_path + 's' + str(i + 1) + '/video/mpg_6000/*.mpg'))

    if len(audio_list) != len(align_list) != len(video_list):
        length = min(len(audio_list), len(align_list))
        print('Error! not equal length in s' + str(i + 1))
    else:
        length = len(audio_list)

    f_vid = h5py.File((data_path + 's' + str(i+1) + '/video/' + 'video_dct100' + '.hdf5'), 'w')
    f_aud = h5py.File((data_path + 's' + str(i+1)+'/audio_mfcc' + '.hdf5'), 'w')

    for j in range(0, length):
        print ('file',j)
        print (audio_list[j])
        if audio_list[j][-10:-4] != align_list[j][-12:-6] != video_list[j][:-4]:
            print(
            "Error! audio file name " + audio_list[j] + " doesn't match with align or video file name " + align_list[j])

        print (video_list[j],'video')
        video = cv2.VideoCapture(video_list[j])

        rd = False
        rd,frame = video.read()

        vid_dct = []

        while rd:
            dct_100,status = detect(frame)

            #vid_dct.append(insert)
            if status == 0:
                badfiles.append(j)
                break
            vid_dct.append(dct_100)
            rd,frame = video.read()

        if status == 0:
            continue
        # vid_dct.append(vid_dct[-1])
        video_dct_interp = np.asarray(vid_dct)
        dset = f_vid.create_dataset(video_list[j][-10:-4], data=video_dct_interp, compression="gzip", chunks=True)
        print ('size of dct',video_dct_interp.shape)

        F, audio = read(audio_list[j])
        # print ('rate',F)
        mfcc_vec = get_feature(audio, F)
        # sa = 'zipped'+str(j)
        dset_a = f_aud.create_dataset(audio_list[j][-10:-4], data=mfcc_vec, compression="gzip", chunks=True)

    print (i,'complete') 
  


