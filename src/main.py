import numpy as np
import asciitable
import matplotlib.pyplot as plot
from scipy.io.wavfile import read, write
from scipy.fftpack import dct
import glob
import time
from pylab import *
from python_speech_features import mfcc

import os
import itertools
import re
import datetime
from scipy import ndimage
from keras import backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, merge, LSTM
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import sequence
import keras.callbacks


# Global variables
# grid_corpus = '../../../himanshu/grid_corpus/'
grid_corpus = 'grid/'
F = 50000.0

# Input Parameters
time_length = 60
feature_length = 120
num_class = 27
max_string_len = 16

# Something about datasets
# trainX        -> shape = (-1, time_length, feature_length)
# trainY        -> shape = (-1, max_string_len, num_class)
#               -> first label_length will have strings, remaining will be pad with zeros
# input_length  -> shape = (-1,)
#               -> value = time_length // (pool_size ** 2) -2
# label_length  -> shape = (-1,)
#               -> value = length of each string
# textX         -> shape = (-1, string_length)
#               -> value = text labels

# Network parameters
conv_num_filters = 16
filter_size = 3
pool_size = 2
time_dense_size = 32
rnn_size = 512

if K.image_dim_ordering() == 'th':
    input_shape = (1, time_length, feature_length)
else:
    input_shape = (time_length, feature_length, 1)

input_time_length = time_length // (pool_size ** 2) -2


def text_to_labels(text):
    ret = []
    for char in text:
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
        elif char == ' ':
            ret.append(26)
    return ret

def get_feature(audio):
    c = mfcc(audio, F, winlen=0.01, winstep=0.005, numcep=40, nfilt=80, nfft=4096, lowfreq=80, highfreq=8000)
    c = np.concatenate((np.zeros((len(c),2)), c, np.zeros((len(c),2))), axis=1)
    delta1 = (c[:,2:]-c[:,0:-2])/2
    delta2 = (delta1[:,2:]-delta1[:,0:-2])/2
    feature = np.concatenate((c[:,2:-2], delta1[:,1:-1], delta2), axis=1)
    return feature

def get_audio_label( size =100, cookies =[0,0,0]):
    [i0,j0,k0] = cookies
    count = 0
    data_audio , data_label = [],[]
    for i in range(i0, 33): 

        audio_list = np.sort(glob.glob(grid_corpus + 's' + str(i+1) + '/*.wav'))
        align_list = np.sort(glob.glob(grid_corpus + 's' + str(i+1) + '/align/*.align'))
        if len(audio_list) !=len(align_list):
            lenth = min(len(audio_list),len(align_list))
            print('Error! not equal length in s' + str(i+1))
        else:
            length = len(audio_list)

        for j in range(j0, length):

            if audio_list[j][-10:-4] != align_list[j][-12:-6]:
                print("Error! audio file name " + audio_list[j] + " doesn't match with align file name " + align_list[j])

            align = asciitable.read(align_list[j])
            F, audio = read(audio_list[j], mmap=False)
            if len(align) ==0:
                print('Error! align file ' + align_list[j] +' is empty')
            for k in range(k0, len(align)):
                if align[k][2] =='sil':
                    continue
                data_audio.append(audio[2*align[k][0]: 2*align[k][1]])
                data_label.append(align[k][2])
                count+=1
                if count >=size:
                    cookies[2] = (k+1)%len(align)
                    cookies[1] = (j+1)%length if cookies[2] == 0 else j
                    cookies[0] = (i+1)%33 if cookies[1] == 0 else i
                    break
            else:
                k0 = 0
                continue
            break
        else:
            j0 = 0
            continue
        break
    i0 = 0

    return np.array(data_audio), np.array(data_label), cookies

def getdata(time_length = 60, input_time_length = 13, max_string_len = 12, size =100, cookies =[0,0,0]):
    data_audio, data_label, cookies = get_audio_label( size, cookies)
    trainX, trainY, label_length = [], [], []
    for i in range(size):
        trainX.append(get_feature(data_audio[i]))
        trainY.append(text_to_labels(data_label[i]))
        label_length.append(len(data_label[i]))
    trainX = sequence.pad_sequences(np.array(trainX), maxlen=time_length, padding='post', dtype=float)
    trainY = sequence.pad_sequences(np.array(trainY), maxlen=max_string_len, padding='post', dtype=float)
    input_length = np.ones(size)*input_time_length
    label_length = np.array(label_length)
    textX = data_label

    return trainX, trainY, input_length, label_length, textX#, cookies

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        # 26 is space, 27 is CTC blank char
        outstr = ''
        for c in out_best:
            if c >= 0 and c < 26:
                outstr += chr(c + ord('a'))
            elif c == 26:
                outstr += ' '
        ret.append(outstr)
    return ret

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]   # The first argument should be from 2: , Since K.image_dim_ordering() == 'tf' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_model():
    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                          activation=act, init='he_normal', name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                          activation=act, init='he_normal', name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (time_length // (pool_size ** 2), (feature_length // (pool_size ** 2)) * conv_num_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN: 
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)    # Giving dimension error ??????????????????????????????????????????????????

    # Two layers of bidirecitonal LSTMs
    lstm_1 = LSTM(rnn_size, return_sequences=True, init='he_normal', name='lstm1')(inner)
    lstm_1b = LSTM(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='lstm1_b')(inner)
    lstm1_merged = merge([lstm_1, lstm_1b], mode='sum')     #see whether to sum or concat ??????????????????????????????????????????????????????????????
    lstm_2 = LSTM(rnn_size, return_sequences=True, init='he_normal', name='lstm2')(lstm1_merged)
    lstm_2b = LSTM(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='lstm2_b')(lstm1_merged)

    # transforms RNN output to character activations:
    inner = Dense(num_class, init='he_normal',
                  name='dense2')(merge([lstm_2, lstm_2b], mode='concat'))   # Giving dimension error ?????????????????????????????????????????
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(input=[input_data], output=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(input=[input_data, labels, input_length, label_length], output=[loss_out])

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    return model

# trainX, trainY, input_length, label_length, textX = getdata(time_length, input_time_length, max_string_len)

model = build_model()
model.fit_generator(generator=getdata(), samples_per_epoch=40, nb_epoch=10)

# write('../data/s1.wav', 50000, data_audio[0])
