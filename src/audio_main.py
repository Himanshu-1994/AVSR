import numpy as np
import asciitable
import matplotlib.pyplot as plot
from scipy.io.wavfile import read, write
from scipy.fftpack import dct
import glob
import time
from pylab import *
from python_speech_features import mfcc
import h5py
import os
import itertools
import re
import datetime
from scipy import ndimage
from keras import backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers import Reshape, Lambda, merge, LSTM
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.preprocessing import sequence
import keras.callbacks

# Global variables
grid_corpus = 'G:/himanshu/grid_corpus/'
mfcc_audios = []
align_list = []
F = 50000.0

# Input Parameters
time_length = 120
feature_length = 120
num_class = 28
max_string_len = 12

vocabulary = np.load('vocabulary.npy')

#
# trainX        -> shape = (-1, time_length, feature_length)
# trainY        -> shape = (-1, max_string_len, num_class)
#               -> first label_length will have strings, remaining will be pad with zeros
# input_length  -> shape = (-1,)
#               -> value = time_length // (pool_size ** 2) -2
# label_length  -> shape = (-1,)
#               -> value = length of each string
# testX         -> shape = (-1, string_length)
#               -> value = text labels

# Network parameters

conv_num_filters = 16
filter_size = 4
pool_size = 2
time_dense_size = 32
rnn_size = 512

if K.image_dim_ordering() == 'th':
    input_shape = (1, time_length, feature_length)
else:
    input_shape = (time_length, feature_length, 1)

input_time_length = time_length // (pool_size ** 2)


def editdistance(target, source):

    i = len(target)
    j = len(source)

    if i == 0:
        return j
    elif j == 0:
        return i

    return (min(editdistance(target[:i - 1], source) + 1,
                editdistance(target, source[:j - 1]) + 1,
                editdistance(target[:i - 1], source[:j - 1]) + substCost(source[j - 1], target[i - 1])))


def substCost(x, y):
    if x == y:
        return 0
    else:
        return 2


def text_to_labels(text):
    ret = []
    ret.append(26)
    for char in text:
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
    ret.append(26)
    return ret


def get_feature(audio):
    c = mfcc(audio, F, winlen=0.01, winstep=0.005, numcep=40, nfilt=80, nfft=4096, lowfreq=80, highfreq=8000)
    c = np.concatenate((np.zeros((len(c), 2)), c, np.zeros((len(c), 2))), axis=1)
    delta1 = (c[:, 2:] - c[:, 0:-2]) / 2
    delta2 = (delta1[:, 2:] - delta1[:, 0:-2]) / 2
    feature = np.concatenate((c[:, 2:-2], delta1[:, 1:-1], delta2), axis=1)
    return feature


def get_audio_label(size=100, cookies=[0, 0, 0], string='train'):
    global mfcc_audios
    global align_list
    [i0, j0, k0] = cookies
    if string == 'train':
        [start, end] = [1, 3]
    elif string == 'test':
        [start, end] = [3, 10]
    count = 0
    data_audio, data_label = [], []
    while (1):
        for i in range(i0, end):
            if [j0, k0] == [0, 0]:
                mfcc_audios = h5py.File(grid_corpus + 's' + str(i + 1) + '/audio_mfcc.hdf5', "r")
                align_list = np.sort(glob.glob(grid_corpus + 's' + str(i + 1) + '/align/*.align'))
            for j in range(j0, len(align_list)):
                try:
                    mfcc_audio = mfcc_audios[align_list[j][-12:-6]][:]
                except KeyError:
                    continue
                align = asciitable.read(align_list[j])

                for k in range(k0, len(align)):
                    if align[k][2] == 'sil' or not (time_length > (align[k][1] // 125 - align[k][0] // 125) > 8):
                        continue
                    data_audio.append(np.float16(np.int16(mfcc_audio[align[k][0] // 125: align[k][1] // 125])) / 100.0)
                    data_label.append(align[k][2])
                    count += 1
                    if count >= size:
                        cookies[2] = (k + 1) % len(align)
                        cookies[1] = (j + 1) % length if cookies[2] == 0 else j
                        cookies[0] = (i + 1 - start) % (end - start) + start if cookies[1] == 0 else i
                        return np.array(data_audio), np.array(data_label), cookies
                k0 = 0
            j0 = 0
        i0 = start
        return

def get_train(size=100, cookies=[0, 0, 0]):
    while 1:
        data_audio, data_label, cookies = get_audio_label(size, cookies, 'train')
        print(cookies)
        [trainX, trainY, label_length] = [data_audio, [], []]
        for i in range(size):
            # trainX.append(get_feature(data_audio[i]))
            trainY.append(text_to_labels(data_label[i]))
            label_length.append(len(text_to_labels(data_label[i])))
        # input_length = np.array([np.ceil(k.shape[0]/4.0) for k in trainX])
        trainX = sequence.pad_sequences(np.array(trainX), maxlen=time_length, padding='post', dtype=float16).reshape(-1,
                                                                                                                     time_length,
                                                                                                                     feature_length,
                                                                                                                     1)
        trainY = sequence.pad_sequences(np.array(trainY), maxlen=max_string_len, padding='post', dtype=int32, value=-1)
        input_length = np.ones(size) * input_time_length
        label_length = np.array(label_length)
        textY = data_label
        # input_length = np.array([np.ceil(k.shape[0] / 4.0) for k in trainX])
        inputs = {'the_input': trainX,
                  'the_labels': trainY,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': textY  # used for visualization only
                  }

        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        yield (inputs, outputs)


def get_test(size=100, cookies=[30, 0, 0]):
    while 1:
        data_audio, data_label, cookies = get_audio_label(size, cookies, 'test')
        print(cookies)
        [trainX, trainY, label_length] = [data_audio, [], []]
        for i in range(size):
            # trainX.append(get_feature(data_audio[i]))
            trainY.append(text_to_labels(data_label[i]))
            label_length.append(len(text_to_labels(data_label[i])))
        # input_length = np.array([np.ceil(k.shape[0]/4.0) for k in trainX])
        trainX = sequence.pad_sequences(np.array(trainX), maxlen=time_length, padding='post', dtype=float16).reshape(-1,
                                                                                                                     time_length,
                                                                                                                     feature_length,
                                                                                                                     1)
        trainY = sequence.pad_sequences(np.array(trainY), maxlen=max_string_len, padding='post', dtype=int32, value=-1)
        input_length = np.ones(size) * input_time_length
        label_length = np.array(label_length)
        textY = data_label
        # input_length = np.array([np.ceil(k.shape[0] / 4.0) for k in trainX])
        inputs = {'the_input': trainX,
                  'the_labels': trainY,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': textY  # used for visualization only
                  }

        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        yield (inputs, outputs)


def decode_batch(test_func, word_batch,input_length):
    ret = []
    for i in range(word_batch.shape[0] // 1000):
        out = test_func([word_batch[i * 1000:(i + 1) * 1000]])[0]
        # print ('out shape\n\n',out.shape)
        for j in range(out.shape[0]):
            # print ('inp_len',input_length[j])
            out_best = list(np.argmax(out[j], 1))  # 0:np.int(input_length[j])], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            # 26 is CTC blank char
            outstr = ''
            for c in out_best:
                if c >= 0 and c < 26:
                    outstr += chr(c + ord('a'))
                    # elif c == 26:
                    #     outstr += ' '
            ret.append(outstr)
    return ret


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def min_distance(res):
    result = res
    distance = np.zeros(len(vocabulary))
    for i in range(len(res)):
        for j in range(len(vocabulary)):
            distance[j] = editdistance(res[i], vocabulary[j])

        ind = argmin(distance)
        result[i] = vocabulary[ind]
    return result


class VizCallback(keras.callbacks.Callback):
    def __init__(self, test_func, text_img_gen, num_display_words=6):
        self.test_func = test_func
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words

    def on_epoch_end(self, epoch, logs={}):
        word_batch = next(self.text_img_gen)[0]
        res = decode_batch(self.test_func, word_batch['the_input'], word_batch['input_length'])
        result2 = min_distance(res.copy())
        print ('result1 is ', res[:20])
        print ('result2 is ', result2[:20])
        print ('actual string  ', list(word_batch['source_str'][:20]))
        print(np.mean(res == word_batch['source_str']), np.mean(result2 == word_batch['source_str']))


# def build_model():
act = 'relu'
input_data = Input(name='the_input', shape=input_shape, dtype='float32')
inner = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                      activation=act, init='he_normal', name='conv1')(input_data)
inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
# inner = Dropout(0.4)(inner)
inner = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                      activation=act, init='he_normal', name='conv2')(inner)
inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)
# inner = Dropout(0.4)(inner)

# inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max3')(inner)

conv_to_rnn_dims = (time_length // (pool_size ** 2), (feature_length // (pool_size ** 2)) * conv_num_filters)
inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

# cuts down input size going into RNN:
inner = Dense(time_dense_size, activation=act, name='dense1')(inner)
# inner = Dropout(0.4)(inner)

# Three layers of bidirecitonal LSTMs
lstm_1 = LSTM(rnn_size, return_sequences=True, init='he_normal', name='lstm1')(inner)
lstm_1b = LSTM(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='lstm1_b')(inner)
################## change here
lstm1_merged = merge([lstm_1, lstm_1b], mode='sum')
lstm_2 = LSTM(rnn_size, return_sequences=True, init='he_normal', name='lstm2')(lstm1_merged)
lstm_2b = LSTM(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='lstm2_b')(lstm1_merged)

lstm2_merged = merge([lstm_2, lstm_2b], mode='sum')
lstm_3 = LSTM(rnn_size, return_sequences=True, init='he_normal', name='lstm3')(lstm2_merged)
lstm_3b = LSTM(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='lstm3_b')(lstm2_merged)

lstm3_merged = merge([lstm_3, lstm_3b], mode='concat')
# lstm3_merged = Dropout(0.4)(lstm3_merged)

# transforms RNN output to character activations:
inner = Dense(num_class, init='he_normal',
              name='dense2')(lstm3_merged)
y_pred = Activation('softmax', name='softmax')(inner)
# Model(input=[input_data], output=y_pred).summary()

labels = Input(name='the_labels', shape=[max_string_len], dtype='int32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

model = Model(input=[input_data, labels, input_length, label_length], output=[loss_out])

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

test_func = K.function([input_data], [y_pred])

viz_cb = VizCallback(test_func, get_test(size=5000, cookies=[4, 0, 0]))

for i in range(14):
    model.fit_generator(generator=get_train(size=128), samples_per_epoch=5, nb_epoch=10)
    viz_cb.on_epoch_end(2)
    print(i)

#model.save_weights('../model/1_model_weights_1.h5')

# model.load_weights('../model/1_model_weights_1.h5', by_name=True)
# model.fit_generator(generator=getdata(), samples_per_epoch=1,
#                    nb_epoch=10)

# build_model()




# trainX, trainY, input_length, label_length, textX = getdata(time_length, input_time_length, max_string_len)
'''
Sample result

Actual String: ['set', 'blue', 'at', 'h', 'ne', 'again', 'set', 'blue', 'at', 'h', 'zero', 'please', 'set', 'blue', 'at', 'ein']

Predicted String ['set' 'blue' 'at' 'h' 'one' 'again' 'set' 'blue' 'at' 'h' 'zero' 'please'
 'set' 'blue' 'at' 'n']
0.875
'''
       
