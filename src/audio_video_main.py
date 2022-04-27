import numpy as np
import asciitable
import matplotlib.pyplot as plot
from scipy.io.wavfile import read, write
from scipy.fftpack import dct
import glob
import time
from pylab import *
# from python_speech_features import mfcc
import h5py
# import editdistance
import os
import itertools
import re
import datetime
from scipy import ndimage
from keras import backend as K
# from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.layers import Input, Dense, Activation, Dropout#, BatchNormalization
from keras.layers import Reshape, Lambda, merge, LSTM
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.preprocessing import sequence
import keras.callbacks


# Global variables
# grid_corpus = '../../../himanshu/grid_corpus/'
#grid_corpus = '../../grid_corpus/'
grid_corpus = 'G:/himanshu/grid_corpus/'
mfcc_audios = []#h5py.File(grid_corpus + 's' + str(1) + '/audio_mfcc.hdf5',"r")
videos = []
align_list = []
F = 50000.0

# Input Parameters
time_length = 15    ##120
feature_length_audio = 120
feature_length_video = 64 ##300    ##120
num_class = 28
max_string_len = 12

vocabulary = np.load('vocabulary.npy')

# Something about datasets
# trainX        -> shape = (-1, time_length, feature_length_video)
# trainY        -> shape = (-1, max_string_len, num_class)
#               -> first label_length will have strings, remaining will be pad with zeros
# input_length  -> shape = (-1,)
#               -> value = time_length // (pool_size ** 2) -2
# label_length  -> shape = (-1,)
#               -> value = length of each string
# textX         -> shape = (-1, string_length)
#               -> value = text labels

# Network parameters
conv_num_filters = 16   ## 16
filter_size = 4
pool_size = 2
size_before_lstm_video = 32
size_before_lstm_audio = 64
rnn_size = 512

if K.image_dim_ordering() == 'th':
    input_shape_audio = (1, time_length*8, feature_length_audio)
else:
    input_shape_audio = (time_length*8, feature_length_audio, 1)

if K.image_dim_ordering() == 'th':
    input_shape_video = (1, time_length, feature_length_video, feature_length_video)
else:
    input_shape_video = (time_length, feature_length_video, feature_length_video, 1)

input_time_length = time_length ##// (pool_size ** 2)

# def editdistance(s, t):
#     """ 
#         iterative_levenshtein(s, t) -> ldist
#         ldist is the Levenshtein distance between the strings 
#         s and t.
#         For all i and j, dist[i,j] will contain the Levenshtein 
#         distance between the first i characters of s and the 
#         first j characters of t
#     """
#     rows = len(s)+1
#     cols = len(t)+1
#     dist = [[0 for x in range(cols)] for x in range(rows)]
#     # source prefixes can be transformed into empty strings 
#     # by deletions:
#     for i in range(1, rows):
#         dist[i][0] = i
#     # target prefixes can be created from an empty source string
#     # by inserting the characters
#     for i in range(1, cols):
#         dist[0][i] = i
        
#     for col in range(1, cols):
#         for row in range(1, rows):
#             if s[row-1] == t[col-1]:
#                 cost = 0
#             else:
#                 cost = 1
#             dist[row][col] = min(dist[row-1][col] + 1,      # deletion
#                                  dist[row][col-1] + 1,      # insertion
#                                  dist[row-1][col-1] + cost) # substitution
 
#     return dist[-1][-1]



def  editdistance(target,source):
   """ Minimum edit distance. Straight from the recurrence. """

   i = len(target); j = len(source)

   if i == 0:  return j
   elif j == 0: return i

   return(min(editdistance(target[:i-1],source)+1,
              editdistance(target, source[:j-1])+1,
              editdistance(target[:i-1], source[:j-1])+substCost(source[j-1], target[i-1])))

def substCost(x,y):
    if x == y: return 0
    else: return 2



def text_to_labels(text):
    ret = []
    ret.append(26)
    #ret.append(26)
    for char in text:
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
            #ret.append(26)
    ret.append(26)
    return ret

def get_feature(c):
    # c = mfcc(audio, F, winlen=0.01, winstep=0.005, numcep=40, nfilt=80, nfft=4096, lowfreq=80, highfreq=8000)
    try:
        c = np.concatenate((np.zeros((len(c),2)), c, np.zeros((len(c),2))), axis=1)
    except ValueError:
        print(c.shape,c)
    delta1 = (c[:,2:]-c[:,0:-2])/2
    delta2 = (delta1[:,2:]-delta1[:,0:-2])/2
    feature = np.concatenate((c[:,2:-2], delta1[:,1:-1], delta2), axis=1)
    return feature

def get_video_label1( size =100, cookies =[0,0,0], string='train'):
    global mfcc_audios
    global videos
    global align_list
    [i0,j0,k0] = cookies
    if string=='train':
        [start, end] = [1, 3]       ##change 2->30
    elif string=='test':
        [start, end] = [1, 31]
    count = 0
    data_audio, data_video , data_label = [],[],[]
    while(1):
        for i in range(i0, end):
            if [j0,k0]==[0,0]:
                # mfcc_audios.close()
                mfcc_audios = h5py.File(grid_corpus + 's' + str(i+1) + '/audio_mfcc.hdf5',"r") 
                videos = h5py.File(grid_corpus + 's' + str(i+1) + '/video' + '/video_lip.hdf5',"r")  
                align_list = np.sort(glob.glob(grid_corpus + 's' + str(i+1) + '/align/*.align'))
            for j in range(j0, len(align_list)):
                try:
                    mfcc_audio = mfcc_audios[align_list[j][-12:-6]][:]
                    video = videos[align_list[j][-12:-6]][:]
                    # dct_video = np.repeat(dct_video,8,axis=0)
                except KeyError:
                    continue
                align = asciitable.read(align_list[j])

                for k in range(k0, len(align)):
                    if align[k][2] =='sil' or not(time_length> (align[k][1]//1000 - align[k][0]//1000)> 1):
                        continue
                    data_audio.append(np.float16(np.int16(mfcc_audio[align[k][0]//125: align[k][1]//125]))/100.0)
                    data_video.append(video[align[k][0]//1000: align[k][1]//1000])  ##change
                    data_label.append(align[k][2])
                    count+=1
                    if count >=size:
                        cookies[2] = (k+1)%len(align)
                        cookies[1] = (j+1)%length if cookies[2] == 0 else j
                        cookies[0] = (i+1-start)%(end-start) +start if cookies[1] == 0 else i
                        return np.array(data_audio), np.array(data_video), np.array(data_label), cookies
                k0 = 0
            j0 = 0
        i0 = start

# for j in range( len(align_list)):
#     try:
#         video = videos[align_list[j][-12:-6]][:]
#         print(j)
#     except KeyError:
#         continue
def get_audio_label( size =100, cookies =[0,0,0], string='train'):
    global mfcc_audios
    global align_list
    [i0,j0,k0] = cookies
    if string=='train':
        [start, end] = [0, 1]       ##change 2->30
    elif string=='test':
        [start, end] = [1, 2]
    count = 0
    data_audio , data_label = [],[]
    while(1):
        for i in range(i0, end):
            if [j0,k0]==[0,0]:
                # mfcc_audios.close()
                mfcc_audios = h5py.File(grid_corpus + 's' + str(i+1) + '/audio_mfcc.hdf5',"r")  
                align_list = np.sort(glob.glob(grid_corpus + 's' + str(i+1) + '/align/*.align'))
            for j in range(j0, len(align_list)):
                try:
                    mfcc_audio = mfcc_audios[align_list[j][-12:-6]][:]
                except KeyError:
                    continue
                align = asciitable.read(align_list[j])

                for k in range(k0, len(align)):
                    if align[k][2] =='sil' or not(time_length> (align[k][1]//125 - align[k][0]//125)> 8):
                        continue
                    data_audio.append(np.float16(np.int16(mfcc_audio[align[k][0]//125: align[k][1]//125]))/100.0)
                    data_label.append(align[k][2])
                    count+=1
                    if count >=size:
                        cookies[2] = (k+1)%len(align)
                        cookies[1] = (j+1)%length if cookies[2] == 0 else j
                        cookies[0] = (i+1-start)%(end-start) +start if cookies[1] == 0 else i
                        return np.array(data_audio), np.array(data_label), cookies
                k0 = 0
            j0 = 0
        i0 = start

    # return np.array(data_audio), np.array(data_label), cookies

def get_train( size =100, cookies =[0,0,0]):
    while 1:
        data_audio, data_video, data_label, cookies = get_video_label1( size, cookies, 'train')
        print(cookies)
        [trainX1, trainX2, trainY, label_length] = [data_audio, data_video, [], []]
        for i in range(size):
            trainY.append(text_to_labels(data_label[i]))
            label_length.append(len(text_to_labels(data_label[i])))
#        input_length = np.array([np.ceil(k.shape[0]/4.0) for k in trainX1])
        trainX1 = sequence.pad_sequences(np.array(trainX1), maxlen=time_length*8, padding='post', dtype=float16).reshape(-1,time_length*8,feature_length_audio,1)
        trainX2 = sequence.pad_sequences(np.array(trainX2), maxlen=time_length, padding='post', dtype=float16).reshape(-1,time_length,feature_length_video,feature_length_video,1)
        trainY = sequence.pad_sequences(np.array(trainY), maxlen=max_string_len, padding='post', dtype=int32,value=-1)
        input_length = np.ones(size)*input_time_length
        label_length = np.array(label_length)
        textY = data_label
    # input_length = np.array([np.ceil(k.shape[0] / 4.0) for k in trainX])
        inputs = {'the_input_audio': trainX1,
                  'the_input_video': trainX2,
                  'the_labels': trainY,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': textY  # used for visualization only
                  }
    #   print inputs
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        yield (inputs, outputs)

def get_test( size =100, cookies =[30,0,0]):
    while 1:
        data_audio, data_video, data_label, cookies = get_video_label1( size, cookies, 'test')
        print(cookies)
        [trainX1, trainX2, trainY, label_length] = [data_audio, data_video, [], []]
        for i in range(size):
            trainY.append(text_to_labels(data_label[i]))
            label_length.append(len(text_to_labels(data_label[i])))
#        input_length = np.array([np.ceil(k.shape[0]/4.0) for k in trainX1])
        trainX1 = sequence.pad_sequences(np.array(trainX1), maxlen=time_length*8, padding='post', dtype=float16).reshape(-1,time_length*8,feature_length_audio,1)
        trainX2 = sequence.pad_sequences(np.array(trainX2), maxlen=time_length, padding='post', dtype=float16).reshape(-1,time_length,feature_length_video,feature_length_video,1)
        trainY = sequence.pad_sequences(np.array(trainY), maxlen=max_string_len, padding='post', dtype=int32,value=-1)
        input_length = np.ones(size)*input_time_length
        label_length = np.array(label_length)
        textY = data_label
    # input_length = np.array([np.ceil(k.shape[0] / 4.0) for k in trainX])
        inputs = {'the_input_audio': trainX1,
                  'the_input_video': trainX2,
                  'the_labels': trainY,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': textY  # used for visualization only
                  }
    #   print inputs
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        yield (inputs, outputs)

    # return trainX, trainY, input_length, label_length, textX#, cookies

def decode_batch(test_func, audio_batch, video_batch, input_length):
    ret = []
    for i in range(audio_batch.shape[0]//100):
        out = test_func([audio_batch[i*100:(i+1)*100], video_batch[i*100:(i+1)*100]])[0]
        #print ('out shape\n\n',out.shape)
        for j in range(out.shape[0]):
            # print ('inp_len',input_length[j])
            out_best = list(np.argmax(out[j],1)) #0:np.int(input_length[j])], 1))
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
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
 #   y_pred = y_pred[:, :, :]   # The first argument should be from 2: , Since K.image_dim_ordering() == 'tf' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def min_distance(res):
    result = res
    distance = np.zeros(len(vocabulary))
    for i in range(len(res)):
        for j in range(len(vocabulary)):
            distance[j] = editdistance(res[i],vocabulary[j])

        ind = argmin(distance)
        result[i] = vocabulary[ind]
    return result


class VizCallback(keras.callbacks.Callback):

    def __init__(self, test_func,text_img_gen,num_display_words=6):
        self.test_func = test_func
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words

    # def show_edit_distance(self, num):
    #     num_left = num
    #     mean_norm_ed = 0.0
    #     mean_ed = 0.0
    #     while num_left > 0:
    #         word_batch = next(self.text_img_gen)[0]
    #         num_proc = min(word_batch['the_input'].shape[0], num_left)
    #         decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
    #         for j in range(0, num_proc):
    #             edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
    #             mean_ed += float(edit_dist)
    #             mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
    #         num_left -= num_proc
    #     mean_norm_ed = mean_norm_ed / num
    #     mean_ed = mean_ed / num
    #     print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
    #           % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):

        word_batch = next(self.text_img_gen)[0]
        res = decode_batch(self.test_func, word_batch['the_input_audio'], word_batch['the_input_video'], word_batch['input_length'])
        result2 = min_distance(res.copy())
        print ('result1 is ',res[:100])
        print ('result2 is ',result2[:100])
        print ('actual string  ',list(word_batch['source_str'][:100]))
        return np.mean(res == word_batch['source_str']), np.mean(result2 == word_batch['source_str'])


# def build_model():
act = 'relu'
# for audio 
input_audio = Input(name='the_input_audio', shape=input_shape_audio, dtype='float32')
# input_audio = BatchNormalization()(input_audio)
audio = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                      activation=act, init='he_normal', name='conv11')(input_audio)
audio = MaxPooling2D(pool_size=(pool_size, pool_size), name='max11')(audio)
# audio = Dropout(.4)(audio)
audio = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                      activation=act, init='he_normal', name='conv12')(audio)
audio = MaxPooling2D(pool_size=(pool_size, pool_size), name='max12')(audio)
# audio = Dropout(.4)(audio)
audio = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                      activation=act, init='he_normal', name='conv13')(audio)
audio = MaxPooling2D(pool_size=(pool_size, pool_size), name='max13')(audio)
# audio = Dropout(.4)(audio)

# audio = MaxPooling2D(pool_size=(pool_size, pool_size), name='max3')(audio)

conv_to_rnn_dims1 = (time_length , (feature_length_audio // (pool_size ** 3)) * conv_num_filters)
audio = Reshape(target_shape=conv_to_rnn_dims1, name='reshape1')(audio)

# cuts down input size going into RNN:
audio = Dense(size_before_lstm_audio, activation=act, name='dense11')(audio)  
# audio = Dropout(.4)(audio)



input_video = Input(name='the_input_video', shape=input_shape_video, dtype='float32')
# input_video = BatchNormalization()(input_video)
video = Conv3D(conv_num_filters, filter_size, border_mode='same',
                      activation=act, init='he_normal', name='conv21')(input_video)
video = MaxPooling3D(pool_size=(1, pool_size, pool_size), name='max21')(video)
# video = Dropout(.4)(video)
video = Conv3D(conv_num_filters, filter_size, border_mode='same',
                      activation=act, init='he_normal', name='conv22')(video)
video = MaxPooling3D(pool_size=(1, pool_size, pool_size), name='max22')(video)
# video = Dropout(.4)(video)

# video = MaxPooling2D(pool_size=(pool_size, pool_size), name='max3')(video)

conv_to_rnn_dims2 = (time_length , ((feature_length_video // (pool_size ** 2))**2) * conv_num_filters)
video = Reshape(target_shape=conv_to_rnn_dims2, name='reshape2')(video)

# cuts down input size going into RNN:
video = Dense(size_before_lstm_video, activation=act, name='dense21')(video)   
# video = Dropout(.4)(video)

merged = merge([audio, video], mode='concat')


# Three layers of bidirecitonal LSTMs
lstm_1 = LSTM(rnn_size, return_sequences=True, init='he_normal', name='lstm1')(merged)
lstm_1b = LSTM(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='lstm1_b')(merged)

lstm1_merged = merge([lstm_1, lstm_1b], mode='sum')     
lstm_2 = LSTM(rnn_size, return_sequences=True, init='he_normal', name='lstm2')(lstm1_merged)
lstm_2b = LSTM(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='lstm2_b')(lstm1_merged)

lstm2_merged = merge([lstm_2, lstm_2b], mode='sum')     
lstm_3 = LSTM(rnn_size, return_sequences=True, init='he_normal', name='lstm3')(lstm2_merged)
lstm_3b = LSTM(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='lstm3_b')(lstm2_merged)

lstm3_merged = merge([lstm_3, lstm_3b], mode='concat')
# lstm3_merged = Dropout(.4)(lstm3_merged)

# transforms RNN output to character activations:
inner = Dense(num_class, init='he_normal',
              name='dense2')(lstm3_merged)      
y_pred = Activation('softmax', name='softmax')(inner)
Model(input=[input_audio, input_video], output=y_pred).summary()

labels = Input(name='the_labels', shape=[max_string_len], dtype='int32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
# Keras doesn't currently support loss funcs with extra parameters
# so CTC loss is implemented in a lambda layer
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

# clipnorm seems to speeds up convergence
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

model = Model(input=[input_audio, input_video, labels, input_length, label_length], output=[loss_out])

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

test_func = K.function([input_audio, input_video],[y_pred])

model.load_weights('../model/1_both_s2_s3_30_times.h5', by_name=True)
avg_acc1,avg_acc2 = 0,0
for i in range(1,31):
	viz_cb = VizCallback(test_func, get_test(size=5000, cookies=[i,0,0]))
	acc1, acc2 = viz_cb.on_epoch_end(2)
	print('s%s : %s, %s'%(i+1, acc1,acc2))
	avg_acc1+= acc1
	avg_acc2+= acc2
print('Avg : %s, %s'%( avg_acc1/28.0,avg_acc2/28.0))
# viz_cb = VizCallback(test_func, get_test(size=6000, cookies=[3,0,0]))

# for i in range(5):
#     model.fit_generator(generator=get_train(size=128), samples_per_epoch=5, nb_epoch=120)
#     viz_cb.on_epoch_end(2)
#     file_name = ('../model/1_both_s2_s3_'+str(6*i+6)+'_times.h5')
#     model.save_weights(file_name)
#     print(i)

# model.save_weights('../model/1_model_weights_1.h5')

#model.load_weights('../model/1_model_weights_1.h5', by_name=True) 
#model.fit_generator(generator=getdata(), samples_per_epoch=1,
#                    nb_epoch=10)

# build_model()
'''
1_both_s2_s3_30times
In [3]: viz_cb = VizCallback(test_func, get_test(size=6000, cookies=[4,0,0]))

In [4]: model.load_weights('../model/1_both_s2_s3_30_times.h5', by_name=True)

In [5]: viz_cb.on_epoch_end(2)
[5, 7, 7]
result1 is  ['bin', 'blue', 'ait', '', 'eight', 'now', 'bin', 'blue', 'it', 'e', 'nine', 'soon', 'bin', 'blue', 'ait', '
s', 'one', 'zeine', 'sevn', 'blue']
result2 is  ['bin', 'blue', 'at', 'm', 'eight', 'now', 'bin', 'blue', 't', 'e', 'nine', 'soon', 'bin', 'blue', 'at', 's'
, 'one', 'nine', 'seven', 'blue']
actual string   ['bin', 'blue', 'at', 'e', 'eight', 'now', 'bin', 'blue', 'at', 'e', 'nine', 'soon', 'bin', 'blue', 'at'
, 'f', 'one', 'again', 'bin', 'blue']
0.575166666667 0.642833333333

In [6]: viz_cb = VizCallback(test_func, get_test(size=6000, cookies=[3,0,0]))

In [7]: viz_cb.on_epoch_end(2)
[4, 13, 2]
result1 is  ['bin', 'blue', 'at', 'e', 'nine', 'now', 'bin', 'blue', 'at', 'f', 'one', 'bin', 'blue', 'at', 'a', 'two',
'again', 'bin', 'blue', 'at']
result2 is  ['bin', 'blue', 'at', 'e', 'nine', 'now', 'bin', 'blue', 'at', 'f', 'one', 'bin', 'blue', 'at', 'a', 'two',
'again', 'bin', 'blue', 'at']
actual string   ['bin', 'blue', 'at', 'e', 'nine', 'now', 'bin', 'blue', 'at', 'f', 'one', 'bin', 'blue', 'at', 'f', 'tw
o', 'again', 'bin', 'blue', 'at']
0.939833333333 0.951666666667


'''



# trainX, trainY, input_length, label_length, textX = getdata(time_length, input_time_length, max_string_len)
'''
result is  

 ['set', 'blue', 'at', 'h', 'ne', 'again', 'set', 'blue', 'at', 'h', 'zero', 'please', 'set', 'blue', 'at', 'ein']
actual strig

 ['set' 'blue' 'at' 'h' 'one' 'again' 'set' 'blue' 'at' 'h' 'zero' 'please'
 'set' 'blue' 'at' 'n']
0.875
'''
# write('../data/s1.wav', 50000, data_audio[0])

