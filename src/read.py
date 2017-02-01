import asciitable
from pyaudio import PyAudio
import numpy as np
import matplotlib.pyplot as plot
from scipy.io.wavfile import read, write
from scipy.fftpack import dct
import glob
import time
from pylab import *

grid_corpus = '../../../himanshu/grid_corpus/'
# grid_corpus = '../data/'
def get_audio_label(cookies =[0,0,0], size =100):
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
			_, audio = read(audio_list[j], mmap=False)
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

data_audio, data_label, cookies = get_audio_label()
# count = 0
# for i in range(5):
# 	for j in range(5):
# 		for k in range(10):
# 			count+=1
# 			print(count,i,j,k)
# 			if count>=50:
# 				break
# 		else:
# 			continue
# 		break
# 	else:
# 		continue
# 	break


# def sound(y,F): 
# 	p = PyAudio()
# 	stream = p.open(format=p.get_format_from_width(1),channels=1,rate=int(F),output=True)

# 	my=max(abs(y))
# 	if my>1:
# 		y=y/my
# 	samples = ( int(k*127 +128) for k in y  )
# 	for buf in zip(*[samples]): 
# 		stream.write(bytes(bytearray(buf)))
# 	stream.stop_stream()
# 	stream.close()
# 	p.terminate()

# a = asciitable.read(grid_corpus+'align/bbaf2n.align')
# F,s = read('s1/', mmap=False)
# write('../data/s1.wav', 50000, data_audio[0])
