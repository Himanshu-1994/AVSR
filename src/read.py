import asciitable
from pyaudio import PyAudio
import numpy as np
import matplotlib.pyplot as plot
from scipy.io.wavfile import read
from scipy.fftpack import dct
import glob
import time
from pylab import *





def sound(y,F): 
	p = PyAudio()
	stream = p.open(format=p.get_format_from_width(1),channels=1,rate=int(F),output=True)

	my=max(abs(y))
	if my>1:
		y=y/my
	samples = ( int(k*127 +128) for k in y  )
	for buf in zip(*[samples]): 
		stream.write(bytes(bytearray(buf)))
	stream.stop_stream()
	stream.close()
	p.terminate()

a=asciitable.read('align/bbaf2n.align')
F,s=read('s1/', mmap=False)