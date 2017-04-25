import numpy as np
import asciitable
import glob
import h5py

grid_corpus = 'G:/himanshu/grid_corpus/'
align_list = []

align_list = np.sort(glob.glob(grid_corpus + 's1'  + '/align/*.align'))
data_label = set()

for j in range(0, len(align_list)):
    align = asciitable.read(align_list[j])

    for k in range(0, len(align)):
        if align[k][2] == 'sil':
            continue
        data_label.add(align[k][2])

vocabulary = list(data_label)

   
