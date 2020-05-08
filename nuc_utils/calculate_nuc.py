import numpy as np
import os
import math

AVG = np.array([0, 0, 0, 0, 0, 0, 0])

K = 20
D = 40
def read_dir(filedir, lable):
    if lable == 0:
        print('calculating AVG ....')
        avg = np.array([0, 0, 0, 0, 0, 0, 0])
        pathDir = os.listdir(filedir)
        for allDir in pathDir:
            if allDir.find('density')>0:
	            child = os.path.join('%s/%s' % (filedir, allDir))
	            print(child)
	            avg = avg + read_each_file(child,lable)
        return avg
    else:
        print('calculating NUC ....')
        cnt = np.array([0, 0, 0, 0, 0, 0, 0])
        pathDir = os.listdir(filedir)
        for allDir in pathDir:
	        if allDir.find('density') > 0:
	            child = os.path.join('%s/%s' % (filedir, allDir))
	            print(child)
	            tmp = read_each_file(child, lable)
	            cnt = cnt + tmp
        print('cnt=', cnt)
        ret = np.sqrt(cnt / (K * D * 1.0))
        return ret

def read_each_file(filepath,lable):
    if lable == 0: # add all the values of XXX_density.xyz
        ret = np.array([0, 0, 0, 0, 0, 0, 0])
        with open(filepath,'r') as f:
            all_data = f.readlines()
            for line in all_data:
                tmp = line.strip().split(' ')
                val = np.array(list(map(float, tmp)))
                ret = ret + val
        return ret
    else:
        ret = np.array([0, 0, 0, 0, 0, 0, 0])
        ans=0
        lab=0
        with open(filepath, 'r') as f:
            all_data = f.readlines()
            for line in all_data:
                tmp = line.strip().split(' ')
                val = np.array(list(map(float, tmp)))
                ret = ret + (val - AVG) * (val - AVG) # np.square(val-AVG)
        return ret

if __name__ == '__main__':
    filedir = './outputs/punet_skip_conn_decay'
    # get the sum of all the values in XXX_density.xyz files
    avg = read_dir(filedir, 0)
    AVG = avg / (K * D * 1.0)
    # get NUC value
    NUC = read_dir(filedir,1)
    print('NUC is')
    print(NUC)
