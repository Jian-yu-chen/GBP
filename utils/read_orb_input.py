import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.lie_algebra import so3log

def read_orb_input(filename):
    lm_list = []
    kf_list = []
    mea_list = []
    intrinsic = []
    maxKFId = 0
    kf_id_list = []
    with open(filename, 'r') as f:
        version = 0
        for line in f.readlines():
            line = line.rstrip('\n')
            if line[:12] == '# nMapPoints':
                version = 0
                continue
            elif line[:12] == '# nKeyFrames':
                version = 1
                continue
            elif line[:13] == '# Measurement':
                version = 2
                continue
                
            l = line.split(' ')
            
            if version == 0:
                # MapPoint : id x y z
                l[0] = int(l[0])
                l[1:] = [float(i) for i in l[1:]]
                lm_list.append(l)
            elif version == 1:
                # KeyFrames : id timestamp [rho q] fx fy cx cy
                l[0] = int(l[0])
                kf_id_list.append(l[0])
                l[1:9] = [float(i) for i in l[1:9]]
                q = l[5:9]
                r_wc = R.from_quat(q).as_matrix()
                # camera coordinate
                omega_cw = so3log(r_wc.T)
                # omega = so3log(r.as_matrix().T)
                t_wc = np.array(l[2:5])
                t_cw = -r_wc.T @ t_wc.T
                
                kf_info = l[:2]
                for t in t_cw:
                    kf_info.append(t)
                
                for o in omega_cw:
                    kf_info.append(o)
                kf_list.append(kf_info)
                
                if l[0] == 0:
                    intrinsic = [float(i) for i in l[9:]]
                    
                if l[0] > maxKFId:
                    maxKFId = l[0]
                    
            elif version == 2:
                # Measurment : kfId lmId x y scale_factor
                l[:2] = [int(i) for i in l[:2]]
                l[2:] = [float(i) for i in l[2:]]
                
                # l[2] = l[2] / 1241.
                # l[3] = l[3] / 1241.
                mea_list.append(l)
    
    new_kf_list = kf_list
    # for i in range(len(kf_list)-1):
    #     new_kf_list[kf_id_list.index(i+1)][4] = kf_list[kf_id_list.index(i+1)][4] - kf_list[kf_id_list.index(i)][4]
    
    for idx in range(len(lm_list)):
        lm_list[idx][0] = lm_list[idx][0] + maxKFId + 1
    
    for idx in range(len(mea_list)):
        mea_list[idx][1] = mea_list[idx][1] + maxKFId + 1        

    K = np.zeros([3, 3])
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = intrinsic
    # K[0, 0], K[1, 1], K[0, 2], K[1, 2] = K[0, 0]/1241., K[1, 1]/1241., K[0, 2]/1241., K[1, 2]/1241.
    K[2, 2] = 1.
    
    
    kf_array = np.array(kf_list)
    lm_array = np.array(lm_list)
    mea_array = np.array(mea_list)
    
    print("keyframe number: {}\nlandmark number: {}\nmeasurement: {}".format(kf_array.shape[0], lm_array.shape[0], mea_array.shape[0]))
    return kf_array, lm_array, mea_array, K


if __name__=="__main__":
    
    read_orb_input('../data/Map.txt')