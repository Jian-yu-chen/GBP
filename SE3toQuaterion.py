import numpy as np
import math
from utils.lie_algebra import so3exp
from scipy.spatial.transform import Rotation as R


SE3_pose = np.loadtxt('./data/Map_keyframe.txt')


# SE3 to quaterion
result = []
for SE3 in SE3_pose:
    result.append([])
    for i in SE3[1:5]:
        result[-1].append(i)
    q = R.from_matrix(so3exp(SE3[5:8]).T).as_quat()
    print(q)
    print('==============================')
    for i in q:
        result[-1].append(i)
    
    
result = np.array(result)
np.savetxt('./t.txt', result)
# print(result.shape)