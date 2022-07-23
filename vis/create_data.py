import sys
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import time

sys.path.append('.')
from utils.lie_algebra import so3log
from gbp.factors.reprojection import meas_fn
import utils.lie_algebra as lie
from scipy.spatial.transform import Rotation as R


class create_data:
    def __init__(self, landmarks):
        self.vis = o3d.visualization.Visualizer()
        
        self.point_his = []
        self.point_his.append(landmarks)
        # self.point_his.append(self.transfer(landmarks))
        
        # self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(1024, 768, 707.091, 707.091, 601.887, 183.11)
        self.view_control = self.vis.get_view_control()
        # .convert_from_pinhole_camera_parameters(self.camera)
        
        self.run()
        
    def get_parameter(self, view_control):
        camera_parameter = view_control.convert_to_pinhole_camera_parameters()
        intrinsic = camera_parameter.intrinsic.intrinsic_matrix
        # print(camera_parameter.intrinsic.height)
        extrinsic = camera_parameter.extrinsic
        return intrinsic, extrinsic
        
    
    
    def run(self):
        self.vis.create_window("test", 1024, 768)
        
        # create 3d-axis
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])
        
        
        point = self.point_his[0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point)
        
        pcd.paint_uniform_color([1, 0, 0])
        self.vis.add_geometry(pcd)
        
        # set the point size of the graph(default:5)
        self.vis.get_render_option().point_size = 12
        
        
        # self.vis.get_view_control().rotate(0.05, -0.3)
        # self.vis.get_view_control().translate(-0.15, -0.2)
        # self.vis.get_view_control().scale(0.)
        
        r_cw = [[], []]
        t_cw = [[], []]
        r_wc = [[], []]
        t_wc = [[], []]
        
        # self.camera_parameters = self.vis.get_view_control().convert_to_pinhole_camera_parameters()   
        
        K, T = self.get_parameter(self.vis.get_view_control())
        
        
        # q0_wc = np.array([-0.00661014, -0.367816, 0.00726249, 0.929847])
        q0_cw = np.array([0, 0, 0, 1])
        r_cw[0] = R.from_quat(q0_cw).as_matrix()
        r_wc[0] = r_cw[0].T
        t_cw[0] = np.array([0, 0, 3.92344149])
        t_wc[0] = -r_wc[0] @ t_cw[0].T
        T0 = np.identity(4)
        T0[:3, :3] = r_cw[0]
        T0[:3, 3] = t_cw[0].T
        
        # print(T0)
        camera1 = o3d.geometry.LineSet().create_camera_visualization(1024, 768, K, T0, 1)
        
        # camera id 3
        
        q1_cw = np.array([0.00282088, 0.125113, 0.0132275, 0.964122])
        r_cw[1] = R.from_quat(q1_cw).as_matrix()
        r_wc[1] = r_cw[1].T
        # t_wc[1] = np.array([0, 2, 2.92344149])
        t_cw[1] = np.array([0, 0, 4.3])
        t_wc[1] = -r_wc[1] @ t_cw[1].T
        # print(t)
        T1 = np.identity(4)
        T1[:3, :3] = r_cw[1]
        T1[:3, 3] = t_cw[1].T
        
        # print(T0, '\n', T1)
        
        
        # K2, T2 = self.get_parameter(self.vis.get_view_control())
        
        
        # r[1] = lie.so3log(T2[0:3, 0:3])
        # print(r)
        # q2 = R.from_matrix(T2[0:3, 0:3].T.copy()).as_quat()
        # t[1] = T2[0:3, 3]
        # print([float(i) for i in t[1]], [float(i) for i in q2])
        
        # camera2 = camera2.transform(T2)
        camera2 = o3d.geometry.LineSet().create_camera_visualization(1024, 768, K, T1, 1)
        
        self.vis.add_geometry(axis)
        self.vis.add_geometry(camera1)
        self.vis.add_geometry(camera2)
         
        self.vis.poll_events()
        self.vis.update_renderer()
        
        # print("camera pose")
        for cam_id in [0, 1]:
            q_wc = R.from_matrix(r_wc[cam_id]).as_quat()
            # t_wc = -r_wc[cam_id]@(t_cw[cam_id].T)
            # print(t_wc)
            print(np.concatenate((t_wc[cam_id].T, q_wc)))
        # print("========================================")
        
        for cam_id in [0, 1]:
            for lam_id, coordinate in enumerate(point):
                measurement = [cam_id, lam_id]
                inp = np.concatenate((t_cw[cam_id], lie.so3log(r_cw[cam_id]), coordinate))
                mx, my = meas_fn(inp, K)
                measurement.append(mx)
                measurement.append(my)
                measurement.append(1)
                string = ''
                for i in measurement:
                    string += str(i)+' '
                string = string[:-1]
                # print(string)
                if mx > 1024 or my > 768 or mx < 0 or my < 0 :
                    print("Error")
         
        self.vis.run()      
            
    
    
if __name__ == '__main__':
    landmarks = np.array([[1, 1, 1],
                 [1, 1, -1],
                 [1, -1, 1],
                 [1, -1, -1],
                 [-1, -1, -1],
                 [-1, -1, 1],
                 [-1, 1, -1],
                 [-1, 1, 1],
                 [0, 1.618, 0.6180],
                 [0, 1.618, -0.6180],
                 [0, -1.618, 0.6180],
                 [0, -1.618, -0.6180],
                 [0.6180, 0, 1.618],
                 [0.6180, 0, -1.618],
                 [-0.6180, 0, 1.618],
                 [-0.6180, 0, -1.618],
                 [1.618, 0.6180, 0],
                 [1.618, -0.6180, 0],
                 [-1.618, 0.6180, 0],
                 [-1.618, -0.6180, 0]])
    
    # landmarks = np.array(landmarks)
    vis = create_data(landmarks)
