import sys
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont

sys.path.append('.')
import utils.lie_algebra as lie



class visualize:
    def __init__(self, graph) -> None:
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        # self.vis.show_settings = True
        self.graph = graph
        graph.cam_nodes
        graph.lmk_nodes
        
        self.lmk_list = [self.transfer(graph.lmk_nodes)]
        self.cam_list = [self.transfer(graph.cam_nodes)]
        
        # self.camera_list = []
        # self.pcd_list = []
        
        self.WINDOW_WIDTH = 1024
        self.WINDOW_HEIGHT = 768
        
        self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(1024, 768, 707.091, 707.091, 601.887, 183.11)
        
        self.K = np.array([[707.091, 0, 601.887],
                           [0, 707.091, 183.11],
                           [0, 0, 1]])
        
        self.vis.create_window("GBP optimizer", 1024, 768)
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])
        
        # self.vis.register_key_action_callback(ord("p"), )
        
        self.show()
        view_control = self.vis.get_view_control() 
        self.pinhole_camera_parameters = view_control.convert_to_pinhole_camera_parameters()
        # view_control.set_lookat((1,0,0))
        # view_control.set_zoom(1)
        
    def transfer(self, node_list):
        mu = []
        for node in node_list:
            mu.append(node.mu)
        return mu
    
    def add_text_panel(self) -> None:
        img = Image.new('RGB', (self.WINDOW_WIDTH, self.WINDOW_HEIGHT), color = (255,255,255))
        font = ImageFont.truetype("arial.ttf", 30)
        
        d = ImageDraw.Draw(img)
        iteration_num = len(self.lmk_list)
        
        are = self.graph.are()
        if iteration_num == 1 :
            information = "This Is Initial Graph"
        else:
            information = "Iteration : {:2d}\nAre : {:3f}".format(iteration_num-1, are)
        d.text((75, 75), information, font=font, fill=(0,0,0))
        
        img.save('./vis/pil_text.png')
        
        im = o3d.io.read_image('./vis/pil_text.png') 
        self.vis.add_geometry(im)
    
    def show_groundtruth(self) -> None:
        point = [[1, 1, 1],
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
                 [-1.618, -0.6180, 0]
                 ]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point)
        pcd.paint_uniform_color([0, 1, 0])
        self.vis.add_geometry(pcd)
        
    
    def delta_translation(self) -> o3d.geometry.LineSet:
        
        point_now = np.array(self.lmk_list[-1])
        point_past = np.array(self.lmk_list[-2])
        
        point = np.concatenate((point_now, point_past))

        point_len = len(point_now)
        
        order = [[i, i+point_len] for i in range(len(point_now))]

        delta = o3d.geometry.LineSet(points = o3d.utility.Vector3dVector(point), lines = o3d.utility.Vector2iVector(order))

        return delta
    
    def show(self):
        
        self.vis.clear_geometries()
        # self.add_text_panel()
        # self.vis.add_geometry(self.axis)
        self.show_groundtruth()
        
        point = self.lmk_list[-1]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point)
        pcd.paint_uniform_color([1, 0, 0])
        self.vis.add_geometry(pcd)
        
        # render options
        self.vis.get_render_option().point_size = 6
        
        if len(self.lmk_list) > 1:
            delta_line_set = self.delta_translation()
            self.vis.add_geometry(delta_line_set)
        
        for camera in self.cam_list[-1]:
            T = np.identity(4)
            R_cw = lie.so3exp(camera[3:])
            t_cw = np.array(camera[:3]).T
            T[:3, :3] = R_cw
            T[:3, 3] = t_cw
            
            camera_vis = o3d.geometry.LineSet().create_camera_visualization(1024, 768, self.K, T, 0.5)
            self.vis.add_geometry(camera_vis)
        
        if len(self.lmk_list) > 1:
            view_control = self.vis.get_view_control()
            view_control.convert_from_pinhole_camera_parameters(self.pinhole_camera_parameters)
        
        self.vis.poll_events()
        self.vis.update_renderer() 
        
        # if len(self.lmk_list) == 10:
        #     time.sleep(20)
        time.sleep(1)
    
    
    def update(self, graph):
        self.lmk_list.append(self.transfer(graph.lmk_nodes))
        self.cam_list.append(self.transfer(graph.cam_nodes))
        self.show()