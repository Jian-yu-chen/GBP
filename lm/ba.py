from dataclasses import dataclass

import g2o
import numpy as np
from gbp.gbp_ba import BAFactorGraph
from utils.lie_algebra import so3exp, so3log


@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float


class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)
        self.initialized = False

    def optimize(self, max_iterations=10):
        if not self.initialized:
            super().initialize_optimization()
            self.initialized = True
        super().optimize(max_iterations)

    def add_pose(self, pose_id, pose, cam, fixed=False):
        sbacam = g2o.SBACam(pose)
        sbacam.set_cam(cam.fx, cam.fy, cam.cx, cam.cy, cam.baseline)

        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)  # internal id
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(
        self,
        point_id,
        pose_id,
        measurement,
        information=np.identity(2),
        robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991)),
    ):  # 95% CI

        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(measurement)  # projection
        edge.set_information(information)

        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, pose_id):
        return self.vertex(pose_id * 2).estimate()

    def get_point(self, point_id):
        return self.vertex(point_id * 2 + 1).estimate()

    @classmethod
    def from_factor_graph(cls, graph: BAFactorGraph):
        ba = cls()

        # Configure camera parameter
        K = graph.factors[0].args[0]  # K
        cam = Camera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0)

        # Add camera poses
        for idx, cam_node in enumerate(graph.cam_nodes):
            cam_node_pose = np.eye(4)
            # Ref: gbp.factors.reprojection.meas_fn
            cam_node_pose[:3, 3] = cam_node.mu[:3]
            cam_node_pose[:3, :3] = so3exp(cam_node.mu[3:6])
            pose = g2o.SE3Quat(cam_node_pose[:3, :3], cam_node_pose[:3, 3])

            ba.add_pose(cam_node.variableID, pose, cam, fixed=(idx == 0))

        # Add landmark positions
        for lmk_node in graph.lmk_nodes:
            ba.add_point(
                lmk_node.variableID, lmk_node.mu, fixed=False, marginalized=True
            )

        # Add measurement factors
        for factor in graph.factors:
            cam_index = factor.adj_var_nodes[0].variableID
            lmk_index = factor.adj_var_nodes[1].variableID
            ba.add_edge(
                pose_id=cam_index, point_id=lmk_index, measurement=factor.measurement
            )

        return ba

    def synchronize_results_with_factor_graph(self, graph: BAFactorGraph):
        # Update camera poses
        for cam_node in graph.cam_nodes:
            pose = self.get_pose(cam_node.variableID)
            cam_node.mu[:3] = pose.position()
            cam_node.mu[3:6] = so3log(pose.orientation().matrix())

        # Update landmark positions
        for lmk_node in graph.lmk_nodes:
            lmk_node.mu = self.get_point(lmk_node.variableID)

        # Update belief
        # FIXME: Check if this conversion is correct
        for factor in graph.factors:
            for idx, node in enumerate(factor.adj_var_nodes):
                factor.adj_beliefs[idx].lam = np.eye(node.dofs)
                factor.adj_beliefs[idx].eta = node.mu
