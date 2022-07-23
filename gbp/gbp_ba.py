"""
    Defines child classes of GBP parent classes for Bundle Adjustment.
    Also defines the function to create the factor graph.
"""

import numpy as np
from gbp import gbp, gbp_ba
from gbp.factors import reprojection
from utils import read_balfile
from utils import read_orb_input
from utils import lie_algebra



class BAFactorGraph(gbp.FactorGraph):
    def __init__(self, **kwargs):
        gbp.FactorGraph.__init__(self, nonlinear_factors=True, **kwargs)

        self.cam_nodes = []
        self.lmk_nodes = []
        self.var_nodes = self.cam_nodes + self.lmk_nodes

    def generate_priors_var(self, weaker_factor=100):
        """
            Sets automatically the std of the priors such that standard deviations
            of prior factors are a factor of weaker_factor
            weaker than the standard deviations of the adjacent factors.
            NB. Jacobian of measurement function effectively sets the scale of the factors.
        """
        for var_node in self.cam_nodes + self.lmk_nodes:
            max_factor_lam = 0.
            # for factor in var_node.adj_factors:
                # if isinstance(factor, gbp_ba.ReprojectionFactor):
                    # max_factor_lam = max(max_factor_lam, np.max(factor.factor.lam))
            # lam_prior = np.eye(var_node.dofs) * max_factor_lam / (weaker_factor ** 2)
            lam_prior = np.eye(var_node.dofs)
            var_node.prior.lam = lam_prior
            var_node.prior.eta = lam_prior @ var_node.mu

    def weaken_priors(self, weakening_factor):
        """
            Increases the variance of the priors by the specified factor.
        """
        for var_node in self.var_nodes:
            var_node.prior.eta *= weakening_factor
            var_node.prior.lam *= weakening_factor

    def set_priors_var(self, priors):
        """
            Means of prior have already been set when graph was initialised. Here we set the variance of the prior factors.
            priors: list of length number of variable nodes where each element is the covariance matrix of the prior
                    distribution for that variable node.
        """
        for v, var_node in enumerate(self.var_nodes):
            var_node.prior.lam = np.linalg.inv(priors[v])
            var_node.prior.eta = var_node.prior.lam @ var_node.mu

    def compute_residuals(self):
        residuals = []
        for factor in self.factors:
            if isinstance(factor, ReprojectionFactor):
                residuals += list(factor.compute_residual())
        return residuals

    def are(self):
        """
            Computes the Average Reprojection Error across the whole graph.
        """
        are = 0
        local_edge = 0
        for factor in self.factors:
            if isinstance(factor, ReprojectionFactor):
                d, scale_factor = factor.compute_residual()
                are += np.linalg.norm(d*scale_factor)
                if scale_factor != 0:
                    local_edge += 1
        return are / local_edge
        # return are / len(self.factors)

    def get_factor_loss(self):

        loss_list = []
        
        for factor in self.factors:
            # pred, measurement, reci_scale_factor = factor.compute_reprojection_terms()

            # ex = loss_fn(pred[0] * reci_scale_factor, measurement[0] * reci_scale_factor)
            # ey = loss_fn(pred[1] * reci_scale_factor, measurement[1] * reci_scale_factor)
            # loss_list.append(ex + ey)
            d, scale_factor = factor.compute_residual()
            loss_list.append(np.linalg.norm(d*scale_factor))

        return loss_list


class LandmarkVariableNode(gbp.VariableNode):
    def __init__(self, variable_id, dofs, l_id=None):
        gbp.VariableNode.__init__(self, variable_id, dofs)
        self.l_id = l_id


class FrameVariableNode(gbp.VariableNode):
    def __init__(self, variable_id, dofs, c_id=None):
        gbp.VariableNode.__init__(self, variable_id, dofs)
        self.c_id = c_id


class ReprojectionFactor(gbp.Factor):
    def __init__(self, factor_id, adj_var_nodes, measurement, scale_factor, gauss_noise_std, loss, Nstds, K):

        gbp.Factor.__init__(self, factor_id, adj_var_nodes, measurement, scale_factor, gauss_noise_std,
                            reprojection.meas_fn, reprojection.jac_fn, loss, Nstds, K)

    def reprojection_err(self):
        """
            Returns the reprojection error at the factor in pixels.
        """
        return np.linalg.norm(self.compute_residual())


def create_ba_graph(filename, configs):
    """
        Create graph object from bal style file.
    """
    # n_keyframes, n_points, n_edges, cam_means, lmk_means, measurements, measurements_camIDs, \
    #         measurements_lIDs, K = read_balfile.read_balfile(bal_file)

    # kf_array, lm_array, mea_array, intrinsic = read_orb_input.read_orb_input(path)
    
    kf_array, lm_array, mea_array, K = read_orb_input.read_orb_input(filename)
    # kf_array = np.reshape(keyframe_pose, [-1, 7])
    # lm_array = np.reshape(landmark_pose, [-1, 4])
    # mea_array = np.reshape(measurement, [-1, 5])

    # K = np.zeros([3, 3])
    # K[0, 0], K[1, 1], K[0, 2], K[1, 2] = [x for x in intrinsic]
    # K[2, 2] = 1.

    graph = BAFactorGraph( eta_damping=configs['eta_damping'],
                          beta=configs['beta'],
                          num_undamped_iters=configs['num_undamped_iters'],
                          min_linear_iters=configs['min_linear_iters'])

    variable_count = 0
    factor_id = 0
    n_edges = 0

    # Initialize variable nodes for frames with prior
    kf_id_list, lm_id_list = [], []

    for m, kf in enumerate(kf_array):
        variable_id = int(kf[0])
        kf_id_list.append(variable_id)
        # m means the row of the original array
        new_cam_node = FrameVariableNode(variable_id, 6, m)

        # Rcw = lie_algebra.so3exp(kf[5:])
        tcw = kf[2:5]
        # mu lies in camera coordinate
        new_cam_node.mu = list(kf[2:])
        for i in range(3):
            new_cam_node.mu[i] = tcw[i].T        
        # add timestamp
        new_cam_node.timestamp = kf[1]
        graph.cam_nodes.append(new_cam_node) 
        variable_count += 1

    # Initialize variable nodes for landmarks with prior
    for l, lm in enumerate(lm_array):
        variable_id = int(lm[0])
        lm_id_list.append(variable_id)
        new_lmk_node = LandmarkVariableNode(variable_id, 3, l)
        new_lmk_node.mu = [i for i in lm[1:]]
        
        # add random noise to landmark
        # new_lmk_node.mu += np.random.rand(3)*0.01
        
        graph.lmk_nodes.append(new_lmk_node)
        variable_count += 1

    for f, mea in enumerate(mea_array):
        # mea[0] is keyframe id
        # mea[1] is landmark id
        for cam in graph.cam_nodes:
            if cam.variableID == int(mea[0]):
                cam_node = cam
                break
        for lmk in graph.lmk_nodes:
            if lmk.variableID == int(mea[1]):
                lmk_node = lmk 
                break
        measurement = [mea[2], mea[3]]
        scale_factor = mea[4]

        new_factor = ReprojectionFactor(factor_id, [cam_node, lmk_node], measurement, scale_factor ,
                                                       configs['gauss_noise_std'], configs['loss'], configs['Nstds'], K)

        linpoint = np.concatenate((cam_node.mu, lmk_node.mu))
        new_factor.compute_factor(linpoint)
        cam_node.adj_factors.append(new_factor)
        lmk_node.adj_factors.append(new_factor)

        graph.factors.append(new_factor)
        factor_id += 1
        n_edges += 2



    graph.n_factor_nodes = factor_id
    graph.n_var_nodes = variable_count
    graph.var_nodes = graph.cam_nodes + graph.lmk_nodes
    graph.n_edges = n_edges

    return graph


