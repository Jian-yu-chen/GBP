#!/usr/bin/env python
"""
Bundle Adjustment using GBP.
"""

from klampt import PointCloud
import numpy as np
import argparse
# from orb_slam2.gbp_for_ORB.gbp import gbp_ba, gbp
from gbp import gbp_ba
import matplotlib.pyplot as plt
from time import time
import open3d as o3d
import open3d.visualization.gui as gui
from vis import visualize


parser = argparse.ArgumentParser()
parser.add_argument("--bal_file", required=False,
                    help="BAL style file with BA data")

parser.add_argument("--n_iters", type=int, default=15,
                    help="Number of iterations of GBP")

parser.add_argument("--gauss_noise_std", type=int, default=0.5,
                    help="Standard deviation of Gaussian noise of measurement model.")
parser.add_argument("--loss", default='huber',
                    help="Loss function: None (squared error), huber or constant.")
parser.add_argument("--Nstds", type=float, default=3.,
                    help="If loss is not None, number of stds at which point the "
                         "loss transitions to linear or constant.")
parser.add_argument("--beta", type=float, default=0.01,
                    help="Threshold for the change in the mean of adjacent beliefs for "
                         "relinearisation at a factor.")
parser.add_argument("--num_undamped_iters", type=int, default=6,
                    help="Number of undamped iterations at a factor node after relinearisation.")
parser.add_argument("--min_linear_iters", type=int, default=8,
                    help="Minimum number of iterations between consecutive relinearisations of a factor.")
parser.add_argument("--eta_damping", type=float, default=0.4,
                    help="Max damping of information vector of messages.")

parser.add_argument("--prior_std_weaker_factor", type=float, default=50.,
                    help="Ratio of std of information matrix at measurement factors / "
                         "std of information matrix at prior factors.")

parser.add_argument("--float_implementation", action='store_true', default=False,
                    help="Float implementation, so start with strong priors that are weakened")
parser.add_argument("--final_prior_std_weaker_factor", type=float, default=100.,
                    help="Ratio of information at measurement factors / information at prior factors "
                         "after the priors are weakened (for floats implementation).")
parser.add_argument("--num_weakening_steps", type=int, default=5,
                    help="Number of steps over which the priors are weakened (for floats implementation)")

args = parser.parse_args()

configs = dict({
    'gauss_noise_std': args.gauss_noise_std,
    'loss': args.loss,
    'Nstds': args.Nstds,
    'beta': args.beta,
    'num_undamped_iters': args.num_undamped_iters,
    'min_linear_iters': args.min_linear_iters,
    'eta_damping': args.eta_damping,
    'prior_std_weaker_factor': args.prior_std_weaker_factor,
           })

if args.float_implementation:
    configs['final_prior_std_weaker_factor'] = args.final_prior_std_weaker_factor
    configs['num_weakening_steps'] = args.num_weakening_steps
    weakening_factor = np.log10(args.final_prior_std_weaker_factor) / args.num_weakening_steps




def gbp_optimizer(filename):
    graph = gbp_ba.create_ba_graph(filename, configs)
    graph.generate_priors_var(weaker_factor=args.prior_std_weaker_factor)
    # graph.update_all_beliefs()
    graph.update_all_beliefs_prior()
    
    # visualize
    vis = visualize.visualize(graph)
    
    min_are = 10000
    
    init_are = graph.are()
    
    with open('./message.txt', 'w') as f :
        f.writelines('{:.4f} 0\n'.format(init_are))
    print("Iteration : 0, loss = {:3f}".format(init_are))
    optimized_kf, optimized_lm = graph.output_to_orb('./trajectory/iter_0.txt')
    # n_iters = args.n_iters + int(((len(measurement)/5-1000) // 1000)*30)
    
    with open('./para_analysis/{}_{}.txt'.format(args.loss, args.eta_damping), 'w') as f:
        f.writelines("{}\n".format(init_are))
        
    for i in range(args.n_iters):
        # print("Iteration : {:d}".format(i+1))
    # To copy weakening of strong priors as must be done on IPU with float
        if args.float_implementation and (i+1) % 2 == 0 and (i < args.num_weakening_steps * 2):
            graph.weaken_priors(weakening_factor)

    # At the start, allow a larger number of iterations before linearising
        if i == 3 or i == 8:
            for factor in graph.factors:
                factor.iters_since_relin = 1
    
        n_factor_relins = 0
        for factor in graph.factors:
            if factor.iters_since_relin == 0:
                n_factor_relins += 1
        # print('start iteration ...')
        graph.synchronous_iteration(robustify=True, local_relin=True)
        # print('end iteration ...')
        
        with open('./message.txt', 'a') as f :
            f.writelines('{:.4f} {}\n'.format(graph.are(), n_factor_relins))
        if i==args.n_iters-1 :
            optimized_kf, optimized_lm = graph.output_to_orb('./trajectory/iter_{}.txt'.format(i+1))
        if graph.are() < min_are:
        #     best_optimized_kf, best_optimized_lm = optimized_kf, optimized_lm
            optimized_kf, optimized_lm = graph.output_to_orb('./trajectory/best.txt')
            min_are = graph.are()
        
        with open('./para_analysis/{}_{}.txt'.format(args.loss, args.eta_damping), 'a') as f:
            f.writelines("{}\n".format(graph.are()))
        
        print("Iteration : {:d}, loss = {:3f}, relinearise point : {}".format(i+1, graph.are(), n_factor_relins))
        vis.update(graph)
    # vis.run()
    final_are = graph.are()
    print("init_loss : {:3f}      final_loss : {:3f} ".format(init_are, final_are))
    min_are = final_are
    
    
    return optimized_kf, optimized_lm, min_are


    

if __name__ =='__main__':
    # gbp_optimizer('./data/kitti_07_without_GBA.txt')
    # gbp_optimizer('./data/small_test.txt')
    gbp_optimizer('./data/demo/Dodecahedron.txt')