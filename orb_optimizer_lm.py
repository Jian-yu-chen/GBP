#!/usr/bin/env python
"""
Bundle Adjustment using GBP.
"""

import argparse
from time import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from klampt import PointCloud

# from orb_slam2.gbp_for_ORB.gbp import gbp_ba, gbp
from gbp import gbp_ba
from lm.ba import BundleAdjustment as BALM
from vis import visualize

parser = argparse.ArgumentParser()
parser.add_argument("--bal_file", required=False, help="BAL style file with BA data")

parser.add_argument(
    "--n_iters", type=int, default=15, help="Number of iterations of GBP"
)

parser.add_argument(
    "--gauss_noise_std",
    type=int,
    default=0.5,
    help="Standard deviation of Gaussian noise of measurement model.",
)
parser.add_argument(
    "--loss",
    default="huber",
    help="Loss function: None (squared error), huber or constant.",
)
parser.add_argument(
    "--Nstds",
    type=float,
    default=3.0,
    help="If loss is not None, number of stds at which point the "
    "loss transitions to linear or constant.",
)
parser.add_argument(
    "--beta",
    type=float,
    default=0.01,
    help="Threshold for the change in the mean of adjacent beliefs for "
    "relinearisation at a factor.",
)
parser.add_argument(
    "--num_undamped_iters",
    type=int,
    default=6,
    help="Number of undamped iterations at a factor node after relinearisation.",
)
parser.add_argument(
    "--min_linear_iters",
    type=int,
    default=8,
    help="Minimum number of iterations between consecutive relinearisations of a factor.",
)
parser.add_argument(
    "--eta_damping",
    type=float,
    default=0.4,
    help="Max damping of information vector of messages.",
)

parser.add_argument(
    "--prior_std_weaker_factor",
    type=float,
    default=50.0,
    help="Ratio of std of information matrix at measurement factors / "
    "std of information matrix at prior factors.",
)

parser.add_argument(
    "--float_implementation",
    action="store_true",
    default=False,
    help="Float implementation, so start with strong priors that are weakened",
)
parser.add_argument(
    "--final_prior_std_weaker_factor",
    type=float,
    default=100.0,
    help="Ratio of information at measurement factors / information at prior factors "
    "after the priors are weakened (for floats implementation).",
)
parser.add_argument(
    "--num_weakening_steps",
    type=int,
    default=5,
    help="Number of steps over which the priors are weakened (for floats implementation)",
)

args = parser.parse_args()

configs = dict(
    {
        "gauss_noise_std": args.gauss_noise_std,
        "loss": args.loss,
        "Nstds": args.Nstds,
        "beta": args.beta,
        "num_undamped_iters": args.num_undamped_iters,
        "min_linear_iters": args.min_linear_iters,
        "eta_damping": args.eta_damping,
        "prior_std_weaker_factor": args.prior_std_weaker_factor,
    }
)

if args.float_implementation:
    configs["final_prior_std_weaker_factor"] = args.final_prior_std_weaker_factor
    configs["num_weakening_steps"] = args.num_weakening_steps
    weakening_factor = (
        np.log10(args.final_prior_std_weaker_factor) / args.num_weakening_steps
    )


def gbp_optimizer(filename):
    graph = gbp_ba.create_ba_graph(filename, configs)
    graph.generate_priors_var(weaker_factor=args.prior_std_weaker_factor)
    # graph.update_all_beliefs()
    graph.update_all_beliefs_prior()

    # visualize
    vis = visualize.visualize(graph)

    min_are = 10000
    init_are = graph.are()

    solver = BALM.from_factor_graph(graph)
    for i in range(args.n_iters):
        solver.optimize(1)
        solver.synchronize_results_with_factor_graph(graph)
        print("Iteration : {:d}, loss = {:3f}".format(i + 1, graph.are()))
        vis.update(graph)

    # vis.run()
    final_are = graph.are()
    print("init_loss : {:3f}      final_loss : {:3f} ".format(init_are, final_are))
    min_are = final_are

    # return optimized_kf, optimized_lm, min_are
    return solver


if __name__ == "__main__":
    # gbp_optimizer('./data/kitti_07_without_GBA.txt')
    # gbp_optimizer('./data/small_test.txt')
    solver = gbp_optimizer("./data/demo/Dodecahedron.txt")
