import matplotlib.pyplot as plt
import numpy as np
from rich import print
from glob import glob
import os

def plot_loss(file):
    robustfn_damping, _ = os.path.splitext(file)
    robustfn_damping = robustfn_damping[2:]
    robustfn, eta_damping = robustfn_damping.split('_')
    
    with open(file, 'r') as f :
        lines = f.readlines()
    
    loss = []
    for line in lines:
        loss.append(float(line))
    
    plt.plot(loss)
    plt.xlabel("Iteration")
    plt.ylabel("are")
    plt.title("robust function  : {}\neta damping : {}".format(robustfn, eta_damping))
    plt.savefig("./figure/{}.png".format(robustfn_damping))
    plt.clf()
    print("{}\t max : {:.5}, min : {:.4}".format(robustfn_damping, max(loss), min(loss)))

file_list = glob("./*.txt")
for file in file_list:
    plot_loss(file)
