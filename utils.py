import matplotlib.pyplot as plt
import torch


def plot_primal_net_frontier(f,x):
    fx = f(x)
    fx = fx.detach().numpy()
    plt.scatter(fx[:, 0], fx[:, 1])
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("Approximated Weakly Efficient Frontier")
    plt.show()


def plot_dual_net_frontier(primal_net, test_loader, f):
    pass
