import matplotlib.pyplot as plt
import torch
def plot_frontier(primal_net, test_loader):
    for w, _ in test_loader:
        with torch.no_grad():
            predictions = primal_net(w)
            plt.scatter(predictions[:, 0], predictions[:, 1])
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("Approximated Weakly Efficient Frontier")
    plt.show()
