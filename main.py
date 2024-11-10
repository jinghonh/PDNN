import torch
from config import get_config
from train import train_model
from test import test_model
from utils import *
from problems import *

if __name__ == "__main__":
    config = get_config()
    problems = Problem(20)
    f_x, d_x, A, b, w, w_test, x_bar = problems.problem1()
    train_model(config, f_x, A=A, b=b, w=w, x_bar=x_bar, w_test=w_test)
    total_primal, total_dual = test_model(config, f_x, A=A, b=b, w_test=w_test, x_bar=x_bar)
    total_primal = torch.tensor(total_primal)
    plot_primal_net_frontier(f_x, total_primal)
