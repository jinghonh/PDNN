import torch
from config import get_config
from train import train_model
from test import test_model
from utils import *
from problems import *

if __name__ == "__main__":
    config = get_config()
    problems = Problem(40)
    f_x, d_x, A, b, w, w_test, x_bar = problems.problem1()
    train_model(config, f_x, A=A, b=b, w=w, x_bar=x_bar, w_test=w_test)
    total_primal, total_dual, primal_net, dual_net = test_model(config, f_x, A=A, b=b, w_test=w_test, x_bar=x_bar)
    total_primal = torch.tensor(total_primal)
    total_dual = torch.tensor(total_dual)
    d_x(total_dual, w_test)
    plot_primal_net_frontier(f_x, total_primal)
    input_dim = f_x(x_bar).shape[1]
    primal_output_dim = A.shape[1]
    dual_output_dim = A.shape[0]
    # draw_neural_net(primal_net, input_dim, config['primal_hidden_dim'], primal_output_dim)
    draw_neural_net_weights_heatmap(primal_net)
    save_weights_to_excel(primal_net, 'primal_net.xlsx')


