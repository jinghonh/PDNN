import torch
from config import get_config
from train import train_model
from test import test_model
from utils import *
from problems import *
import warnings


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    problems = Problem(5, 100, 200)
    problem_config = problems.problem2()
    f_x = problem_config['f_x']
    # d_x = problem_config['d_x']
    w_test = problem_config['w_test']
    x_bar = problem_config['x_bar']
    # 训练模型
    train_model(config, problem_config)
    # 测试模型
    total_primal, total_dual, primal_net, dual_net = test_model(config, problem_config)
    # 获取模型输出
    total_primal = torch.tensor(total_primal)
    total_dual = torch.tensor(total_dual)
    # 绘制primal模型输出
    plot_primal_net_frontier(f_x, total_primal, problem_config['f_true'])
    # 绘制primal模型热力图
    input_dim = f_x(x_bar).shape[1]
    primal_output_dim = problem_config['primal_output_dim']
    dual_output_dim = problem_config['dual_output_dim']
    # draw_neural_net(primal_net, input_dim, config['primal_hidden_dim'], primal_output_dim)
    # draw_neural_net_weights_heatmap(primal_net)
    # save_weights_to_excel(primal_net, 'primal_net.xlsx')
