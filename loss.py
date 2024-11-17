import torch
import torch.nn as nn
import numpy as np

def compute_jacobian(func, x):
    """
    :param func:
    :param x:
    :return jacobian mat:
    """

    jacobian = torch.autograd.functional.jacobian(func, x, create_graph=True, vectorize=True)
    return jacobian


def kkt_loss_function(x, lambda_, weight, problem_config):
    """
    论文中描述的基于KKT条件的损失函数的PyTorch实现。

    参数：
        - x (torch.Tensor): 原始神经网络的输出，大小为 [batch_size, num_variables]。
        - lambda_ (torch.Tensor): 对偶神经网络的输出，大小为 [batch_size, num_constraints]。
        - weight (torch.Tensor): 标量化权重向量，大小为 [batch_size, num_objectives]。
        - f_x (function): 接收 x 并返回目标值的函数，大小为 [batch_size, num_objectives]。
        - g_x (function): 接收 x 并返回约束值的函数，大小为 [batch_size, num_constraints]。
    返回：
        torch.Tensor: 计算的KKT损失值。
    """

    f_x = problem_config['f_x']
    g_x = problem_config['g_x']

    # 计算jacobian矩阵
    J_f = compute_jacobian(f_x, x).sum(dim=-2)
    J_g = compute_jacobian(g_x, x).sum(dim=-2)

    # 一阶条件 (4)：J_f(x)^T * weight + J_g(x)^T * lambda = 0
    kkt_first_order = (torch.bmm(J_f.transpose(1, 2), weight.unsqueeze(-1)).squeeze(-1) +
                       torch.bmm(J_g.transpose(1, 2), lambda_.unsqueeze(-1)).squeeze(-1))
    kkt_first_order_loss = torch.norm(kkt_first_order, p=2, dim=1) ** 2

    # 互补松弛条件 (7)：lambda_i * g_i(x) = 0，且 g_i(x) <= 0
    g_values = g_x(x)  # 约束值
    slackness = lambda_ * g_values
    complementary_slackness_loss = torch.norm(slackness, p=2, dim=1) ** 2
    g = torch.norm(g_values, p=2, dim=1)
    l = torch.norm(lambda_, p=2, dim=1)
    eta = torch.mean(kkt_first_order_loss) / torch.mean(complementary_slackness_loss)
    # eta = 0.1
    # 一阶条件损失和互补松弛损失的加权和
    total_loss = eta * kkt_first_order_loss + complementary_slackness_loss
    # mean_loss = torch.mean(total_loss / l)
    mean_loss = torch.mean(total_loss)

    return mean_loss

# np.array(torch.stack((kkt_first_order_loss,complementary_slackness_loss),dim=1).cpu().detach())
# fx = f_x(x)
# fx = np.array(fx.cpu().detach())
