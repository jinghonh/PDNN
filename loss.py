import torch
import torch.nn as nn


def kkt_loss_function(x, lambda_, w, f_x, A, b, eta=10):
    """
    论文中描述的基于KKT条件的损失函数的PyTorch实现。

    参数：
        x (torch.Tensor): 原始神经网络的输出，大小为 [batch_size, num_variables]。
        lambda_ (torch.Tensor): 对偶神经网络的输出，大小为 [batch_size, num_constraints]。
        w (torch.Tensor): 标量化权重向量，大小为 [batch_size, num_objectives]。
        f_x (function): 接收 x 并返回目标值的函数，大小为 [batch_size, num_objectives]。
        A (torch.Tensor): 表示线性约束的矩阵，大小为 [num_constraints, num_variables]。
        b (torch.Tensor): 表示线性约束边界的向量，大小为 [num_constraints]。
        eta (float, optional): 互补松弛部分损失的权重因子。默认值为 10。

    返回：
        torch.Tensor: 计算的KKT损失值。
    """
    batch_size, num_variables = x.size()
    num_constraints = A.size(0)

    # 使用autograd计算f(x)的雅可比矩阵
    x.requires_grad_(True)
    f_values = f_x(x)  # [batch_size, num_objectives]
    J_f = []
    for i in range(f_values.size(1)):
        grad_f = torch.autograd.grad(f_values[:, i].sum(), x, create_graph=True)[0]
        J_f.append(grad_f.unsqueeze(1))
    J_f = torch.cat(J_f, dim=1)  # [batch_size, num_objectives, num_variables]

    # 计算g(x) = Ax - b的雅可比矩阵
    J_g = A.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_constraints, num_variables]

    # 一阶条件 (4)：J_f(x)^T * w + J_g(x)^T * lambda = 0
    # 调整w的形状以确保矩阵乘法的兼容性
    w = w.unsqueeze(-1)  # [batch_size, num_objectives, 1]
    kkt_first_order = torch.bmm(J_f.transpose(1, 2), w).squeeze(-1) + torch.bmm(J_g.transpose(1, 2),
                                                                                lambda_.unsqueeze(-1)).squeeze(-1)
    kkt_first_order_loss = torch.norm(kkt_first_order, p=2, dim=1) ** 2

    # 互补松弛条件 (7)：lambda_i * g_i(x) = 0，且 g_i(x) <= 0
    g_x = torch.bmm(J_g, x.unsqueeze(-1)).squeeze(-1) - b  # 约束值
    slackness = lambda_ * g_x
    complementary_slackness_loss = torch.norm(slackness, p=2, dim=1) ** 2
    eta = torch.mean(kkt_first_order_loss) / torch.mean(complementary_slackness_loss)
    # 一阶条件损失和互补松弛损失的加权和
    total_loss = torch.mean(kkt_first_order_loss + eta * complementary_slackness_loss)

    return total_loss


# 示例数据的使用
batch_size = 10
num_variables = 5
num_objectives = 2
num_constraints = 3

# 示例输入张量
x = torch.randn(batch_size, num_variables, requires_grad=True)
lambda_ = torch.abs(torch.randn(batch_size, num_constraints, requires_grad=True))  # Lambda 应该是非负的
w = torch.randn(batch_size, num_objectives)
A = torch.randn(num_constraints, num_variables)
b = torch.randn(num_constraints)


# 示例目标函数
def f_x(x):
    """
    示例目标函数 f(x)
    假设有两个目标函数 f1 和 f2
    """
    f1 = x[:, 0]**2 + x[:, 1]**2
    f2 = x[:, 2] * x[:, 3] * x[:, 4]
    return torch.stack((f1, f2), dim=1)


# 计算损失
loss = kkt_loss_function(x, lambda_, w, f_x, A, b)
print("Loss value:", loss.item())
