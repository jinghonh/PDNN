import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimalNet(nn.Module):
    """
    原问题（Primal）神经网络，用于逼近优化问题的弱有效解。
    输入：标量化权重向量 w ∈ R^P
    输出：近似的决策变量 x* ∈ R^N
    """

    def __init__(self, input_dim, hidden_dim, output_dim, x_bar, A, b, device):
        super(PrimalNet, self).__init__()

        # 定义三层隐藏层，每层包含 hidden_dim 个神经元
        self.layer1 = nn.Linear(input_dim, hidden_dim, device=device)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim, device=device)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim, device=device)

        # 输出层，输出维度为决策变量的维度 N
        self.output_layer = nn.Linear(hidden_dim, output_dim, device=device)
        self.FeasibleOutputLayer = FeasibleOutputLayer(x_bar, A, b)

    def forward(self, w):
        """
        前向传播
        """
        # 输入通过隐藏层，使用 Tanh 激活函数
        x = torch.tanh(self.layer1(w))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))

        # 输出层使用线性激活函数
        output = self.output_layer(x)
        output = self.FeasibleOutputLayer(output)
        return output


class DualNet(nn.Module):
    """
    对偶问题（Dual）神经网络，用于估计拉格朗日乘子 λ*
    输入：标量化权重向量 w ∈ R^P
    输出：拉格朗日乘子 λ ∈ R^M
    """

    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(DualNet, self).__init__()

        # 定义三层隐藏层，每层包含 hidden_dim 个神经元
        self.layer1 = nn.Linear(input_dim, hidden_dim, device=device)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim, device=device)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim, device=device)

        # 输出层，输出维度为对偶变量的维度 M
        self.output_layer = nn.Linear(hidden_dim, output_dim, device=device)

    def forward(self, w):
        """
        前向传播
        """
        # 输入通过隐藏层，使用 Tanh 激活函数
        x = torch.tanh(self.layer1(w))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))

        # 输出层使用 Softplus 激活函数，以确保输出为非负值
        output = F.softplus(self.output_layer(x))
        return output


class FeasibleOutputLayer(nn.Module):
    """
    实现公式 (9) 和 (10) 的自定义激活函数，以确保输出的解 x(w) 是可行的。
    """

    def __init__(self, x_bar, A, b):
        super(FeasibleOutputLayer, self).__init__()
        self.x_bar = x_bar
        self.A = A  # size: M x N
        self.b = b  # size: 1 x M
        self.g_x_bar = self.g(x_bar, A, b)  # size: 1 x M
        self.e = torch.min(torch.abs(self.g_x_bar)) / 2

    def forward(self, z):
        """
        z: 神经网络预测的解 size: B x N
        x_bar: 参考点（可行解）
        g_funcs: 约束函数列表

        返回：调整后的可行解 x(w)
        """

        # 计算所有约束函数 g_j(z)
        g_z = self.g(z, self.A, self.b)  # size: B x M

        # 计算 t*(z) 的值
        with torch.no_grad():
            t_values = (g_z + self.e) / (g_z - self.g_x_bar + 1e-8)  # 避免除以零
            t_values[g_z < self.e] = 0  # 只对违反约束的项计算
            t_star = torch.max(t_values).clamp(min=0, max=1)  # 确保 t* 在 [0, 1] 范围内

        # 根据公式 (9) 计算最终的可行解 x(w)
        x_w = (1 - t_star) * z + t_star * self.x_bar
        return x_w  # size: B x N

    def g(self, x, A, b):
        return torch.mm(A, x.T).T - b
