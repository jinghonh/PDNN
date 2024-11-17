import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimalNet(nn.Module):
    """
    原问题（Primal）神经网络，用于逼近优化问题的弱有效解。
    输入：标量化权重向量 weight ∈ R^P
    输出：近似的决策变量 x* ∈ R^N
    """

    def __init__(self, input_dim, hidden_dim, num_layers, problem_config, device):
        super(PrimalNet, self).__init__()

        output_dim = problem_config['primal_output_dim']
        dropout_rate = problem_config['dropout_rate']

        # 定义输入层
        self.input_layer = nn.Linear(input_dim, hidden_dim, device=device)

        # 动态创建隐藏层
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))

        # 输出层，输出维度为决策变量的维度 N
        self.output_layer = nn.Linear(hidden_dim, output_dim, device=device)

        self.FeasibleOutputLayer = FeasibleOutputLayer(problem_config)

        # 激活函数
        self.activation = nn.Tanh()

        # Dropout 层
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 输入层
        x = self.activation(self.input_layer(x))

        # 逐层通过隐藏层，并在每一层后应用激活函数
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            x = self.dropout(x)

        # 输出层（不加激活函数）
        x = self.output_layer(x)
        # 通过可行输出层处理
        x = self.FeasibleOutputLayer(x)
        return x


class DualNet(nn.Module):
    """
    对偶问题（Dual）神经网络，用于估计拉格朗日乘子 λ*
    输入：标量化权重向量 weight ∈ R^P
    输出：拉格朗日乘子 λ ∈ R^M
    """

    def __init__(self, input_dim, hidden_dim, num_layers, problem_config, device):
        super(DualNet, self).__init__()
        output_dim = problem_config['dual_output_dim']
        dropout_rate = problem_config['dropout_rate']

        # 定义输入层
        self.input_layer = nn.Linear(input_dim, hidden_dim, device=device)

        # 动态创建隐藏层
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))

        # 输出层，输出维度为对偶变量的维度 M
        self.output_layer = nn.Linear(hidden_dim, output_dim, device=device)

        # 激活函数
        self.activation = nn.Tanh()

        # Dropout 层
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 输入层
        x = self.activation(self.input_layer(x))

        # 逐层通过隐藏层，并在每一层后应用激活函数
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            x = self.dropout(x)

        # 输出层使用 Softplus 激活函数，以确保输出为非负值
        output = F.softplus(self.output_layer(x))
        return output


class FeasibleOutputLayer(nn.Module):
    """
    实现公式 (9) 和 (10) 的自定义激活函数，以确保输出的解 x(weight) 是可行的。
    """

    def __init__(self, problem_config):
        x_bar = problem_config['x_bar']
        self.g_x = problem_config['g_x']
        super(FeasibleOutputLayer, self).__init__()
        self.x_bar = x_bar
        self.g_x_bar = self.g_x(x_bar)  # size: 1 x M
        self.e = 5e-5  # 避免除以零
        # self.e = 2e-4

    def forward(self, z):
        # 计算所有约束函数 g_j(z)
        g_z = self.g_x(z)  # size: B x M

        # 计算 t*(z) 的值
        with torch.no_grad():
            t_values = (g_z - self.e) / (g_z - self.g_x_bar - self.e)  # 避免除以零
            t_values[g_z < self.e] = 0  # 只对违反约束的项计算
            t_star = torch.max(t_values, dim=1)

        # 根据公式 (9) 计算最终的可行解 x(weight)
        x_w = (1 - t_star[0]).unsqueeze(-1) * z + t_star[0].unsqueeze(-1) * self.x_bar
        return x_w  # size: B x N