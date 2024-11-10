import numpy as np
import torch

N = 40


class Problem:
    def __init__(self, N):
        self.N = N

    def problem1(self):
        def f_x(x: torch.Tensor):

            f1 = torch.norm(x, p=2, dim=1) ** 2 / self.N
            f2 = torch.norm(x - 2, p=2, dim=1) ** 2 / self.N
            return torch.stack((f1, f2), dim=1)

        # 定义对偶函数 d_lambda
            # 定义对偶函数 d_lambda
        def dual_function(lambda_var, w):
            # lambda_var 的形状为 (2N, B)
            # 计算 A A^T lambda_var，确保适应批次维度
            B = lambda_var.shape[1]
            AAt_lambda = A @ (A.t() @ lambda_var)  # 形状: (2N, B)
            # 计算二次项，保持每个 batch 都有单独的计算
            quad_term = - (N / 4) * torch.sum(lambda_var * AAt_lambda, dim=0)  # 形状: (B,)

            # 计算线性项
            ones_vec = torch.ones(N, 1)  # 形状: (N, 1)
            Aw1 = 2 * w[1] * (A @ ones_vec)  # 形状: (2N, 1)，将 A 和 1 向量相乘
            Aw1 = Aw1.expand(-1, B)  # 扩展以适应 batch size

            linear_term = torch.sum((Aw1 - b) * lambda_var, dim=0)  # 形状: (B,)

            # 计算常数项
            const_term = 4 * w[0] * w[1]

            # 计算对偶函数值
            dual_value = quad_term + linear_term + const_term  # 形状: (B,)
            return dual_value  # 返回 (B,) 维度的张量，表示每个 batch 的对偶值

        A = torch.cat((torch.eye(self.N), -torch.eye(self.N)), dim=1).T

        b = torch.cat((torch.ones(1, self.N), torch.zeros(1, self.N)), dim=1)

        w_train = torch.tensor([[0, 1],
                          [1 / 3, 2 / 3],
                          [2 / 3, 1 / 3],
                          [1, 0]], dtype=torch.float)

        w_test = torch.tensor([
            [i / 1000, 1 - i / 1000] for i in range(1001)
        ], dtype=torch.float32)
        x_bar = 0.5 * torch.ones(1, self.N, dtype=torch.float)

        return f_x, dual_function, A, b, w_train, w_test, x_bar
