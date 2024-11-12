import numpy as np
import torch

N = 40


def compute_gx(A):
    n = A.shape[0]
    B = torch.zeros_like(A)
    for i in range(n):
        A_i = A[i, :]
        # 检查是否为零向量
        if torch.norm(A_i) == 0:
            # 如果是零向量，任意向量都与之正交
            y = torch.randn_like(A_i)
        else:
            # 生成随机向量
            x = torch.randn_like(A_i)
            # 计算 x 在 A_i 方向上的投影
            proj = (A_i @ x) / (A_i @ A_i) * A_i
            # 得到与 A_i 正交的向量
            y = x - proj
            # 可选：将向量归一化
            y = y / torch.norm(y)
        # 将结果赋值给 B 的第 i 列
        B[i, :] = y
    return B


class Problem:
    def __init__(self, N=50):
        self.N = N

    def problem1(self):
        def f_x(x: torch.Tensor):
            f1 = torch.norm(x, p=2, dim=1) ** 2 / self.N
            f2 = torch.norm(x - 2, p=2, dim=1) ** 2 / self.N
            return torch.stack((f1, f2), dim=1)

        # 定义对偶函数 d_lambda
        def recover_x(lambda_var, w):
            # lambda_var: 形状为 (2N, B)，对偶变量
            # w: 权重向量，形状为 (2,)
            lambda_var = lambda_var.T
            N, B = lambda_var.shape[0] // 2, lambda_var.shape[1]

            # 构建 A^T
            A_T = torch.cat((torch.eye(N), -torch.eye(N)), dim=1)  # 形状: (N, 2N)

            # 计算 A^T @ lambda_var，形状: (N, B)
            A_T_lambda = A_T @ lambda_var  # 形状: (N, B)

            # 计算 x^*
            x_star = 2 * w[:,1] * torch.ones(N, B) - (N / 2) * A_T_lambda

            # 由于 x^* 可能不满足约束 [0, 1]，需要将其截断到 [0, 1] 范围内
            x_star = torch.clamp(x_star, min=0, max=1)

            return f_x(x_star)  # 形状: (N, B)

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

        return f_x, recover_x, A, b, w_train, w_test, x_bar
