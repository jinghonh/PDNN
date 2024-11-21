import numpy as np
import pandas as pd
import torch
from config import *
import matlab.engine
N = 40


class Problem:
    def __init__(self, N=50, n_train=20, n_test=20):
        self.P = None
        self.D1 = None
        self.D2 = None
        self.N = N
        self.config = get_config()
        n_train -= 1
        n_test -= 1
        self.w_train = torch.tensor([
            [i / n_train, 1 - i / n_train] for i in range(n_train + 1)
        ], dtype=torch.float32)

        self.w_test = torch.tensor([
            [i / n_test, 1 - i / n_test] for i in range(n_test + 1)
        ], dtype=torch.float32)

    def biObjProblem(self):
        def f_x(x):
            f1 = torch.norm(x, p=2, dim=1) ** 2 / self.N
            f2 = torch.norm(x - 2, p=2, dim=1) ** 2 / self.N
            return torch.stack((f1, f2), dim=1)

        # 定义对偶函数 d_lambda
        def d_x(lambda_var, w):
            # lambda_var: 形状为 (2N, B)，对偶变量
            # weight: 权重向量，形状为 (2,)
            lambda_var = lambda_var.T
            N, B = lambda_var.shape[0] // 2, lambda_var.shape[1]

            # 构建 A^T
            A_T = torch.cat((torch.eye(N), -torch.eye(N)), dim=1)  # 形状: (N, 2N)

            # 计算 A^T @ lambda_var，形状: (N, B)
            A_T_lambda = A_T @ lambda_var  # 形状: (N, B)

            # 计算 x^*
            x_star = 2 * w[:, 1] * torch.ones(N, B) - (N / 2) * A_T_lambda
            x_star = torch.clamp(x_star, min=0, max=1)

            return f_x(x_star)  # 形状: (N, B)

        A = torch.cat((torch.eye(self.N), -torch.eye(self.N)), dim=1).T.to(self.config['device'])

        b = torch.cat((torch.ones(1, self.N), torch.zeros(1, self.N)), dim=1).to(self.config['device'])

        def g_x(x, A=A, b=b):
            return torch.mm(A, x.T).T - b

        def x_star(w, N):
            return torch.where(w[:, 1].unsqueeze(1) <= 0.5, 2 * w[:, 1].unsqueeze(1) * torch.ones((w.size(0), N)),
                               torch.ones((w.size(0), N)))


        x_bar = 0.5 * torch.ones(1, self.N, dtype=torch.float)
        n = 1000
        w_true = torch.tensor([
            [i / n, 1 - i / n] for i in range(n + 1)
        ], dtype=torch.float32)
        # if w2>0.5,x=1, else x = 2*[w2,w2]
        x_true = x_star(w_true, self.N)

        f_true = f_x(x_true)
        # f_x, recover_x, A, b, w_train, w_test, x_bar
        return {
            'problem_name': 'problem1',
            'f_x': f_x,
            'd_x': d_x,
            'g_x': g_x,
            'w_train': self.w_train.to(self.config['device']),
            'w_test': self.w_test.to(self.config['device']),
            'x_bar': x_bar.to(self.config['device']),
            'f_true': f_true.to('cpu').numpy(),
            'primal_output_dim': self.N,
            'dual_output_dim': 2 * self.N,
            'input_dim': 2
        }

    def linearReachableSetProblem(self):
        """
        求解线性可达集问题
        :return: problem_config
        """
        N = self.N
        t_0 = 0
        t_f = 2
        u_num = 2
        h = (t_f - t_0) / N
        x_init = [0, 0]
        n = 1
        A = [[0, 1 * n], [-2 * n, -3 * n]]
        B = [[1 * n, 0], [0, 1 * n]]
        D1 = []
        D2 = []
        for u_i in range(u_num):
            for u_t in range(0, N):
                u_value = [[0] * u_num for _ in range(N)]
                u_value[u_t][u_i] = 1
                x = [x_init] + [[0, 0] for _ in range(N)]
                for t in range(1, N + 1):
                    x_1 = x[t - 1][0] + h * (
                            A[0][0] * x[t - 1][0] + A[0][1] * x[t - 1][1] + B[0][0] * u_value[t - 1][0] + B[0][1] *
                            u_value[t - 1][1])
                    x_2 = x[t - 1][1] + h * (
                            A[1][0] * x[t - 1][0] + A[1][1] * x[t - 1][1] + B[1][0] * u_value[t - 1][0] + B[1][1] *
                            u_value[t - 1][1])
                    x[t][0] = x_1
                    x[t][1] = x_2
                D1.append(x[-1][0])
                D2.append(x[-1][1])

        self.D1 = torch.tensor(D1).to(self.config['device']).reshape(-1, 1)
        self.D2 = torch.tensor(D2).to(self.config['device']).reshape(-1, 1)

        def f_x(x):
            f1 = torch.mm(x, self.D1.to(x.device))
            f2 = torch.mm(x, self.D2.to(x.device))
            return torch.cat((f1, f2), dim=1)

        def d_x(x):
            pass

        def g_x(x):
            x1 = x[:, :self.N]
            x2 = x[:, self.N:]
            g1 = torch.sqrt(x1 ** 2 + x2 ** 2) - 1
            return g1

        def expand_directions(w_train):
            if not isinstance(w_train, torch.Tensor):
                raise TypeError("w_train must be a torch.Tensor")
            if w_train.dim() != 2 or w_train.size(1) != 2:
                raise ValueError("w_train must have shape (N, 2)")

            # 生成四个方向的张量
            w_pos = w_train * torch.tensor([1, 1])
            w_neg = w_train * torch.tensor([-1, -1])
            w_cross1 = w_train * torch.tensor([1, -1])
            w_cross2 = w_train * torch.tensor([-1, 1])

            # 沿行方向拼接
            w_expanded = torch.cat((w_pos, w_neg, w_cross1, w_cross2), dim=0)
            # w_expanded = torch.cat((w_cross1, w_cross2), dim=0)
            return w_expanded

        x_bar = torch.ones(1, 2 * N) * 0
        # 在matlab命令行输入matlab.engine.shareEngine并将matlab_code文件夹添加到matlab路径
        eng = matlab.engine.connect_matlab()  # 连接matlab
        # 将python数据转换为matlab数据
        A, B, T, N = map(matlab.double, [A, B, t_f, N])
        lb, ub = matlab.double([]), matlab.double([])
        # 调用matlab函数
        data = eng.reachset(A, B, T, N, lb, ub)
        # data = pd.read_csv('data/problem2.csv', header=None, sep=' ')
        f_true = np.array(data)

        w_train = expand_directions(self.w_train)
        w_test = expand_directions(self.w_test)

        return {
            "problem_name": "problem2",
            'f_x': f_x,
            'd_x': d_x,
            'g_x': g_x,
            'w_train': w_train.to(self.config['device']),
            'w_test': w_test.to(self.config['device']),
            'x_bar': x_bar.to(self.config['device']),
            'f_true': f_true,
            'primal_output_dim': 2 * self.N,
            'dual_output_dim': 1 * self.N,
            'input_dim': 2
        }

    def multiObjProblem(self, P=2):

        def f_x(x):
            f = torch.zeros(x.shape[0], P).to(x.device)
            for i in range(P):
                f[:, i] = (x[:, i] - 1) ** 2 + torch.sum(x[:, [j for j in range(self.N) if j != i]] ** 2, dim=1)
            return f

        def d_x(lambda_, weight):
            x = torch.zeros_like(weight)
            for i in range(P):
                x[:, i] = (weight[:, i] + lambda_[:, i]) * f_x(weight[:, i] + lambda_[:, i]) - lambda_[:, i]
            return x

        def g_x(x):
            return f_x(x) - 1

        x_bar = torch.cat((torch.ones(1, P) / P, torch.zeros(1, self.N - P)), dim=1).to(self.config['device'])

        return {
            "problem_name": "problem3",
            'f_x': f_x,
            'd_x': d_x,
            'g_x': g_x,
            'w_train': self.w_train.to(self.config['device']),
            'w_test': self.w_test.to(self.config['device']),
            'x_bar': x_bar.to(self.config['device']),
            'f_true': None,
            # 'f_true': f_true.to(self.config['device']),
            'primal_output_dim': self.N,
            'dual_output_dim': 2,
            'input_dim': 2
        }


if __name__ == '__main__':
    problems = Problem(5, 100, 100)
    problem_config = problems.linearReachableSetProblem()
    print(problem_config['f_true'])
