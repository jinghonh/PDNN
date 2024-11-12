import torch
from dataset import generate_data
from models import PrimalNet, DualNet
from loss import kkt_loss_function
from tqdm import tqdm
import numpy as np


def test_model(config, f_x, A, b, x_bar, w_test):
    # 加载模型
    input_dim = f_x(x_bar).shape[1]
    primal_output_dim = A.shape[1]
    dual_output_dim = A.shape[0]
    # 将数据移至设备上
    A = A.to(config['device'])
    b = b.to(config['device'])
    x_bar = x_bar.to(config['device'])
    w_test = w_test.to(config['device'])

    # 初始化模型并加载已训练好的权重
    primal_net = PrimalNet(
        input_dim=input_dim,
        hidden_dim=config['primal_hidden_dim'],
        output_dim=primal_output_dim,
        x_bar=x_bar,
        A=A,
        b=b,
        device=config['device']
    )
    dual_net = DualNet(
        input_dim=input_dim,
        hidden_dim=config['dual_hidden_dim'],
        output_dim=dual_output_dim,
        device=config['device']
    )

    # 加载已训练的模型参数
    primal_net.load_state_dict(torch.load('primal_net.pth', map_location=config['device']))
    dual_net.load_state_dict(torch.load('dual_net.pth', map_location=config['device']))

    # 将模型放入评估模式
    primal_net.eval()
    dual_net.eval()

    # 测试数据集加载器
    test_loader = generate_data(w_test)

    # 累积测试损失
    total_loss = 0.0

    total_primal_output = []
    total_dual_output = []

    with tqdm(total=len(test_loader), desc="Testing Progress") as pbar:
        for w, in test_loader:  # 获取每个测试权重向量

            # 前向传播
            primal_output = primal_net(w)
            dual_output = dual_net(w)

            # 计算损失
            loss = kkt_loss_function(primal_output, dual_output, w, f_x, A, b)

            # 累加损失
            total_loss += loss.item()

            total_primal_output.append(primal_output.detach().cpu().numpy())
            total_dual_output.append(dual_output.detach().cpu().numpy())

            # 更新进度条信息
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)

    # 计算平均损失
    average_loss = total_loss / len(test_loader)

    print(f"Test Average Loss: {average_loss:.4f}")

    # np拼接list
    total_primal_output = np.concatenate(total_primal_output, axis=0)
    total_dual_output = np.concatenate(total_dual_output, axis=0)

    return total_primal_output, total_dual_output, primal_net, dual_net
