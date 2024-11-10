import torch
from torch import device
from torch.optim import Adam
from dataset import generate_data
from models import PrimalNet, DualNet
from loss import kkt_loss_function


def train_model(config, f_x, A, b, x_bar, w):
    input_dim = f_x(x_bar).shape[1]
    primal_output_dim = A.shape[1]
    dual_output_dim = A.shape[0]
    # 加载数据
    A = A.to(config['device'])
    b = b.to(config['device'])
    x_bar = x_bar.to(config['device'])
    w = w.to(config['device'])

    # 初始化模型
    primal_net = PrimalNet(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        output_dim=primal_output_dim,
        x_bar=x_bar,
        A=A,
        b=b,
        device=config['device']
    )
    dual_net = DualNet(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim']*2,
        output_dim=dual_output_dim,
        device=config['device']
    )

    # 定义优化器
    optimizer = Adam(list(primal_net.parameters()) + list(dual_net.parameters()), lr=config['learning_rate'])
    # 训练循环
    for epoch in range(config['epochs']):
        total_loss = 0
        # for w in train_loader:

        # 前向传播
        primal_output = primal_net(w)
        dual_output = dual_net(w)

        # 计算损失
        loss = kkt_loss_function(primal_output, dual_output, w, f_x, A, b)  # x, lambda_, w, f_x, A, b

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {total_loss:.4f}")
    torch.save(primal_net.state_dict(), 'primal_net.pth')
    torch.save(dual_net.state_dict(), 'dual_net.pth')
