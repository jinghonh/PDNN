import torch
from torch.optim import Adam
from dataset import generate_data
from models import PrimalNet, DualNet
from loss import kkt_loss


def train_model(config):
    # 加载数据
    train_loader = generate_data(config['num_samples'], config['input_dim'])

    # 初始化模型
    primal_net = PrimalNet(config['input_dim'], config['hidden_dim'], config['output_dim']).to(config['device'])
    dual_net = DualNet(config['input_dim'], config['hidden_dim'], config['output_dim']).to(config['device'])

    # 定义优化器
    optimizer = Adam(list(primal_net.parameters()) + list(dual_net.parameters()), lr=config['learning_rate'])

    # 训练循环
    for epoch in range(config['epochs']):
        total_loss = 0
        for w, x, lambda_values in train_loader:
            w, x, lambda_values = w.to(config['device']), x.to(config['device']), lambda_values.to(config['device'])

            # 前向传播
            primal_output = primal_net(w)
            dual_output = dual_net(w)

            # 计算损失
            loss = kkt_loss(primal_output, dual_output, w, x, lambda_values)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {total_loss:.4f}")
    torch.save(primal_net.state_dict(), 'primal_net.pth')
    torch.save(dual_net.state_dict(), 'dual_net.pth')
