from torch.optim import Adam

from dataset import generate_data
from loss import kkt_loss_function
from models import PrimalNet, DualNet
from utils import *


def train_model(config, problem_config):
    input_dim = problem_config['input_dim']
    problem_config['dropout_rate'] = config['dropout_rate']

    # 初始化模型
    primal_net = PrimalNet(
        input_dim=input_dim,
        hidden_dim=config['primal_hidden_dim'],
        num_layers=config['primal_num_layers'],
        problem_config=problem_config,
        device=config['device']
    )
    dual_net = DualNet(
        input_dim=input_dim,
        hidden_dim=config['dual_hidden_dim'],
        num_layers=config['dual_num_layers'],
        problem_config=problem_config,
        device=config['device']
    )

    # 定义优化器
    optimizer = Adam(list(primal_net.parameters()) + list(dual_net.parameters()), lr=config['learning_rate'],
                     weight_decay=1e-4)
    train_loader = generate_data(problem_config['w_train'])

    # 训练循环
    with tqdm(total=config['epochs'], desc="Training Progress") as pbar:
        for epoch in range(config['epochs']):
            total_loss = 0
            # 训练模式
            primal_net.train()
            dual_net.train()

            for w, in train_loader:  # 假设 train_loader 提供权重向量 weight

                # 前向传播
                primal_output = primal_net(w)
                dual_output = dual_net(w)

                # 计算损失
                loss = kkt_loss_function(primal_output, dual_output, w, problem_config)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 累加损失
                total_loss += loss.item()

            # 更新进度条信息
            pbar.set_postfix(loss=f"{total_loss:.4f}")
            pbar.update(1)

            if epoch % 10 == 0:
                primal_net.eval()
                plot_primal_net_frontier(problem_config['f_x'], primal_net(problem_config['w_test']).to('cpu'),
                                         problem_config['f_true'], epoch, total_loss)

    torch.save(primal_net.state_dict(), 'save_model/primal_net.pth')
    torch.save(dual_net.state_dict(), 'save_model/dual_net.pth')
