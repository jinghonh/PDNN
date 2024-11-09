import torch


def kkt_loss(primal_output, dual_output, w, x, lambda_values, eta=1.0):
    # 计算 KKT 条件中的一阶条件
    first_order_condition = torch.norm(primal_output.T @ w + dual_output @ x, p=2)

    # 计算互补松弛条件
    complementary_slackness = torch.norm(torch.diag(dual_output) @ x, p=2)

    # 总损失函数
    loss = first_order_condition + eta * complementary_slackness
    return loss
