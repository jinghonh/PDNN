import matplotlib.pyplot as plt
import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
import pandas as pd


def plot_primal_net_frontier(f, x, fx_true=None, epoch=None, loss=None):
    if fx_true is not None:
        plt.scatter(fx_true[:, 0], fx_true[:, 1], color='r', label="True Frontier")
    fx = f(x)
    fx = fx.detach().numpy()
    plt.scatter(fx[:, 0], fx[:, 1])
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    if epoch is not None:
        if loss is not None:
            plt.title(f"Approximated Weakly Efficient Frontier (Epoch {epoch}, Loss: {loss:.4f})")
        else:
            plt.title(f"Approximated Weakly Efficient Frontier (Epoch {epoch})")
    else:
        plt.title("Approximated Weakly Efficient Frontier")
    plt.show()


def plot_dual_net_frontier(primal_net, test_loader, f):
    pass


def draw_neural_net(model, input_dim, hidden_dim, output_dim):
    # 获取权重
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.data.cpu().numpy())

    G = nx.DiGraph()

    # 创建节点并添加层属性
    # 输入层
    for i in tqdm(range(input_dim), desc="Adding input layer nodes"):
        G.add_node(f'I{i}', layer=0)

    # 三层隐藏层
    for l in tqdm(range(3), desc="Adding hidden layer nodes"):
        for j in range(hidden_dim):
            G.add_node(f'H{l}_{j}', layer=l + 1)

    # 输出层
    for i in tqdm(range(output_dim), desc="Adding output layer nodes"):
        G.add_node(f'O{i}', layer=4)

    # 使用 multipartite 布局
    pos = nx.multipartite_layout(G, subset_key='layer')
    plt.figure(figsize=(12, 8))

    # 添加边和权重
    all_weights = []  # 用于存储所有的权重值，以便后续归一化
    for l, weight_matrix in enumerate(tqdm(weights, desc="Adding edges and weights for each layer")):
        if l == 0:
            for i in range(input_dim):
                for j in range(hidden_dim):
                    weight = weight_matrix[j, i]
                    all_weights.append(np.abs(weight))
                    G.add_edge(f'I{i}', f'H{l}_{j}', weight=weight)
        elif l == len(weights) - 1:
            for j in range(hidden_dim):
                for o in range(output_dim):
                    weight = weight_matrix[o, j]
                    all_weights.append(np.abs(weight))
                    G.add_edge(f'H{l - 1}_{j}', f'O{o}', weight=weight)
        else:
            for j in range(hidden_dim):
                for k in range(hidden_dim):
                    weight = weight_matrix[k, j]
                    all_weights.append(np.abs(weight))
                    G.add_edge(f'H{l - 1}_{j}', f'H{l}_{k}', weight=weight)

    # 计算归一化参数
    weight_min = min(all_weights)
    weight_max = max(all_weights)

    # 绘制图形
    for (u, v, d) in tqdm(G.edges(data=True), desc="Drawing edges"):
        weight = d['weight']
        # 对 weight 进行归一化到 [0, 1] 的范围
        if weight_max > weight_min:  # 确保分母不为零
            normalized_weight = (np.abs(weight) - weight_min) / (weight_max - weight_min)
        else:
            normalized_weight = 0.5  # 如果所有权重相同，统一设为中等颜色

        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], alpha=0.5, edge_color=plt.cm.Blues(normalized_weight))

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color='red', node_size=50)
    # nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("Neural Network Graph with Weighted Edges (Normalized)")
    plt.show()


def draw_neural_net_weights_heatmap(model):
    # 获取权重
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.data.cpu().numpy())

    num_layers = len(weights)
    plt.figure(figsize=(15, 5 * num_layers))

    # 绘制每一层的权重热力图
    for idx, weight_matrix in enumerate(tqdm(weights, desc="Drawing heatmaps for each layer")):
        plt.subplot(num_layers, 1, idx + 1)
        plt.imshow(weight_matrix, cmap='hot', aspect='auto')  # 使用 'hot' 颜色映射绘制热力图
        plt.colorbar()  # 添加颜色条以指示权重大小
        plt.title(f'Weight Heatmap for Layer {idx + 1}')
        plt.xlabel('Input Neurons')
        plt.ylabel('Output Neurons')

    plt.tight_layout()
    plt.show()


def save_weights_to_excel(model, file_name="neural_net_weights.xlsx"):
    # 获取权重
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.data.cpu().numpy())

    # 创建一个 Pandas 的 ExcelWriter，用于保存权重矩阵
    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
        for idx, weight_matrix in enumerate(tqdm(weights, desc="Saving weights to Excel")):
            # 将权重矩阵转换为 DataFrame
            df = pd.DataFrame(weight_matrix)
            # 给每个 sheet 命名为 "Layer {idx + 1}"
            sheet_name = f'Layer_{idx + 1}'
            # 保存到对应的 Excel sheet 中
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"权重矩阵已成功保存到 '{file_name}' 文件中。")
