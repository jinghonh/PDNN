import torch


def get_config():
    return {
        'primal_hidden_dim': 800,
        'dual_hidden_dim': 1600,
        'num_samples': 1000,
        'learning_rate': 1e-4,
        'epochs': 1000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
