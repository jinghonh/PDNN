import torch


def get_config():
    return {
        'primal_hidden_dim': 500,
        'dual_hidden_dim': 500,
        'learning_rate': 5e-5,
        'epochs': 500,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
