import torch


def get_config():
    return {
        'hidden_dim': 800,
        'num_samples': 1000,
        'learning_rate': 1e-4,
        'epochs': 1000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
