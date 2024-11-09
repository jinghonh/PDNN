import torch


def get_config():
    return {
        'input_dim': 3,
        'output_dim': 3,
        'hidden_dim': 64,
        'num_samples': 1000,
        'learning_rate': 1e-4,
        'epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
