import torch


def get_config():
    return {
        'epochs': 1000,
        'primal_hidden_dim': 100,
        'dual_hidden_dim': 100,
        'learning_rate': 1e-4,
        'weight_decay': 5e-4,
        'batch_size': 32,
        'dropout_rate': 0.1,
        'primal_num_layers': 3,
        'dual_num_layers': 3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
