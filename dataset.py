import torch
from torch.utils.data import DataLoader, TensorDataset


def generate_data(num_samples, input_dim):
    w = torch.rand(num_samples, input_dim)
    x = torch.rand(num_samples, input_dim)
    lambda_values = torch.rand(num_samples, input_dim)

    dataset = TensorDataset(w, x, lambda_values)
    return DataLoader(dataset, batch_size=32, shuffle=True)
