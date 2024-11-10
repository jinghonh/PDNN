import torch
from torch.utils.data import DataLoader, TensorDataset


def generate_data(w):
    dataset = TensorDataset(w)
    return DataLoader(dataset, batch_size=32, shuffle=True)


