import torch
from config import get_config
from train import train_model

if __name__ == "__main__":
    config = get_config()
    train_model(config)
