import torch
import torch.nn as nn
import torch.optim as optim
from models.SADFNet.sadfnet import SADFNet
from utils.data_loader import get_train_val_loaders
from utils.metrics import calculate_metrics
import yaml


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_sadfnet(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SADFNet(**config['model_params']).to(device)

    train_loader, val_loader = get_train_val_loaders(config['data_params'])
    optimizer = optim.Adam(model.parameters(), lr=config['train_params']['lr'])
    criterion = nn.MSELoss()

    for epoch in range(config['train_params']['epochs']):
        model.train()
        total_loss = 0
        for images, sensors, targets in train_loader:
            images, sensors, targets = images.to(device), sensors.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images, sensors)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader)}")

        # Validation
        model.eval()
        calculate_metrics(model, val_loader, device)


if __name__ == "__main__":
    config = load_config("experiments/config/sadfnet_config.yaml")
    train_sadfnet(config)
