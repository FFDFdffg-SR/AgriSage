import torch
from models.SADFNet.sadfnet import SADFNet
from utils.data_loader import get_test_loader
from utils.metrics import calculate_metrics
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_sadfnet(config, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SADFNet(**config['model_params']).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    test_loader = get_test_loader(config['data_params'])
    calculate_metrics(model, test_loader, device)

if __name__ == "__main__":
    config = load_config("experiments/config/sadfnet_config.yaml")
    evaluate_sadfnet(config, "path_to_your_model_checkpoint.pth")
