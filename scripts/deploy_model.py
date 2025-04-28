import torch
from models.SADFNet.sadfnet import SADFNet
import yaml


def load_model(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = SADFNet(**config['model_params'])
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


def predict(model, image_data, sensor_data):
    model.eval()
    with torch.no_grad():
        outputs = model(image_data, sensor_data)
    return outputs


if __name__ == "__main__":
    model = load_model("experiments/config/sadfnet_config.yaml", "path_to_trained_model.pth")
    # Load or prepare your image_data and sensor_data here
    # outputs = predict(model, image_data, sensor_data)
    print("Model is ready for deployment.")
