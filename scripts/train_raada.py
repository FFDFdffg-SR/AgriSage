import yaml
from models.RAADA.raada import RAADA
from models.RAADA.environment import CropProtectionEnvironment
from utils.data_loader import load_environment_data

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_raada(config):
    field_data, resource_constraints = load_environment_data(config['data_params'])
    env = CropProtectionEnvironment(field_data, resource_constraints)
    state_dim = config['model_params']['state_dim']
    action_dim = config['model_params']['action_dim']

    raada_trainer = RAADA(env, state_dim, action_dim, learning_rate=config['train_params']['lr'])
    raada_trainer.train(episodes=config['train_params']['episodes'])

if __name__ == "__main__":
    config = load_config("experiments/config/raada_config.yaml")
    train_raada(config)
