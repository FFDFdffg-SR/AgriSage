import torch
import torch.nn as nn
import torch.optim as optim

class RAADAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(RAADAgent, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.model(state)

    def select_action(self, state):
        with torch.no_grad():
            logits = self.forward(state)
            action = torch.argmax(logits, dim=-1)
        return action.item()
