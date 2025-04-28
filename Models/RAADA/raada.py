import torch
import torch.optim as optim
from .agent import RAADAgent

class RAADA:
    def __init__(self, env, state_dim, action_dim, learning_rate=1e-3):
        self.env = env
        self.agent = RAADAgent(state_dim, action_dim)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = torch.tensor(self.env.reset(), dtype=torch.float32)
            done = False
            total_reward = 0

            while not done:
                action = self.agent.select_action(state.unsqueeze(0))
                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)

                # Assume simple TD update
                target = reward + (0.99 * torch.max(self.agent(next_state.unsqueeze(0))))
                prediction = self.agent(state.unsqueeze(0))[0, action]

                loss = self.criterion(prediction, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                state = next_state
                total_reward += reward

            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
