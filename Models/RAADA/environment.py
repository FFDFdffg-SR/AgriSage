import numpy as np

class CropProtectionEnvironment:
    def __init__(self, field_data, resource_constraints):
        self.field_data = field_data
        self.resource_constraints = resource_constraints
        self.state = None

    def reset(self):
        self.state = self.field_data.reset()
        return self.state

    def step(self, action):
        next_state, reward, done, info = self.field_data.simulate(action, self.resource_constraints)
        return next_state, reward, done, info
