from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.rewards = []
        self.actions = []
        self.latencys = []

    def _on_step(self) -> bool:
        # Access rewards and actions from the model
        reward = self.locals['rewards']
        action = self.locals['actions']
        info = self.locals['infos'][0]
        latency = info.get('latency', None)

        # Store the rewards and actions
        self.rewards.append(reward)
        self.actions.append(action)
        self.latencys.append(latency)

        return True

    def get_training_data(self):
        # Convert lists to numpy arrays for easier handling
        return np.array(self.actions), np.array(self.rewards), np.array(self.latencys)
