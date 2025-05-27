"""
Class to collect and save all transitions from a policy training function. 
"""

import numpy as np

class CollectDataset:

    def __init__(self):
        self.history = {'obs': [],
                        'action': [],
                        'next_obs': [],
                        'reward': [],
                        'done': []}

    def __call__(self, log_dict):
        self.history['obs'].append(log_dict['obs'])
        self.history['action'].append(log_dict['action'])
        self.history['next_obs'].append(log_dict['next_obs'])
        self.history['reward'].append(log_dict['reward'])
        self.history['done'].append(log_dict['done'])

    def get_combined_history(self):
        return {'obs': np.array(self.history['obs']),
                'action': np.array(self.history['action']),
                'next_obs': np.array(self.history['next_obs']),
                'reward': np.array(self.history['reward']),
                'done': np.array(self.history['done'])}