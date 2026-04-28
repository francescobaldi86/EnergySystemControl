from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np

class ExplorationPolicy(ABC):

    def __init__(self, config_info):
        self.rng = np.random.default_rng()
        self.config_info = config_info

    @abstractmethod
    def select_action(self, agent, state, obs):
        """
        agent: gives access to Q-values or policy
        state: current state
        """
        pass

    def update(self, *args, **kwargs):
        """Optional: for decay, schedules, etc."""
        pass

    def make_exploration_policy(exporation_policy_info):
        exporation_policy_type = exporation_policy_info['type']
        exporation_policy_config_info = exporation_policy_info['config info']
        try:
            return EXPLORATION_POLICY_REGISTRY[exporation_policy_type](exporation_policy_config_info)
        except KeyError:
            raise ValueError(f"Unknown agent type: {exporation_policy_type}")


class BiasFunction():
    def __init__(self, sensor_name: str | None = None, config_info: Dict[Tuple, Dict] | None = None):
        self.sensor_name = sensor_name
        self.info = config_info
        for bounds, probs in self.info.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 :
                raise KeyError(f'Config information about the RL exploration policy for interval {bounds} is not correct.')
            if not isinstance(probs, (list, tuple)):
                raise ValueError(f'The probabilities provided for the interval {bounds} are not correct. Provided values are {probs}')
    
    def get_bias(self, valid_actions, obs):
        value = obs[self.sensor_name]
        if len(valid_actions) == 1:
            return [1.0]
        else:
            for bounds, probs in self.info.items():
                if value > bounds[0] and value <= bounds[1]:
                    probs = [v for k, v in probs if k in valid_actions]
                    return [p / sum(probs) for p in probs]
            raise ValueError(f'The value {value} is not in any of the intervals provided in the config information.')

    @classmethod
    def make_bias_function(cls, control_variable, config_info):
        return cls(control_variable, config_info)


class EpsilonGreedy(ExplorationPolicy):
    def __init__(self, config_info: dict):
        super().__init__(config_info)
        if 'bias function' in config_info.keys():
            self.bias_function = BiasFunction(config_info['bias function']['control variable'], config_info['bias function']['config info'])
        else:
            self.bias_function = None

    def select_action(self, state, valid_actions, obs):
        if self.config_info is None:
            probs = [1/len(valid_actions) for k in valid_actions]
        else:
            probs = self.bias_function.get_bias(valid_actions, obs)
        return self.rng.choice(valid_actions, p = probs)
        


EXPLORATION_POLICY_REGISTRY = {
    "epsilon-greedy": EpsilonGreedy,
}