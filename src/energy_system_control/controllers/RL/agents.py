from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from itertools import product
import numpy as np
from collections import defaultdict
from energy_system_control.controllers.RL.exploration_policies import ExplorationPolicy


class RLAgent(ABC):
    @abstractmethod
    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, next_action):
        pass

    def make_agent(agent_info):
        agent_type = agent_info['type']
        agent_actions = agent_info['actions']
        agent_config_info = agent_info['config info']
        try:
            return AGENT_REGISTRY[agent_type](agent_actions, **agent_config_info)
        except KeyError:
            raise ValueError(f"Unknown agent type: {agent_type}")






class DiscreteActionRLAgent(RLAgent):
    def __init__(self, actions: Dict[str, List[Any]], alpha: float, gamma: float, epsilon: float, decay: float, min_epsilon: float, exploration_policy: ExplorationPolicy | None = None):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self._create_action_space()
        self.q_table = defaultdict(lambda: np.zeros(len(self.action_space)))
        self.last_reward = None
        self.last_td_error = None
        self.rng = np.random.default_rng()
        if exploration_policy:
            self.exploration_policy = exploration_policy
        else:
            self.exploration_policy = lambda state, valid_actions: self.rng.choice(valid_actions)

    def _create_action_space(self):
        # Implementation for creating action space
        for comp, values in self.actions.items():  # First check that the input is valid
            if not isinstance(values, list):
                raise ValueError(f"Actions for {comp} must be a list")
        keys = list(self.actions.keys())
        values = list(self.actions.values())
        combinations = product(*values)
        self.action_space = {i: dict(zip(keys, combo)) for i, combo in enumerate(combinations)}

    def map_to_action_space(self, valid_actions):
        output = []
        for action_id, action_info in self.action_space.items():
            keep_action = True
            for comp, valid_actions_comp in valid_actions.items():
                if action_info[comp] not in valid_actions_comp:
                    keep_action = False
                    break
            if keep_action:
                output.append(action_id)
        return output

    def select_action(self, state, valid_actions):
        if self.rng.random() < self.epsilon:
            return self.exploration_policy.select_action(state, valid_actions)
        else:
            return self.greedy_action(state, valid_actions)

    def greedy_action(self, state, valid_actions=None):
        q_values = self.q_table[state]
        if valid_actions is None:
            idx = self.rng.choice(np.flatnonzero(q_values == q_values.max()))
            return self.action_space[idx]
        valid_indices = list(valid_actions)
        permitted_q_values = q_values[valid_actions]
        # map valid actions to indices
        best_local_idxs = np.flatnonzero(permitted_q_values == permitted_q_values.max())
        best_global_idxs = [valid_indices[i] for i in best_local_idxs]
        idx = self.rng.choice(best_global_idxs)
        return self.action_space[idx]
    
    def update(self):
        if self.decay:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)


class QLearningAgent(DiscreteActionRLAgent):

    def __init__(self, actions, exploration_policy: ExplorationPolicy | None = None, alpha: float = 0.1, gamma: float = 0.99, epsilon : float = 0.1, decay: float | None = None, min_epsilon: float = 0.0):
        super().__init__(actions, alpha, gamma, epsilon, decay, min_epsilon, exploration_policy)

    def update(self, last_state, last_action, current_reward, next_state, next_action=None):
        # a_idx = self.action_space.index(last_action)
        super().update()
        a_idx = next(k for k, v in self.action_space.items() if v == last_action)
        best_next = np.max(self.q_table[next_state])
        td_target = current_reward + self.gamma * best_next
        td_error = td_target - self.q_table[last_state][a_idx]
        self.q_table[last_state][a_idx] += self.alpha * td_error
        self.last_td_error = td_error
        self.last_reward = current_reward
        self.exploration_policy.update()
        

class SARSAAgent(DiscreteActionRLAgent):
    def __init__(self, actions, exploration_policy: ExplorationPolicy, alpha=0.1, gamma=0.9):
        super().__init__(actions, exploration_policy, alpha, gamma)
    
    def update(self, last_state, last_action, current_reward, next_state, next_action):
        # a_idx = self.action_space.index(last_action)
        # next_a_idx = self.action_space.index(next_action)
        super().update()
        a_idx = next(k for k, v in self.action_space.items() if v == last_action)
        next_a_idx = next(k for k, v in self.action_space.items() if v == next_action)
        td_target = current_reward + self.gamma * self.q_table[next_state][next_a_idx]
        td_error = td_target - self.q_table[last_state][a_idx]
        self.q_table[last_state][a_idx] += self.alpha * td_error
        self.last_td_error = td_error
        self.last_reward = current_reward



AGENT_REGISTRY = {
    "q_learning": QLearningAgent,
    "sarsa": SARSAAgent,
}



