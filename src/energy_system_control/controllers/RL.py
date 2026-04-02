from abc import ABC, abstractmethod
from itertools import product
from typing import Any, Dict, Literal, List
import numpy as np
import pandas as pd
from collections import defaultdict
from energy_system_control.controllers.base import Controller  # or move Controller to core/model.py
from energy_system_control.core.base_classes import InitContext, Sensor
from energy_system_control.controllers.predictors import Predictor
from energy_system_control.controllers.reward_functions import RewardFunction, CompositeReward, REWARD_REGISTRY
from energy_system_control.sim.state import SimulationState
from energy_system_control.core.registry import SignalRegistry

RLType = Literal["q_learning", "sarsa", "dqn", "deep_sarsa"]

class RLAgent(ABC):
    @abstractmethod
    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, next_action):
        pass

class DiscreteActionRLAgent(RLAgent):
    def __init__(self, actions: Dict[str, List[Any]]):
        self.actions = actions
        self._create_action_space()

    def _create_action_space(self):
        # Implementation for creating action space
        for comp, values in self.actions.items():  # First check that the input is valid
            if not isinstance(values, list):
                raise ValueError(f"Actions for {comp} must be a list")
        keys = list(self.actions.keys())
        values = list(self.actions.values())
        combinations = product(*values)
        self.action_space = {i: dict(zip(keys, combo)) for i, combo in enumerate(combinations)}


class TemporalAggregator:
    def __init__(self, n_blocks: int, agg_func: str = "mean"):
        """
        Parameters
        ----------
        n_blocks : int
            Number of time blocks to reduce to
        agg_func : str
            Aggregation function: "mean", "sum", "max"
        """
        self.n_blocks = n_blocks
        self.agg_func = agg_func

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.ndim != 1:
            raise ValueError("TemporalAggregator only supports 1D arrays")
        n = len(values)
        if n < self.n_blocks:
            return values  # If there are fewer values than blocks, just return the original values
        else:
            block_size = n // self.n_blocks
            reduced = []


            for i in range(min(self.n_blocks, n)):  # in case n < n_blocks  
                start = i * block_size
                end = (i + 1) * block_size if i < self.n_blocks - 1 else n
                block = values[start:end]

                if self.agg_func == "mean":
                    reduced.append(np.mean(block))
                elif self.agg_func == "sum":
                    reduced.append(np.sum(block))
                elif self.agg_func == "max":
                    reduced.append(np.max(block))
                else:
                    raise ValueError(f"Unknown aggregation: {self.agg_func}")

            return np.array(reduced)
    

class Discretizer:
    def __init__(
        self,
        vmin: float,
        vmax: float,
        n_bins: int,
        temporal_aggregator: TemporalAggregator | None = None,
    ):
        self.vmin = vmin
        self.vmax = vmax
        self.n_bins = n_bins
        self.temporal_aggregator = temporal_aggregator
        self.bin_edges = np.linspace(vmin, vmax, n_bins + 1)

    def _digitize(self, values: np.ndarray) -> np.ndarray:
        values = np.clip(values, self.vmin, self.vmax)
        bin_idx = np.digitize(values, self.bin_edges) - 1
        return np.minimum(bin_idx, self.n_bins - 1)

    def discretize(self, value) -> np.ndarray:
        if isinstance(value, np.ndarray):
            if self.temporal_aggregator:  # If there is a temporal aggregator, we use it
                values = self.temporal_aggregator.transform(value)
            else:
                values = value
        else: # In case it's a float or int
            values = np.array([value])

        return self._digitize(values)

    
class StateDiscretizer:
    def __init__(self, config: dict):
        self.discretizers = {}

        for var, cfg in config.items():
            aggregator = None

            if "temporal" in cfg:
                aggregator = TemporalAggregator(
                    n_blocks=cfg["temporal"]["n_blocks"],
                    agg_func=cfg["temporal"].get("agg", "mean"),
                )

            self.discretizers[var] = Discretizer(
                vmin=cfg["min"],
                vmax=cfg["max"],
                n_bins=cfg["bins"],
                temporal_aggregator=aggregator,
            )

    def transform(self, obs: dict = {}, predictions: dict = {}) -> tuple:
        state = []
        for var, discretizer in self.discretizers.items():
            if var in obs:
                values = discretizer.discretize(obs[var])
            elif var in predictions:
                values = discretizer.discretize(predictions[var])
            else:
                raise KeyError(f"{var} not found in obs or predictions")
            state.extend(values.tolist())

        return tuple(state)



class QLearningAgent(DiscreteActionRLAgent):
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(actions)
        self.q_table = defaultdict(lambda: np.zeros(len(self.action_space)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            idx = np.random.randint(len(self.action_space))
        else:
            av = self.q_table[state]
            idx = np.random.choice(np.flatnonzero(av == av.max()))

        return self.action_space[idx]

    def update(self, last_state, last_action, current_reward, next_state, next_action=None):
        # a_idx = self.action_space.index(last_action)
        a_idx = next(k for k, v in self.action_space.items() if v == last_action)
        best_next = np.max(self.q_table[next_state])
        td_target = current_reward + self.gamma * best_next
        td_error = td_target - self.q_table[last_state][a_idx]
        self.q_table[last_state][a_idx] += self.alpha * td_error

class SARSAAgent(DiscreteActionRLAgent):
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            idx = np.random.randint(len(self.actions))
        else:
            av = self.q_table[state]
            idx = np.random.choice(np.flatnonzero(av == av.max()))
        return self.action_space[idx]
    
    def update(self, last_state, last_action, current_reward, next_state, next_action):
        # a_idx = self.action_space.index(last_action)
        # next_a_idx = self.action_space.index(next_action)
        a_idx = next(k for k, v in self.action_space.items() if v == last_action)
        next_a_idx = next(k for k, v in self.action_space.items() if v == next_action)
        td_target = current_reward + self.gamma * self.q_table[next_state][next_a_idx]
        td_error = td_target - self.q_table[last_state][a_idx]
        self.q_table[last_state][a_idx] += self.alpha * td_error


class RLController(Controller):
    def __init__(
        self,
        name,
        controlled_components,
        sensors: Dict[str, str],
        agent: RLAgent,
        reward_function: RewardFunction,
        predictors: Dict[str, str] = {},
        prediction_horizon: int = 1
    ):
        super().__init__(name, controlled_components, sensors, predictors)
        self.agent = agent
        self.reward_function = reward_function
        self.horizon = prediction_horizon

    def initialize(self, ctx: InitContext):
        super().initialize(ctx)
        self.last_state = None
        self.last_action = None
        # Initialize reward function if needed
        self.reward_function.initialize(ctx)

    def create_agent_data_registry(self, ctx: InitContext):
        rl_registry = SignalRegistry()
        # Example signals
        rl_registry.register("rl", "reward")
        rl_registry.register("rl", "td_error")
        rl_registry.register("rl", "action")
        ctx.environment.rl_registry = rl_registry
        

    
    def preprocess_state(self) -> List[float]:
        # Transforms the state and prediction in a numpy array or similar structure that can be fed to the agent
        sensor_measurements = list(self.obs.values()) 
        predictions = list(self.predictions.values())
        return sensor_measurements + predictions
        
    @classmethod
    def from_config(
        cls,
        name,
        controlled_components,
        sensors,
        agent_type: RLType,
        agent_kwargs: dict,
        reward_function,
    ):
        agent = make_agent(agent_type, **agent_kwargs)

        return cls(
            name=name,
            controlled_components=controlled_components,
            sensors=sensors,
            agent=agent,
            reward_function=reward_function,
        )
    

class RLControllerTabular(RLController):
    def __init__(self,
                name: str,
                sensors: Dict[str, str],
                predictors: Dict[str, str],
                actions: Dict[str, List[Any]],
                reward_function: RewardFunction, 
                state_discretizer: StateDiscretizer,
                agent_type: RLType,
                agent_kwargs: dict = {}):
        agent = make_agent(agent_type, actions=actions, **agent_kwargs)
        controlled_components = list(actions.keys())
        super().__init__(name = name, controlled_components=controlled_components, sensors=sensors, agent = agent, predictors=predictors, reward_function=reward_function)
        self.state_discretizer = state_discretizer

    def preprocess_state(self):
        return self.state_discretizer.transform(obs = self.obs, predictions=self.predictions)


class QLearningController(RLControllerTabular):
    def __init__(self,
                name: str,
                sensors: Dict[str, str],
                actions: Dict[str, List[Any]],
                reward_function: RewardFunction, 
                state_discretizer: StateDiscretizer,
                predictors: Dict[str, str] = {},
                agent_kwargs: dict = {}):
        super().__init__(name = name, sensors=sensors, predictors=predictors, actions=actions, reward_function=reward_function, state_discretizer=state_discretizer, agent_type="q_learning", agent_kwargs=agent_kwargs)

    def get_action(self, state: SimulationState):
        RL_state = self.preprocess_state()
        reward = self.reward_function.compute(state)
        if self.last_state is not None and self.last_action is not None:
            self.agent.update(
                last_state = self.last_state, 
                last_action = self.last_action, 
                current_reward = reward,
                next_state = RL_state,
                next_action = None) # In Q-learning the next action is not needed for the update)
        action = self.agent.select_action(RL_state)
        self.last_state = RL_state
        self.last_action = action
        return action
    
class SARSAController(RLControllerTabular):
    def __init__(self,
                name: str,
                sensors: Dict[str, str],
                predictors: Dict[str, str],
                actions: Dict[str, List[Any]],
                reward_function: RewardFunction, 
                state_discretizer: StateDiscretizer,
                agent_kwargs: dict = {}):
        super().__init__(name = name, sensors=sensors, predictors=predictors, actions=actions, reward_function=reward_function, state_discretizer=state_discretizer, agent_type="sarsa", agent_kwargs=agent_kwargs)

    def get_action(self, state: SimulationState):
        RL_state = self.preprocess_state()
        reward = self.reward_function.compute(state)
        action = self.agent.select_action(RL_state)
        if self.last_state is not None and self.last_action is not None:
            self.agent.update(
                last_state = self.last_state, 
                last_action = self.last_action, 
                current_reward = reward,
                next_state = RL_state,
                next_action = action) # In SARSA the next action is needed for the update)
        self.last_state = RL_state
        self.last_action = action
        return action


AGENT_REGISTRY = {
    "q_learning": QLearningAgent,
    "sarsa": SARSAAgent,
}

def make_agent(agent_type: RLType, **kwargs) -> RLAgent:
    try:
        return AGENT_REGISTRY[agent_type](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown agent type: {agent_type}")