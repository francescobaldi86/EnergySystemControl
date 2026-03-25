from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, List
import numpy as np
import pandas as pd
from collections import defaultdict
from energy_system_control.controllers.base import Controller  # or move Controller to core/model.py
from energy_system_control.core.base_classes import InitContext, Sensor
from energy_system_control.controllers.predictors import Predictor
from energy_system_control.controllers.reward_functions import RewardFunction, CompositeReward, REWARD_REGISTRY
from energy_system_control.sim.state import SimulationState

RLType = Literal["q_learning", "sarsa", "dqn", "deep_sarsa"]

class RLAgent(ABC):
    @abstractmethod
    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, done: bool):
        pass

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

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if len(df.columns) > 1:
            raise ValueError("TemporalAggregator only supports single-column DataFrames")
        values = df.values.flatten()
        n = len(values)

        block_size = n // self.n_blocks
        reduced = []

        for i in range(self.n_blocks):
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
        if isinstance(value, pd.DataFrame):
            if self.temporal_aggregator is None:
                raise ValueError("TemporalAggregator required for DataFrame input")

            values = self.temporal_aggregator.transform(value)

        else:
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



class QLearningAgent(RLAgent):
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _state_to_key(self, state):
        return tuple(state.values())  # you may want something smarter

    def select_action(self, state):
        key = self._state_to_key(state)

        if np.random.rand() < self.epsilon:
            idx = np.random.randint(len(self.actions))
        else:
            idx = np.argmax(self.q_table[key])

        return self.actions[idx]

    def update(self, state, action, reward, next_state, done):
        s = self._state_to_key(state)
        s_next = self._state_to_key(next_state)

        a_idx = self.actions.index(action)

        best_next = np.max(self.q_table[s_next])

        td_target = reward + self.gamma * best_next * (1 - done)
        td_error = td_target - self.q_table[s][a_idx]

        self.q_table[s][a_idx] += self.alpha * td_error


class RLController(Controller):
    def __init__(
        self,
        name,
        controlled_components,
        sensors: List[str],
        predictors: List[str],
        agent: RLAgent,
        reward_function: RewardFunction,
    ):
        super().__init__(name, controlled_components, sensors, predictors)
        self.agent = agent
        self.reward_function = reward_function


    def load_sensors(self, sensors: Dict[str, Sensor], predictors: Dict[str, Predictor]):
        super().load_sensors(sensors)
        self.predictors = {var: predictors[predictor_name] for var, predictor_name in self.predictor_names.items()}

    def initialize(self, ctx: InitContext):
        super().initialize(ctx)
        self.last_state = None
        self.last_action = None
        # Actualize the reward function requirements based on the sensors and predictors

    def get_action(self, state: SimulationState):
        state = self.get_state()

        action = self.agent.select_action(state)

        self.last_state = state
        self.last_action = action

        return action
    
    def get_state(self):
        state = {}
        for sensor_name in self.sensors:
            state[sensor_name] = self.env.sensors[sensor_name].get_value(self.env, self.env.state)
        for predictor_name in self.predictors:
            state[predictor_name] = self.env.controllers[predictor_name].get_prediction(self.env, self.env.state)
        return state

    def step(self, next_state, reward, done):
        """
        Call this AFTER environment transition
        """
        self.agent.update(
            self.last_state,
            self.last_action,
            reward,
            next_state,
            done
        )
        
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
    

class RLControllerValueBased(RLController):
    def __init__(self, *args, state_discretizer: StateDiscretizer, **kwargs):  
        super().__init__(*args, **kwargs)
        self.state_discretizer = state_discretizer

    def get_action(self):
        continuous_state = self.obs
        discrete_state = self.state_discretizer.transform(continuous_state)

        action = self.agent.select_action(discrete_state)

        self.last_state = discrete_state
        self.last_action = action

        return action






AGENT_REGISTRY = {
    "q_learning": QLearningAgent,
}

def make_agent(agent_type: RLType, **kwargs) -> RLAgent:
    try:
        return AGENT_REGISTRY[agent_type](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown agent type: {agent_type}")