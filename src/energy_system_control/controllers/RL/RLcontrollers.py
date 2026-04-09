from typing import Any, Dict, Literal, List, Tuple
from energy_system_control.controllers.base import Controller  # or move Controller to core/model.py
from energy_system_control.core.base_classes import InitContext, Sensor
from energy_system_control.controllers.predictors import Predictor
from energy_system_control.controllers.RL.reward_functions import RewardFunction
from energy_system_control.controllers.RL.agents import RLAgent
from energy_system_control.controllers.RL.exploration_policies import ExplorationPolicy
from energy_system_control.controllers.RL.discretizers import TemporalAggregator, StateDiscretizer
from energy_system_control.sim.state import SimulationState
from energy_system_control.core.registry import SignalRegistry

RLType = Literal["q_learning", "sarsa", "dqn", "deep_sarsa"]

class ValidStatesFunction():
    
    def __init__(self, control_variable: str, config_info: Dict[Tuple, Dict]):
        self.control_variable = control_variable
        self.info = config_info
        for bounds, valid_actions in self.info.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 :
                raise KeyError(f'Config information about the RL exploration policy for interval {bounds} is not correct.')
            if not isinstance(valid_actions, dict):
                raise ValueError(f'The valid actions provided for the interval {bounds} are not correct. Provided values are {valid_actions}')
    
    def get_valid_states(self, obs):
        value = obs[self.control_variable]
        for bounds, valid_actions in self.info.items():
            if value > bounds[0] and value <= bounds[1]:
                return valid_actions
        raise ValueError(f'The value {value} is not in any of the intervals provided in the config information.')
    
    def __call__(self, state):
        return self.get_valid_states(state)


class RLController(Controller):
    def __init__(
        self,
        name,
        controlled_components,
        sensors: Dict[str, str],
        agent: RLAgent | Dict,
        reward_function: RewardFunction | Dict,
        exploration_policy: ExplorationPolicy | Dict = {'type': 'epsilon-greedy', 'config_info': {}},
        valid_states_function: ValidStatesFunction | Dict | None =  None,
        predictors: Dict[str, str] = {},
        prediction_horizon: int = 1
    ):
        super().__init__(name, controlled_components, sensors, predictors)
        self.reward_function = reward_function if isinstance(reward_function, RewardFunction) else RewardFunction.make_reward(reward_function)
        if valid_states_function:
            self.valid_states_function = valid_states_function if isinstance(valid_states_function, ValidStatesFunction) else ValidStatesFunction(valid_states_function['control variable'], valid_states_function['config info'])
        else:
            self.valid_states_function = lambda x: self.action_space
        agent['config info']['exploration_policy'] = exploration_policy if isinstance(exploration_policy, ExplorationPolicy) else ExplorationPolicy.make_exploration_policy(exploration_policy)
        self.agent = agent if isinstance(agent, RLAgent) else RLAgent.make_agent(agent)
        self.horizon = prediction_horizon

    def initialize(self, ctx: InitContext):
        super().initialize(ctx)
        self.last_state = None
        self.last_action = None
        # Initialize reward function if needed
        self.reward_function.initialize(ctx)

    
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
        agent = RLAgent.make_agent(agent_type, **agent_kwargs)

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
                agent: RLAgent | Dict,
                sensors: Dict[str, str],
                predictors: Dict[str, str],
                actions: Dict[str, List[Any]],
                reward_function: RewardFunction | Dict, 
                exploration_policy: ExplorationPolicy | Dict,
                valid_states_function: ValidStatesFunction | Dict,
                state_discretizer: StateDiscretizer | Dict
                ):
        controlled_components = list(actions.keys())
        self.state_discretizer = state_discretizer if isinstance(state_discretizer, StateDiscretizer) else StateDiscretizer(state_discretizer)
        if isinstance(agent, dict):
            agent['actions'] = actions
        super().__init__(name = name, controlled_components=controlled_components, sensors=sensors, agent = agent, predictors=predictors, reward_function=reward_function, exploration_policy = exploration_policy, valid_states_function=valid_states_function)
        

    def preprocess_state(self):
        return self.state_discretizer.transform(obs = self.obs, predictions=self.predictions)


class QLearningController(RLControllerTabular):
    def __init__(self,
                name: str,
                sensors: Dict[str, str],
                actions: Dict[str, List[Any]],
                reward_function: RewardFunction | Dict, 
                state_discretizer: StateDiscretizer | Dict,
                exploration_policy: ExplorationPolicy | Dict = {},
                valid_states_function: ValidStatesFunction | Dict = {},
                agent_config_info: Dict = {},
                predictors: Dict[str, str] = {}):
        super().__init__(name = name, 
                         agent = {'type': "q_learning", 'config info': agent_config_info},
                         sensors=sensors, 
                         predictors=predictors, 
                         actions=actions, 
                         reward_function=reward_function, 
                         exploration_policy=exploration_policy,
                         valid_states_function = valid_states_function, 
                         state_discretizer=state_discretizer)

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
        if self.obs['storage tank temperature'] > 350:
            pass
        valid_actions = self.agent.map_to_action_space(self.valid_states_function(self.obs))
        action = self.agent.select_action(RL_state, valid_actions)
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
        valid_actions = self.agent.map_to_action_space(self.valid_states_function(self.obs))
        action = self.agent.select_action(RL_state, valid_actions)
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



