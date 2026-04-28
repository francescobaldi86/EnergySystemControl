from typing import Any, Dict, Literal, List, Tuple
from numpy import sin, cos, pi
from energy_system_control.controllers.base import Controller  # or move Controller to core/model.py
from energy_system_control.core.base_classes import InitContext
from energy_system_control.controllers.RL.reward_functions import RewardFunction
from energy_system_control.controllers.RL.agents import RLAgent
from energy_system_control.controllers.RL.exploration_policies import ExplorationPolicy
from energy_system_control.controllers.RL.discretizers import StateDiscretizer, Discretizer
from energy_system_control.sim.state import SimulationState

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
                return {k: v for k, v in valid_actions.items()}
                # return valid_actions
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
                state_discretizer: StateDiscretizer | Dict,
                include_hour_of_day: bool = False,
                include_day_of_the_year: bool = False,
                minimum_time_between_state_switches_h: Dict[str, float] | None = None
                ):
        controlled_components = list(actions.keys())
        self.state_discretizer = state_discretizer if isinstance(state_discretizer, StateDiscretizer) else StateDiscretizer(state_discretizer)
        if isinstance(agent, dict):
            agent['actions'] = actions
        self.minimum_time_between_state_switches = {k: v * 3600 for k, v in minimum_time_between_state_switches_h.items()} if minimum_time_between_state_switches_h is not None else None
        self.include_hour_of_day = include_hour_of_day
        self.include_day_of_the_year = include_day_of_the_year
        super().__init__(name = name, controlled_components=controlled_components, sensors=sensors, agent = agent, predictors=predictors, reward_function=reward_function, exploration_policy = exploration_policy, valid_states_function=valid_states_function)
        
    def initialize(self, ctx):
        super().initialize(ctx)
        if self.minimum_time_between_state_switches is not None:
            self.last_switch_time = {k: -3600 for k in self.controlled_component_names if k in self.minimum_time_between_state_switches.keys()}
            self.current_mode = {k: 0 for k in self.controlled_component_names if k in self.minimum_time_between_state_switches.keys()}
            for component_name in self.minimum_time_between_state_switches.keys():
                self.state_discretizer.discretizers[f'Time_since_last_switch_{component_name}'] = Discretizer(
                    vmin = 0.0,
                    vmax = self.minimum_time_between_state_switches[component_name] + 3600,
                    n_bins = 10
                )
        if self.include_hour_of_day:
            self.state_discretizer.discretizers[f'Hour of day (sin)'] = Discretizer(vmin = -1.0, vmax = 1.0, n_bins = 10)
            self.state_discretizer.discretizers[f'Hour of day (cos)'] = Discretizer(vmin = -1.0, vmax = 1.0, n_bins = 10)
        if self.include_day_of_the_year:
            self.state_discretizer.discretizers[f'Day of year (sin)'] = Discretizer(vmin = -1.0, vmax = 1.0, n_bins = 10)
            self.state_discretizer.discretizers[f'Day of year (cos)'] = Discretizer(vmin = -1.0, vmax = 1.0, n_bins = 10)

    def preprocess_state(self, state: SimulationState):
        if self.minimum_time_between_state_switches:
            for component, last_switch_time in self.last_switch_time.items():
                self.obs[f'Time_since_last_switch_{component}'] = state.time - last_switch_time
                self.obs[f'Current mode_{component}'] = int(self.current_mode[component])
        if self.include_hour_of_day:
            self.obs[f'Hour of day (sin)'] = sin(2 * pi * state.time / 86400)
            self.obs[f'Hour of day (cos)'] = cos(2 * pi * state.time / 86400)
        if self.include_day_of_the_year:
            self.obs[f'Day of year (sin)'] = sin(2 * pi * state.time / (365*86400))
            self.obs[f'Day of year (cos)'] = cos(2 * pi * state.time / (365*86400))
        return self.state_discretizer.transform(obs = self.obs, predictions=self.predictions)
    
    def update_switch_state(self, action, current_time):
        if self.minimum_time_between_state_switches is not None:
            for component_name, action in action.items():
                if action != self.current_mode[component_name]:
                    self.current_mode[component_name] = action
                    self.last_switch_time[component_name] = current_time

    def get_valid_states_based_on_minimum_switch_time(self, valid_actions):
        if self.minimum_time_between_state_switches is not None:
            for component, minimum_switch_time in self.minimum_time_between_state_switches.items():
                time_since_last_switch = self.obs[f'Time_since_last_switch_{component}']
                if time_since_last_switch < minimum_switch_time:
                    valid_actions[component] = [self.current_mode[component]]
        return valid_actions




class QLearningController(RLControllerTabular):
    def __init__(self,
                name: str,
                sensors: Dict[str, str],
                actions: Dict[str, List[Any]],
                reward_function: RewardFunction | Dict, 
                state_discretizer: StateDiscretizer | Dict,
                exploration_policy: ExplorationPolicy | Dict = {},
                valid_states_function: ValidStatesFunction | Dict = {},
                include_hour_of_day: bool = False,
                include_day_of_the_year: bool = False,
                agent_config_info: Dict = {},
                minimum_time_between_state_switches_h: Dict[str, float] | None = None,
                predictors: Dict[str, str] = {}):
        super().__init__(name = name, 
                         agent = {'type': "q_learning", 'config info': agent_config_info},
                         sensors=sensors, 
                         predictors=predictors, 
                         actions=actions, 
                         reward_function=reward_function, 
                         exploration_policy=exploration_policy,
                         valid_states_function = valid_states_function, 
                         include_day_of_the_year = include_day_of_the_year,
                         include_hour_of_day = include_hour_of_day,
                         state_discretizer=state_discretizer,
                         minimum_time_between_state_switches_h=minimum_time_between_state_switches_h)

    def get_action(self, state: SimulationState):
        RL_state = self.preprocess_state(state)
        reward = self.reward_function.compute(state)
        if self.last_state is not None and self.previous_action is not None:
            self.agent.update(
                state = state,
                last_state = self.last_state, 
                last_action = self.previous_action, 
                current_reward = reward,
                next_state = RL_state,
                next_action = None) # In Q-learning the next action is not needed for the update)
        valid_actions_temp = self.valid_states_function(self.obs)
        valid_actions_temp = self.get_valid_states_based_on_minimum_switch_time(valid_actions_temp)
        valid_actions = self.agent.map_to_action_space(valid_actions_temp)
        action = self.agent.select_action(RL_state, valid_actions, self.obs)
        self.update_switch_state(action, state.time)
        self.last_state = RL_state
        self.previous_action = action
        if state.time > 3600*24*50:
            pass
        return action
    
class SARSAController(RLControllerTabular):
    def __init__(self,
                name: str,
                sensors: Dict[str, str],
                predictors: Dict[str, str],
                actions: Dict[str, List[Any]],
                reward_function: RewardFunction, 
                state_discretizer: StateDiscretizer,
                include_hour_of_day: bool = False,
                include_day_of_the_year: bool = False,
                minimum_time_between_state_switches_h: Dict[str, float] | None = None,
                agent_kwargs: dict = {}):
        super().__init__(name = name, 
                         sensors=sensors, 
                         predictors=predictors, 
                         actions=actions, 
                         reward_function=reward_function, 
                         state_discretizer=state_discretizer, 
                         include_day_of_the_year = include_day_of_the_year,
                         include_hour_of_day = include_hour_of_day,
                         minimum_time_between_state_switches_h=minimum_time_between_state_switches_h,
                         agent_type="sarsa", 
                         agent_kwargs=agent_kwargs)

    def get_action(self, state: SimulationState):
        RL_state = self.preprocess_state()
        reward = self.reward_function.compute(state)
        valid_actions = self.agent.map_to_action_space(self.valid_states_function(self.obs))
        action = self.agent.select_action(RL_state, valid_actions, self.obs)
        if self.last_state is not None and self.previous_action is not None:
            self.agent.update(
                state = state,
                last_state = self.last_state, 
                last_action = self.previous_action, 
                current_reward = reward,
                next_state = RL_state,
                next_action = action) # In SARSA the next action is needed for the update)
        self.last_state = RL_state
        self.previous_action = action
        return action



