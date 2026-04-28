from abc import ABC, abstractmethod
from typing import Any, Dict, List
from energy_system_control.core.base_classes import InitContext
from energy_system_control.helpers import C2K
from energy_system_control.sim.state import SimulationState

class RewardFunction(ABC):
    """Abstract base class for reward functions in reinforcement learning.
    
    Reward functions define the objective for the RL agent to optimize. They compute
    a scalar reward signal based on the current simulation state.
    
    Attributes:
        required_observations (set): Set of observation keys required by this reward function.
        required_actions (set): Set of action keys required by this reward function.
    """
    required_observations: set = set()
    required_actions: set = set()

    @abstractmethod
    def compute(self, state: SimulationState) -> float:
        """Compute the reward signal.
        
        Args:
            state (SimulationState): Current simulation state.
            
        Returns:
            float: The computed reward value.
        """
        pass

    def initialize(self, ctx: InitContext):
        """Initialize the reward function with environment context.
        
        Called once before simulation starts to resolve component references
        and perform any setup operations.
        
        Args:
            ctx (InitContext): Initialization context containing environment and components.
        """
        pass

    def validate(self, obs_keys: set, action_keys: set):
        """Validate that required observations and actions are available.
        
        Args:
            obs_keys (set): Set of available observation keys.
            action_keys (set): Set of available action keys.
            
        Raises:
            ValueError: If required observations or actions are missing.
        """
        missing_obs = self.required_observations - obs_keys
        missing_actions = self.required_actions - action_keys

        if missing_obs:
            raise ValueError(f"Missing observations: {missing_obs}")
        if missing_actions:
            raise ValueError(f"Missing actions: {missing_actions}")
    
    def make_reward(config: dict):
        """Factory method to create a reward function from configuration.
        
        Args:
            config (dict): Configuration dictionary with keys:
                - 'type': Reward function type (must be in REWARD_REGISTRY)
                - 'components': (optional) List of sub-configs for composite rewards
                - 'kwargs': (optional) Keyword arguments to pass to the reward class
                
        Returns:
            RewardFunction: Instantiated reward function.
            
        Raises:
            KeyError: If the specified reward type is not in REWARD_REGISTRY.
        """
        if config["type"] == "composite":
            return CompositeReward([
                CompositeReward.make_reward(sub) for sub in config["components"]
            ])

        cls = REWARD_REGISTRY[config["type"]]
        return cls(**config.get("kwargs", {}))
        
class CompositeReward(RewardFunction):
    """Combines multiple reward functions into a single composite reward.
    
    The composite reward sums the individual reward signals. Requirements
    (observations and actions) are aggregated from all component rewards.
    
    Attributes:
        rewards (list): List of RewardFunction instances to combine.
    """
    def __init__(self, rewards: list):
        """Initialize composite reward with list of reward functions.
        
        Args:
            rewards (list): List of RewardFunction instances to combine.
        """
        self.rewards = rewards

        # aggregate requirements
        self.required_observations = set().union(
            *(r.required_observations for r in rewards)
        )
        self.required_actions = set().union(
            *(r.required_actions for r in rewards)
        )

    def initialize(self, ctx: InitContext):
        """Initialize all component reward functions.
        
        Args:
            ctx (InitContext): Initialization context.
        """
        for r in self.rewards:
            r.initialize(ctx)

    def compute(self, state: SimulationState) -> float:
        """Compute composite reward as sum of component rewards.
        
        Args:
            state (SimulationState): Current simulation state.
            
        Returns:
            float: Sum of all component reward values.
        """
        return sum(r.compute(state) for r in self.rewards)
    
    def make_reward(config: dict) -> RewardFunction:
        """Factory method to create a single reward function from configuration.
        
        Args:
            config (dict): Configuration dictionary with keys:
                - 'type': Reward function type (must be in REWARD_REGISTRY)
                - 'kwargs': (optional) Keyword arguments to pass to the reward class
                
        Returns:
            RewardFunction: Instantiated reward function.
        """
        cls = REWARD_REGISTRY[config["type"]]
        return cls(**config.get("kwargs", {}))



class TemperatureTrackingReward(RewardFunction):
    """Reward for tracking a target temperature.
    
    Provides negative reward proportional to the squared temperature error.
    Higher weight increases the penalty for deviations from the target.
    
    Attributes:
        target (float): Target temperature in Kelvin.
        weight (float): Weight factor for the reward penalty.
        sensor_name (str): Name of the temperature sensor to monitor.
        sensor: Reference to the temperature sensor (set during initialization).
    """

    def __init__(self, sensor_name: str, target: float, weight: float = 1.0):
        """Initialize temperature tracking reward.
        
        Args:
            sensor_name (str): Name of the temperature sensor.
            target (float): Target temperature in Celsius (will be converted to Kelvin).
            weight (float, optional): Weight factor for the penalty. Defaults to 1.0.
        """
        self.target = C2K(target)
        self.weight = weight
        self.sensor_name = sensor_name

    def initialize(self, ctx: InitContext):
        """Initialize sensor reference.
        
        Args:
            ctx (InitContext): Initialization context.
        """
        self.sensor = ctx.environment.sensors[self.sensor_name]

    def compute(self, state: SimulationState) -> float:
        """Compute reward based on temperature tracking error.
        
        Reward = -weight * (measured_temperature - target_temperature)^2
        
        Args:
            state (SimulationState): Current simulation state.
            
        Returns:
            float: Reward value (negative, penalizing deviation from target).
        """
        error = self.sensor.get_measurement() - self.target
        return -self.weight * error**2

class TemperatureMinMaxReward(RewardFunction):
    """Reward for keeping temperature within a specified range.
    
    Provides negative reward when temperature exceeds the minimum or maximum bounds.
    Temperature within the range receives zero penalty.
    
    Attributes:
        min_temp (float): Minimum allowed temperature in Kelvin.
        max_temp (float): Maximum allowed temperature in Kelvin.
        weight (float): Weight factor for the reward penalty.
        sensor_name (str): Name of the temperature sensor to monitor.
        sensor: Reference to the temperature sensor (set during initialization).
    """
    def __init__(self, sensor_name: str, min_temp: float, max_temp: float, weight: float = 1.0):
        """Initialize temperature min/max reward.
        
        Args:
            sensor_name (str): Name of the temperature sensor.
            min_temp (float): Minimum temperature in Celsius (will be converted to Kelvin).
            max_temp (float): Maximum temperature in Celsius (will be converted to Kelvin).
            weight (float, optional): Weight factor for the penalty. Defaults to 1.0.
        """
        self.min_temp = C2K(min_temp)
        self.max_temp = C2K(max_temp)
        self.weight = weight
        self.sensor_name = sensor_name

    def initialize(self, ctx: InitContext):
        """Initialize sensor reference.
        
        Args:
            ctx (InitContext): Initialization context.
        """
        self.sensor = ctx.environment.sensors[self.sensor_name]

    def compute(self, state: SimulationState) -> float:
        """Compute reward based on temperature bounds violation.
        
        Reward = -weight * (max_violation + min_violation)^2
        where violations are positive only when bounds are exceeded.
        
        Args:
            state (SimulationState): Current simulation state.
            
        Returns:
            float: Reward value (negative for out-of-bounds temperature, zero otherwise).
        """
        temperature = self.sensor.get_measurement()
        error_max = max(temperature - self.max_temp, 0)
        error_min = max(self.min_temp - temperature, 0)
        reward = -self.weight * state.time_step / 3_600 * (error_max + error_min)**2  # We multiply by the time step for coherence with energy-related rewards. This way, the balance between rewards does not depend on the time step
        return reward

class ComponentCostReward(RewardFunction):
    """Reward for minimizing energy cost of a controlled component.
    
    Computes the cost of operating a specific component based on its power output
    and energy cost. Provides negative reward proportional to energy consumption cost.
    
    Attributes:
        default_power_kW (float): Default power rating of the component in kW.
        energy_cost_per_kWh (float): Cost of energy in currency units per kWh.
        weight (float): Weight factor for the reward penalty.
        controlled_component_name (str): Name of the component being controlled.
        controller_name (str): Name of the controller managing the component.
        controller: Reference to the controller (set during initialization).
    """
    default_power_kW: float
    energy_cost_per_kWh: float
    
    def __init__(self, default_power_kW: float, energy_cost_per_kWh: float, controller_name: str, controlled_component_name: str, weight: float = 1):
        """Initialize component cost reward.
        
        Args:
            default_power_kW (float): Default power rating in kW.
            energy_cost_per_kWh (float): Energy cost in currency units per kWh.
            controller_name (str): Name of the controller.
            controlled_component_name (str): Name of the controlled component.
            weight (float, optional): Weight factor for the penalty. Defaults to 1.
        """
        self.default_power_kW = default_power_kW
        self.energy_cost_per_kWh = energy_cost_per_kWh
        self.weight = weight
        self.controlled_component_name = controlled_component_name
        self.controller_name = controller_name

    def initialize(self, ctx: InitContext):
        """Initialize controller reference.
        
        Args:
            ctx (InitContext): Initialization context.
        """
        self.controller = ctx.environment.controllers[self.controller_name]
    
    def compute(self, state: SimulationState) -> float:
        """Compute reward based on component operating cost.
        
        Reward = -weight * energy_cost_per_kWh * default_power_kW * action_value * time_step / 3600
        
        Args:
            state (SimulationState): Current simulation state.
            
        Returns:
            float: Reward value (negative, penalizing component operation). Returns 0 if no action was taken.
        """
        if self.controller.last_action is None:
            return 0
        else:
            reward = -self.weight * self.energy_cost_per_kWh * self.default_power_kW * self.controller.last_action[self.controlled_component_name] * state.time_step / 3600
            return reward
    

class EnergyCostReward(RewardFunction):
    """Reward for minimizing total energy cost from multiple energy sources.
    
    Tracks energy purchases and sales across multiple components (e.g., grids, storage),
    applying purchase costs for consumption and revenue credits for feed-in.
    
    Attributes:
        required_observations (set): Contains 'energy_exchange' required for this reward.
        cost_components (list): List of cost component dictionaries with sensor references
                               and cost/revenue values (set during initialization).
    """
    required_observations = {"energy_exchange"}

    def __init__(self, cost_components: List[Dict[str, Any]]):
        """Initialize energy cost reward with multiple cost components.
        
        Args:
            cost_components (List[Dict[str, Any]]): List of cost component specifications.
                Each dict must contain:
                - 'component': Name of the component (e.g., 'electricity_grid')
                - 'sensor': Name of the energy exchange sensor for that component
                
                Example:
                    [{'component': 'electricity_grid', 'sensor': 'grid_power_sensor'}]
                
        Raises:
            ValueError: If any cost component is missing 'component' or 'sensor' keys.
        """
        # The cost_components is a list of dicts, each dict specifying a component of the cost, e.g.:
        # [{"component": "electricity_grid", "sensor": "grid_power_sensor"]]
        for component in cost_components:
            if "component" not in component or "sensor" not in component:
                raise ValueError("Each cost component must have 'component' and 'sensor' keys")
        self.cost_components_raw = cost_components
        self.cost_components = None  # to be actualized in initialize

    def initialize(self, ctx: InitContext):
        """Initialize sensor and cost/revenue references for all components.
        
        Args:
            ctx (InitContext): Initialization context.
        """
        self.cost_components = []
        for comp in self.cost_components_raw:
            self.cost_components.append({
                "sensor": ctx.environment.sensors[comp["sensor"]],
                "purchase_cost": ctx.environment.components[comp['component']].cost_of_energy_purchased,
                "feed_in_revenue": ctx.environment.components[comp['component']].value_of_energy_sold
            })

    def compute(self, state: SimulationState) -> float:
        """Compute total energy cost across all components.
        
        Positive energy exchange (purchase) is charged at purchase_cost.
        Negative energy exchange (feed-in) is credited at feed_in_revenue.
        
        Reward = -total_cost (negative because we want to minimize cost)
        
        Args:
            state (SimulationState): Current simulation state.
            
        Returns:
            float: Reward value (negative, penalizing energy costs).
        """
        total_cost = 0.0
        for comp in self.cost_components:
            energy_exchange = comp["sensor"].get_measurement()  # positive for purchase, negative for feed-in
            if energy_exchange > 0:  # purchasing energy
                total_cost += energy_exchange * comp["purchase_cost"] * state.time_step / 3_600 # EUR/kWh * kW * s * h/s 
            else:  # selling energy
                total_cost -= energy_exchange * comp["feed_in_revenue"] * state.time_step / 3_600  # note that energy_exchange is negative here
        return -total_cost

REWARD_REGISTRY = {
    "temperature_tracking": TemperatureTrackingReward,
    "energy_cost": EnergyCostReward,
    "energy_cost_component": ComponentCostReward,
    "temperature_minmax": TemperatureMinMaxReward,
}

