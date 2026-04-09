from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, List
import numpy as np
from collections import defaultdict
from energy_system_control.controllers.base import Controller  # or move Controller to core/model.py
from energy_system_control.core.base_classes import InitContext, Sensor
from energy_system_control.controllers.predictors import Predictor
from energy_system_control.helpers import C2K
from energy_system_control.sim.state import SimulationState

class RewardFunction(ABC):
    required_observations: set = set()
    required_actions: set = set()

    @abstractmethod
    def compute(self, state: SimulationState) -> float:
        pass

    def initialize(self, ctx: InitContext):
        pass

    def validate(self, obs_keys: set, action_keys: set):
        missing_obs = self.required_observations - obs_keys
        missing_actions = self.required_actions - action_keys

        if missing_obs:
            raise ValueError(f"Missing observations: {missing_obs}")
        if missing_actions:
            raise ValueError(f"Missing actions: {missing_actions}")
    
    def make_reward(config: dict):
        if config["type"] == "composite":
            return CompositeReward([
                CompositeReward.make_reward(sub) for sub in config["components"]
            ])

        cls = REWARD_REGISTRY[config["type"]]
        return cls(**config.get("kwargs", {}))
        
class CompositeReward(RewardFunction):
    def __init__(self, rewards: list):
        self.rewards = rewards

        # aggregate requirements
        self.required_observations = set().union(
            *(r.required_observations for r in rewards)
        )
        self.required_actions = set().union(
            *(r.required_actions for r in rewards)
        )

    def initialize(self, ctx: InitContext):
        for r in self.rewards:
            r.initialize(ctx)

    def compute(self, state: SimulationState):
        return sum(r.compute(state) for r in self.rewards)
    
    def make_reward(config: dict) -> RewardFunction:
        cls = REWARD_REGISTRY[config["type"]]
        return cls(**config.get("kwargs", {}))



class TemperatureTrackingReward(RewardFunction):

    def __init__(self, sensor_name, target, weight=1.0):
        self.target = C2K(target)
        self.weight = weight
        self.sensor_name = sensor_name

    def initialize(self, ctx: InitContext):
        self.sensor = ctx.environment.sensors[self.sensor_name]

    def compute(self, state: SimulationState):
        error = self.sensor.get_measurement() - self.target
        return -self.weight * error**2

class TemperatureMinMaxReward(RewardFunction):
    def __init__(self, sensor_name, min_temp, max_temp, weight=1.0):
        self.min_temp = C2K(min_temp)
        self.max_temp = C2K(max_temp)
        self.weight = weight
        self.sensor_name = sensor_name

    def initialize(self, ctx: InitContext):
        self.sensor = ctx.environment.sensors[self.sensor_name]

    def compute(self, state: SimulationState):
        error_max = max(self.sensor.get_measurement() - self.max_temp, 0)
        error_min = max(self.min_temp - self.sensor.get_measurement(), 0)
        return -self.weight * (error_max + error_min)**2

class ComponentCostReward(RewardFunction):
    default_power_kW: float
    energy_cost_per_kWh: float
    def __init__(self, default_power_kW: float, energy_cost_per_kWh: float, controller_name: str, controlled_component_name: str, weight: float = 1) :
        self.default_power_kW = default_power_kW
        self.energy_cost_per_kWh = energy_cost_per_kWh
        self.weight = weight
        self.controlled_component_name = controlled_component_name
        self.controller_name = controller_name

    def initialize(self, ctx):
        self.controller = ctx.environment.controllers[self.controller_name]
    
    def compute(self, state: SimulationState):
        if self.controller.last_action is None:
            return 0
        else:
            return -self.weight * self.energy_cost_per_kWh * self.default_power_kW * self.controller.last_action[self.controlled_component_name] * state.time_step / 3600
    

class EnergyCostReward(RewardFunction):
    required_observations = {"energy_exchange"}

    def __init__(self, cost_components: List[Dict[str, Any]]):
        # The cost_components is a list of dicts, each dict specifying a component of the cost, e.g.:
        # [{"component": "electricity_grid", "sensor": "grid_power_sensor"]
        for component in cost_components:
            if "component" not in component or "sensor" not in component:
                raise ValueError("Each cost component must have 'component' and 'sensor' keys")
        self.cost_components_raw = cost_components
        self.cost_components = None  # to be actualized in initialize

    def initialize(self, ctx: InitContext):
        self.cost_components = []
        for comp in self.cost_components_raw:
            self.cost_components.append({
                "sensor": ctx.environment.sensors[comp["sensor"]],
                "purchase_cost": ctx.environment.components[comp['component']].cost_of_energy_purchased,
                "feed_in_revenue": ctx.environment.components[comp['component']].value_of_energy_sold
            })

    def compute(self, state: SimulationState):
        total_cost = 0.0
        for comp in self.cost_components:
            energy_exchange = comp["sensor"].get_measurement()  # positive for purchase, negative for feed-in
            if energy_exchange > 0:  # purchasing energy
                total_cost += energy_exchange * comp["purchase_cost"]
            else:  # selling energy
                total_cost += energy_exchange * comp["feed_in_revenue"]  # note that energy_exchange is negative here
        return -total_cost

REWARD_REGISTRY = {
    "temperature_tracking": TemperatureTrackingReward,
    "energy_cost": EnergyCostReward,
    "energy_cost_component": ComponentCostReward,
    "temperature_minmax": TemperatureMinMaxReward,
}

