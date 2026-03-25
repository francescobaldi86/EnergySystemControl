from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, List
import numpy as np
from collections import defaultdict
from energy_system_control.controllers.base import Controller  # or move Controller to core/model.py
from energy_system_control.core.base_classes import InitContext, Sensor
from energy_system_control.controllers.predictors import Predictor


class RewardFunction(ABC):
    required_observations: set = set()
    required_actions: set = set()

    @abstractmethod
    def compute(self, state, action, next_state, **kwargs) -> float:
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

    def compute(self, state, action, next_state, **kwargs):
        return sum(r.compute(state, action, next_state, **kwargs) for r in self.rewards)

    @staticmethod
    def make_reward(config: dict) -> RewardFunction:
        if config["type"] == "composite":
            return CompositeReward([
                CompositeReward.make_reward(sub) for sub in config["components"]
            ])

        cls = REWARD_REGISTRY[config["type"]]
        return cls(**config.get("kwargs", {}))


class TemperatureTrackingReward(RewardFunction):

    def __init__(self, sensor_name, target, weight=1.0):
        self.target = target
        self.weight = weight
        self.sensor_name = sensor_name

    def initialize(self, ctx: InitContext):
        self.sensor = ctx.environment.sensors[self.sensor_name]

    def compute(self, state, action, next_state, **kwargs):
        error = self.sensor.get_measurement() - self.target
        return -self.weight * error**2
    
class EnergyCostReward(RewardFunction):
    required_observations = {"energy_exchange"}

    def __init__(self, cost_components: List[Dict[str, Any]]):
        # The cost_components is a list of dicts, each dict specifying a component of the cost, e.g.:
        # [{"component": "electricity_grid", "sensor": "grid_power_sensor"]
        for component in cost_components:
            if "component" not in component or "sensor" not in component or "purchase_cost" not in component or "feed_in_revenue" not in component:
                raise ValueError("Each cost component must have 'component' and 'sensor' keys")
        self.cost_components_raw = cost_components
        self.cost_components = None  # to be actualized in initialize

    def initialize(self, ctx: InitContext):
        self.cost_components = []
        for comp in self.cost_components_raw:
            sensor = ctx.sensors[comp["sensor"]]
            self.cost_components.append({
                "sensor": ctx.environment.sensors[comp["sensor"]],
                "purchase_cost": ctx.environment.components[comp['component']].cost_of_energy_purchased,
                "feed_in_revenue": ctx.environment.components[comp['component']].value_of_energy_sold
            })

    def compute(self, state, action, next_state, **kwargs):
        total_cost = 0.0
        for comp in self.cost_components:
            energy_exchange = comp["sensor"].get_value(kwargs['env'], kwargs['env'].state)
            if energy_exchange > 0:  # purchasing energy
                total_cost += energy_exchange * comp["purchase_cost"]
            else:  # selling energy
                total_cost += energy_exchange * comp["feed_in_revenue"]  # note that energy_exchange is negative here
        return -total_cost

class EnergyPenaltyReward(RewardFunction):
    required_actions = {"heater"}

    def __init__(self, weight=0.1):
        self.weight = weight

    def compute(self, state, action, next_state, **kwargs):
        return -self.weight * abs(action["heater"])


REWARD_REGISTRY = {
    "temperature_tracking": TemperatureTrackingReward,
    "energy_penalty": EnergyPenaltyReward,
    "energy_cost": EnergyCostReward,
}