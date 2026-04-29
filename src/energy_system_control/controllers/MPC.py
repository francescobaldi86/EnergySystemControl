from __future__ import annotations
from dataclasses import dataclass
from abc import abstractmethod
from typing import Dict, Any, Sequence, Optional, Tuple, Literal, List
import numpy as np
import pandas as pd
import cvxpy as cp
import math
from energy_system_control.controllers.base import Controller
from energy_system_control.controllers.predictors import Predictor
from energy_system_control.core.base_classes import InitContext
from energy_system_control import HotWaterDemand, HotWaterStorage, ElectricityDemand, PVpanel, Battery, Inverter, ColdWaterGrid, HeatPump, ResistanceHeater, ColdWaterGrid, ElectricityGrid, Inverter, ResistanceHeater
from energy_system_control.sim.state import SimulationState
from energy_system_control.helpers import find_object_of_type
from energy_system_control.constants import WATER


SolverName = Literal["OSQP", "HIGHS"]

class MPCController(Controller):
    """
    Generic linear MPC base controller.

    Subclasses must implement:
      - get_model(...)
      - get_current_state(...)
      - get_disturbance_forecast(...)  (or return None)
      - build_stage_constraints(...)
      - build_objective(...)
      - action_from_u0(...)
    """

    def __init__(
        self,
        name: str,
        controlled_components: List[str],
        sensors: Dict[str, str],
        predictors: Dict[str, str],
        horizon: float,
        solver: SolverName = "HIGHS",
    ):
        super().__init__(name, controlled_components, sensors, predictors)
        if horizon <= 0:
            raise ValueError("horizon_steps must be > 0")
        self.horizon = horizon 
        self.solver = solver

    def initialize(self, ctx: InitContext):
        super().initialize(ctx)


class MPCController_HybridDHW(MPCController):
    """
    This class extends the MPCController class to provide an helpful framework for an MPC Controller of a system
    that implements a Hybrid DHW system. Components are:
    - Heat source (compulsory, Boiler or heat pump)
    - Heat demand (compulsory)
    - PV panels (compulsory)
    - Inverter (compulsory)
    - Grid (compulsory)
    - Battery (optional)
    - Thermal solar panels (optional)
    """

    def __init__(self,
                    name: str,
                    horizon: float,
                    heat_pump_name: str,
                    storage_temperature_sensor: str | None = None,
                    battery_SOC_sensor: str | None = None,
                    PV_power_predictor_name: str | None = None,
                    heat_demand_predictor_name: str | None = None,
                    electricity_demand_predictor_name: str | None = None,
                    bounds_SOC: Tuple[float, float] = (0.3, 0.9),
                    bounds_temperature: Tuple[float, float] = (313.15, 353.15),
                    cost_of_temperature_violation: float = 1000.0
                    ):
        self.PV_power_predictor_name = PV_power_predictor_name
        self.heat_demand_predictor_name = heat_demand_predictor_name
        self.electricity_demand_predictor_name = electricity_demand_predictor_name
        sensors = {"temperature_storage": storage_temperature_sensor, "soc_battery": battery_SOC_sensor}
        sensors = {k: v for k, v in sensors.items() if v is not None}
        predictors = {"pv_power": PV_power_predictor_name, "heat_demand": heat_demand_predictor_name, "electricity_demand": electricity_demand_predictor_name}
        predictors = {k: v for k, v in predictors.items() if v is not None}
        self.bounds_SOC = bounds_SOC
        self.bounds_temperature = bounds_temperature
        self.cost_of_temperature_violation = cost_of_temperature_violation
        super().__init__(
            name = name,
            controlled_components = [heat_pump_name],
            sensors = sensors,
            predictors = predictors,
            horizon = horizon,
            solver = cp.HIGHS)
        
    def load_controlled_components(self, components):
        self.heat_pump = find_object_of_type(HeatPump, components)
        self.pv_panel = find_object_of_type(PVpanel, components)
        self.battery = find_object_of_type(Battery, components)
        self.inverter = find_object_of_type(Inverter, components)
        self.electricity_grid = find_object_of_type(ElectricityGrid, components)
        self.water_grid = find_object_of_type(ColdWaterGrid, components)
        self.hot_water_storage = find_object_of_type(HotWaterStorage, components)
        self.resistance_heater = find_object_of_type(ResistanceHeater, components)
        self.dhw_demand = find_object_of_type(HotWaterDemand, components)
        self.electricity_demand = find_object_of_type(ElectricityDemand, components)
        self.controlled_components = {}
        if self.heat_pump is not None:
            self.controlled_components[self.heat_pump.name] = self.heat_pump
        if self.resistance_heater is not None:
            self.controlled_components[self.resistance_heater.name] = self.resistance_heater
        # Check consistency between predictors and components
        if self.pv_panel and self.PV_power_predictor_name is None:
            raise(BaseException, 'PV panel is present but no PV predictor is provided')
        if self.dhw_demand and self.heat_demand_predictor_name is None:
            raise(BaseException, 'Heat demand is present but no heat demand predictor is provided')
        if self.electricity_demand and self.electricity_demand_predictor_name is None:
            raise(BaseException, 'Electricity demand is present but no electricity demand predictor is provided')
        
    def initialize(self, ctx: InitContext):
        super().initialize(ctx)
        # The optimization problem is initialized at the beginning, and then updated using parameters
        # Connecting to the predictors:
        self.PV_power_predictor = ctx.environment.predictors[self.PV_power_predictor_name] if self.PV_power_predictor_name is not None else None
        self.heat_demand_predictor = ctx.environment.predictors[self.heat_demand_predictor_name] if self.heat_demand_predictor_name is not None else None
        self.electricity_demand_predictor = ctx.environment.predictors[self.electricity_demand_predictor_name] if self.electricity_demand_predictor_name is not None else None
        self.predictors = [self.PV_power_predictor, self.heat_demand_predictor, self.electricity_demand_predictor]
        self.predictors = [predictor for predictor in self.predictors if predictor is not None]
        # Getting first obs
        self.get_obs(ctx.environment, ctx.state)
        # Declaring variables
        self.problem = MPCProblem()
        problem = self.problem
        
        # Calculating first parameters
        TIME_STEP = ctx.state.time_step / 3600
        TIME_HORIZON = int(self.horizon // TIME_STEP)
        problem.variables = {
            'temperature_hot_water_storage': cp.Variable(TIME_HORIZON),
            'energy_battery': cp.Variable(TIME_HORIZON),
            'power_from_grid': cp.Variable(TIME_HORIZON),
            'power_to_grid': cp.Variable(TIME_HORIZON),
            'power_to_battery': cp.Variable(TIME_HORIZON),
            'power_from_battery': cp.Variable(TIME_HORIZON),
            'power_heat_pump': cp.Variable(TIME_HORIZON),
            'status_heat_pump': cp.Variable(TIME_HORIZON, boolean=True),
            'power_resistance': cp.Variable(TIME_HORIZON),
            'status_resistance': cp.Variable(TIME_HORIZON, boolean=True),
            'slack_temperature': cp.Variable(TIME_HORIZON)}
        problem.constant_parameters = {
            'DISPERSION_COEFFICIENT': self.hot_water_storage.convection_coefficient_losses,
            'DISPERSION_SURFACE': self.hot_water_storage.surface,
            'HEAT_CAPACITY_WATER': WATER.cp / 3600,
            'MASS_STORAGE': self.hot_water_storage.volume * WATER.rho,
            'EFFICIENCY_EES_CHA': self.battery.charger.efficiency_charge * self.inverter.converter.efficiency,
            'EFFICIENCY_EES_DIS': self.battery.charger.efficiency_discharge * self.inverter.converter.efficiency,
            'POWER_HP_EL': self.heat_pump.Qdot_design if self.heat_pump else 0.0,
            'POWER_HP_TH': self.heat_pump.Qdot_design * self.heat_pump.COP_design if self.heat_pump else 0.0,
            'POWER_RESISTANCE_EL': self.resistance_heater.power if self.resistance_heater else 0.0,
            'POWER_RESISTANCE_TH': self.resistance_heater.power if self.resistance_heater else 0.0,
            'ENERGY_COST': self.electricity_grid.cost_of_energy_purchased,
            'ENERGY_VALUE': self.electricity_grid.value_of_energy_sold,
            'POWER_BATTERY_MAX_CHA': abs(self.battery.charger.max_charging_power),
            'POWER_BATTERY_MAX_DIS': abs(self.battery.charger.max_discharging_power),
            'ENERGY_BATTERY_MAX': self.battery.battery_pack.max_capacity * self.bounds_SOC[1] / 3600 if self.battery else 0.0,
            'ENERGY_BATTERY_MIN': self.battery.battery_pack.max_capacity * self.bounds_SOC[0] / 3600 if self.battery else 0.0,
            'CAPACITY_BATTERY': self.battery.battery_pack.max_capacity / 3600 if self.battery else 0.0,
            'TEMPERATURE_STORAGE_MAX': self.bounds_temperature[1],
            'TEMPERATURE_STORAGE_MIN': self.bounds_temperature[0],
            'COST_OF_TEMPERATURE_VIOLATION': self.cost_of_temperature_violation
        }
        problem.variable_parameters = {
            'POWER_PV': cp.Parameter(TIME_HORIZON),
            'EL_DEMAND': cp.Parameter(TIME_HORIZON),
            'TH_DEMAND': cp.Parameter(TIME_HORIZON),
            'TEMPERATURE_STORAGE_0': cp.Parameter(),
            'SOC_0': cp.Parameter(),
            'B_TES': cp.Parameter(TIME_HORIZON-1)
        }
        # Matrices and vectors for thermal storage temperature update
        A_TES = np.zeros(shape=[TIME_HORIZON-1,TIME_HORIZON])
        A_TES_main = np.array([problem.constant_parameters['DISPERSION_COEFFICIENT'] * problem.constant_parameters['DISPERSION_SURFACE'] * TIME_STEP / (problem.constant_parameters['MASS_STORAGE'] * WATER.cp) - 1] * TIME_HORIZON)
        A_TES_above = np.array([1] * (TIME_HORIZON-1))
        A_TES = np.diag(A_TES_main) + np.diag(A_TES_above, k=1)
        problem.constant_parameters['A_TES'] = A_TES = A_TES[:-1, :]
        problem.constant_parameters['B1_TES'] = TIME_STEP / (problem.constant_parameters['MASS_STORAGE'] * problem.constant_parameters['HEAT_CAPACITY_WATER'])

        # Matrices and vectors for electrical storage energy update
        A_EES = np.zeros(shape=[TIME_HORIZON-1,TIME_HORIZON])
        A_EES = np.diag(np.ones(TIME_HORIZON-1), k=1) - np.diag(np.ones(TIME_HORIZON)) 
        problem.constant_parameters['A_EES'] = A_EES = A_EES[:-1, :]
        problem.constant_parameters['B1_EES_CHA'] = TIME_STEP * problem.constant_parameters['EFFICIENCY_EES_CHA']
        problem.constant_parameters['B1_EES_DIS'] = TIME_STEP / problem.constant_parameters['EFFICIENCY_EES_DIS']

        # Update problem parameters, with values available at start
        self.update_problem_parameters(state = ctx.state)
        constraints = [
            problem.variables['power_heat_pump'] == problem.variables['status_heat_pump'] * problem.constant_parameters['POWER_HP_EL'],
            problem.variables['power_resistance'] == problem.variables['status_resistance'] * problem.constant_parameters['POWER_RESISTANCE_EL'],
            problem.variables['power_from_grid'] + problem.variable_parameters['POWER_PV'] + problem.variables['power_from_battery'] - problem.variables['power_to_grid'] - problem.variable_parameters['EL_DEMAND'] - problem.variables['power_to_battery'] - problem.variables['power_heat_pump'] - problem.variables['power_resistance'] == 0,
            problem.variables['temperature_hot_water_storage'][0] == problem.variable_parameters['TEMPERATURE_STORAGE_0'],
            problem.variables['energy_battery'][0] == problem.variable_parameters['SOC_0'] * problem.constant_parameters['CAPACITY_BATTERY'],
            A_TES @ problem.variables['temperature_hot_water_storage'] - problem.variables['status_heat_pump'][:-1] * problem.constant_parameters['POWER_HP_TH'] * problem.constant_parameters['B1_TES'] - problem.variables['status_resistance'][:-1] * problem.constant_parameters['POWER_RESISTANCE_TH'] * problem.constant_parameters['B1_TES'] + problem.variable_parameters['B_TES'] == 0,
            A_EES @ problem.variables['energy_battery'] - problem.variables['power_to_battery'][:-1] * problem.constant_parameters['B1_EES_CHA'] + problem.variables['power_from_battery'][:-1] * problem.constant_parameters['B1_EES_DIS'] == 0,
            problem.variables['temperature_hot_water_storage'] <= problem.constant_parameters['TEMPERATURE_STORAGE_MAX'],
            problem.variables['temperature_hot_water_storage'] >= problem.constant_parameters ['TEMPERATURE_STORAGE_MIN'] - problem.variables['slack_temperature'],
            problem.variables['temperature_hot_water_storage'][0] >= problem.variable_parameters['TEMPERATURE_STORAGE_0'],
            problem.variables['energy_battery'] <= problem.constant_parameters['ENERGY_BATTERY_MAX'],
            problem.variables['energy_battery'] >= problem.constant_parameters['ENERGY_BATTERY_MIN'],
            problem.variables['energy_battery'][-1] >= problem.variable_parameters['SOC_0'] * problem.constant_parameters['ENERGY_BATTERY_MAX'],
            problem.variables['power_to_battery'] <= problem.constant_parameters['POWER_BATTERY_MAX_CHA'],
            problem.variables['power_from_battery'] <= problem.constant_parameters['POWER_BATTERY_MAX_DIS'],
            problem.variables['power_to_battery'] >= 0,
            problem.variables['power_from_battery'] >= 0,
            problem.variables['power_to_grid'] >= 0,
            problem.variables['power_from_grid'] >= 0,
            problem.variables['slack_temperature'] >= 0,
        ]
        objective = cp.Minimize(problem.constant_parameters['ENERGY_COST'] *np.ones([1,TIME_HORIZON]) @ problem.variables['power_from_grid'] - problem.constant_parameters['ENERGY_VALUE']*np.ones([1,TIME_HORIZON]) @ problem.variables['power_to_grid'] + problem.constant_parameters['COST_OF_TEMPERATURE_VIOLATION'] * cp.sum(problem.variables['slack_temperature']))
        self.problem.problem = cp.Problem(objective, constraints)
    
    def get_action(self, state):
        action = {}
        self.update_problem_parameters(state)
        self.problem.problem.solve(self.solver)
        if self.problem.problem.status not in {'optimal', 'optimal_inaccurate'}:
            raise ValueError(f'The optimisation solver could not find an optimal solution at time step {state.time_id} at simulation time {state.time}')
        if self.heat_pump:
            temp = self.problem.variables['status_heat_pump'].value[0]
            temp_rounded = round(temp,0)
            if math.isclose(temp, temp_rounded, abs_tol=1e-5):
                action[self.heat_pump.name] = int(temp_rounded)
            else:
                raise ValueError(f'The heat pump status is not an integer at time step {state.time_id} at simulation time {state.time}. Optimal heat pump status is {temp}')
        if self.resistance_heater:
            temp = self.problem.variables['status_resistance'].value[0]
            temp_rounded = round(temp,0)
            if math.isclose(temp, temp_rounded, abs_tol=1e-5):
                action[self.resistance_heater.name] = int(temp_rounded)
            else:
                raise ValueError(f'The resistance status is not an integer at time step {state.time_id} at simulation time {state.time}. Optimal resistance status is {temp}')
        self.previous_action = action
        return action
    
    def get_obs(self, environment, state) -> Dict[str, Any]:
        self.obs = {}
        for var_name, sensor in self.sensors.items():
            if sensor:
                self.obs[var_name] = sensor.get_measurement()
        return self.obs


    def update_problem_parameters(self, state: SimulationState):
        # Method that updates problem parameters depending on the current state of the simulation
        param = self.problem.variable_parameters
        # Prediction of future heat demand
        param['TH_DEMAND'].value = self.safe_predict(self.heat_demand_predictor, state)[:param['TH_DEMAND'].size]
        param['POWER_PV'].value = self.safe_predict(self.PV_power_predictor, state)[:param['TH_DEMAND'].size]
        param['EL_DEMAND'].value = self.safe_predict(self.electricity_demand_predictor, state)[:param['TH_DEMAND'].size]
        param['TEMPERATURE_STORAGE_0'].value = self.obs['temperature_storage'] if 'temperature_storage' in self.obs.keys() else 273.15+50
        param['SOC_0'].value = self.obs['soc_battery'] if 'soc_battery' in self.obs.keys() else 0.5
        param['B_TES'].value = param['TH_DEMAND'].value[:-1] * self.problem.constant_parameters['B1_TES']
    
    def safe_predict(self, predictor: Predictor | None, state: SimulationState):
        if predictor:  # If no predictor is loaded, it takes "None" value
            return predictor.predict(self.horizon, state)
        else:  # If there is no predictor, we interpret it as that there is no demand
            return np.zeros(int(self.horizon // (state.time_step/3600)))

@dataclass
class MPCProblem:
    constraints: List | None = None
    variables: Dict[str, cp.Variable] | None = None
    constant_parameters: Dict[str, float] | None = None
    variable_parameters: Dict[str, cp.Parameter] | None = None
    problem: cp.Problem | None = None