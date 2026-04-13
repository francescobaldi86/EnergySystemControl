from energy_system_control.components.base import Component, TimeSeriesData
from energy_system_control.helpers import *
from energy_system_control.sim.state import SimulationState
from typing import List, Dict
import os, yaml, csv, json, requests
import numpy as np
import pandas as pd
from typing import Literal


class Producer(Component):
    port_name: str
    production_type: str
    def __init__(self, name: str, production_type: str):
        self.production_type = production_type
        self.port_name = f'{name}_{self.production_type}_port'
        super().__init__(name, {self.port_name: self.production_type})


class ConstantPowerProducer(Producer):
    def __init__(self, name: str, production_type: str, power: float):
        super().__init__(name, production_type)
        self.power = power  # 
    
    def step(self, state: SimulationState, action = None): 
        self.ports[self.port_name].flow[self.production_type] = -self.power * state.time_step  # Since it is a producer, the net energy flow is always negative


