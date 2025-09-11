from EnergySystemControl.environments.base_environment import Component
from EnergySystemControl.helpers import read_timeseries_data_to_numpy, C2K, K2C
import os, yaml
import numpy as np


class Producer(Component):
    def __init__(self, name: str, node: str, project_path: str):
        super().__init__(self, name, project_path)