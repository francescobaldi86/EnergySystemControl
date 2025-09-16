from EnergySystemControl.environments.base_classes import Component
from EnergySystemControl.helpers import *
import os, yaml
import numpy as np


class Producer(Component):
    def __init__(self, name: str, node: str):
        super().__init__(self, name)