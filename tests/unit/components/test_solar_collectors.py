from datetime import datetime
from math import isclose

import numpy as np
import pytest

from energy_system_control.components.implicit_components.solar_collectors import (
    EvacuatedTubeCollector,
    FlatPlateCollector,
    SolarCollector,
    UnglazedCollector,
)
from energy_system_control.core.base_classes import EnvironmentalData
from energy_system_control.sim.state import SimulationState


COLLECTOR_PARAMETERS = [
    (UnglazedCollector, 0.85, 20.0),
    (FlatPlateCollector, 0.75, 4.75),
    (EvacuatedTubeCollector, 0.675, 1.5),
]


def make_collector(collector_type):
    return collector_type(
        name="collector",
        tilt=30.0,
        azimuth=0.0,
        surface_area=4.0,
        latitude=45.0,
        longitude=8.0,
    )


@pytest.mark.parametrize("collector_type, efficiency_0, efficiency_1", COLLECTOR_PARAMETERS)
def test_collector_subclasses_use_solar_collector_model(
    collector_type, efficiency_0, efficiency_1
):
    collector = make_collector(collector_type)

    assert isinstance(collector, SolarCollector)
    assert collector.efficiency_0 == pytest.approx(efficiency_0)
    assert collector.efficiency_1 == pytest.approx(efficiency_1)
    assert collector.tilt == 30.0
    assert isclose(collector.tilt_rad, np.pi / 6)


def test_solar_collector_ports_and_tilt_validation():
    collector = SolarCollector(
        name="collector",
        tilt=30.0,
        azimuth=0.0,
        surface_area=4.0,
        latitude=45.0,
        longitude=8.0,
        efficiency_0=0.75,
        efficiency_1=4.75,
    )

    ports = collector.create_ports()

    assert set(ports) == {
        "collector_water_input_port",
        "collector_water_outlet_port",
    }
    assert collector.ports == ports


def test_solar_collector_efficiency_changes_with_temperature_difference():
    collector = SolarCollector(
        name="collector",
        tilt=30.0,
        azimuth=0.0,
        surface_area=4.0,
        latitude=45.0,
        longitude=8.0,
        efficiency_0=0.8,
        efficiency_1=4.0,
    )

    assert collector.get_efficiency(800.0, 293.15, 293.15) == 0.8
    assert collector.get_efficiency(800.0, 313.15, 293.15) == 0.9

def test_solar_collector_step_adds_heat_and_preserves_mass(monkeypatch):
    collector = SolarCollector(
        name="collector",
        tilt=30.0,
        azimuth=0.0,
        surface_area=4.0,
        latitude=45.0,
        longitude=8.0,
        efficiency_0=0.8,
        efficiency_1=0.0,
    )
    collector.create_ports()
    input_port = collector.ports[collector.input_port_name]
    output_port = collector.ports[collector.output_port_name]
    input_port.T = 293.15
    input_port.flows["mass"] = 0.01
    input_port.flows["heat"] = 2.0

    monkeypatch.setattr(
        "energy_system_control.components.implicit_components.thermal_solar_panels.calculate_solar_angles",
        lambda *args: (45.0, 0.0),
    )
    monkeypatch.setattr(
        "energy_system_control.components.implicit_components.thermal_solar_panels.calculate_effective_irradiance",
        lambda *args: 500.0,
    )
    state = SimulationState(
        simulation_start_datetime=datetime(2025, 1, 1),
        environmental_data=EnvironmentalData(
            temperature_ambient=293.15,
            direct_irradiation=400.0,
            diffuse_irradiation=100.0,
        ),
    )

    collector.step(state)

    expected_heat_gain = 500.0 * 4.0 * 0.8
    assert output_port.flows["mass"] == input_port.flows["mass"]
    assert output_port.flows["heat"] == input_port.flows["heat"] + expected_heat_gain
    assert output_port.T > input_port.T


def test_solar_collector_step_does_not_add_heat_without_flow(monkeypatch):
    collector = SolarCollector(
        name="collector",
        tilt=30.0,
        azimuth=0.0,
        surface_area=4.0,
        latitude=45.0,
        longitude=8.0,
        efficiency_0=0.8,
        efficiency_1=0.0,
    )
    collector.create_ports()
    input_port = collector.ports[collector.input_port_name]
    output_port = collector.ports[collector.output_port_name]
    input_port.T = 293.15
    input_port.flows["mass"] = 0.0
    input_port.flows["heat"] = 2.0
    monkeypatch.setattr(
        "energy_system_control.components.implicit_components.thermal_solar_panels.calculate_solar_angles",
        lambda *args: (45.0, 0.0),
    )
    monkeypatch.setattr(
        "energy_system_control.components.implicit_components.thermal_solar_panels.calculate_effective_irradiance",
        lambda *args: 500.0,
    )
    state = SimulationState(
        simulation_start_datetime=datetime(2025, 1, 1),
        environmental_data=EnvironmentalData(direct_irradiation=500.0),
    )

    collector.step(state)

    assert output_port.flows["mass"] == 0.0
    assert output_port.flows["heat"] == input_port.flows["heat"]
    assert output_port.T == input_port.T