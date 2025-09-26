from EnergySystemControl.environments.base_environment import Environment, ElectricalNode
from EnergySystemControl.environments.utilities import HeatPumpConstantEfficiency, BalancingUtility, ColdWaterGrid, GenericUtility
from EnergySystemControl.environments.storage_units import  HotWaterStorage, Battery
from EnergySystemControl.environments.demands import IEAHotWaterDemand
from EnergySystemControl.controllers.base_controller import HeaterControllerWithBandwidth, Inverter
from EnergySystemControl.environments.producers import PVpanelFromPVGIS
import pytest, math, os

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def test_1():
    # One node: house thermal mass
    nodes = {"contatore": ElectricalNode("contatore")}
    components = {
        "demand_DHW": IEAHotWaterDemand(name= "demand_DHW", thermal_node = "hot_water_storage_thermal_node", mass_node = "hot_water_storage_mass_node", reference_temperature = 40, profile_name='M'),
        "heat_pump": HeatPumpConstantEfficiency(name = 'heat_pump', thermal_node = "hot_water_storage_thermal_node", electrical_node = 'contatore', Qdot_max = 1.5, COP = 3.2),
        "hot_water_storage": HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45),
        "electric_grid": BalancingUtility(name = 'electric_grid', nodes = ['contatore']),
        "water_grid": ColdWaterGrid(name = 'water_grid', nodes = ['hot_water_storage_mass_node', 'hot_water_storage_thermal_node'])
    }
    controllers = [
        HeaterControllerWithBandwidth('heat_pump_controller', 'hot_water_storage_thermal_node', 'heat_pump', 40, 10)
    ]

    env = Environment(nodes=nodes, components=components, controllers = controllers)  # dt = 60 s
    time_step = 0.5
    env.run(time_step = time_step, time_end = 24.0*7)  # simulate 6 hours
    df_nodes, df_comps = env.to_dataframe()
    assert math.isclose(df_nodes.loc[10.0, 'hot_water_storage_thermal_node'], 325, abs_tol = 1)

def test_2():
    # Testing problem 1 with different time steps. In particular, it verifies that when changing the time step the consumption of the heat pump remains approximately constant
    nodes = {"contatore": ElectricalNode("contatore")}
    components = {
        "demand_DHW": IEAHotWaterDemand(name= "demand_DHW", thermal_node = "hot_water_storage_thermal_node", mass_node = "hot_water_storage_mass_node", reference_temperature = 40, profile_name='M'),
        "heat_pump": HeatPumpConstantEfficiency(name = 'heat_pump', thermal_node = "hot_water_storage_thermal_node", electrical_node = 'contatore', Qdot_max = 1.5, COP = 3.2),
        "hot_water_storage": HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45),
        "electric_grid": BalancingUtility(name = 'electric_grid', nodes = ['contatore']),
        "water_grid": ColdWaterGrid(name = 'water_grid', nodes = ['hot_water_storage_mass_node', 'hot_water_storage_thermal_node'])
    }
    controllers = [
        HeaterControllerWithBandwidth('heat_pump_controller', 'hot_water_storage_thermal_node', 'heat_pump', 40, 10)
    ]
    env = Environment(nodes=nodes, components=components, controllers = controllers)  # dt = 60 s
    for time_step in [1, 0.5, 0.25, 1/6, 5/60, 1/60]:
    # Test that results remain similar when changing the time step
        env.run(time_step = time_step, time_end = 24.0*7)  # simulate 6 hours
        df_nodes, df_comps = env.to_dataframe()
        heat_pump_energy_demand = -df_comps[('heat_pump', 'contatore')].sum() * time_step
        assert math.isclose(heat_pump_energy_demand, 13, abs_tol = 2)

def test_3():
    # Trying a more complex system, with PV panels 
    nodes = {"contatore": ElectricalNode("contatore")}
    components = {
        "demand_DHW": IEAHotWaterDemand(name= "demand_DHW", thermal_node = "hot_water_storage_thermal_node", mass_node = "hot_water_storage_mass_node", reference_temperature = 40, profile_name='M'),
        "heat_pump": HeatPumpConstantEfficiency(name = 'heat_pump', thermal_node = "hot_water_storage_thermal_node", electrical_node = 'contatore', Qdot_max = 1.5, COP = 3.2),
        "hot_water_storage": HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45),
        "pv_panels": PVpanelFromPVGIS(name = 'pv_panels', electrical_node='contatore', installed_power=3.0, latitude=44.511, longitude=11.335, tilt=30, azimuth=90),
        "electric_grid": BalancingUtility(name = 'electric_grid', nodes = ['contatore']),
        "water_grid": ColdWaterGrid(name = 'water_grid', nodes = ['hot_water_storage_mass_node', 'hot_water_storage_thermal_node'])
    }
    controllers = [
        HeaterControllerWithBandwidth('heat_pump_controller', 'hot_water_storage_thermal_node', 'heat_pump', 40, 10)
    ]

    env = Environment(nodes=nodes, components=components, controllers = controllers)  # dt = 60 s
    time_step = 0.5
    env.run(time_step = time_step, time_end = 24.0*7)  # simulate 6 hours
    df_nodes, df_comps = env.to_dataframe()
    heat_pump_energy_demand = -df_comps[('heat_pump', 'contatore')].sum() * time_step
    assert math.isclose(heat_pump_energy_demand, 13, abs_tol = 2)
    electricity_from_pv = df_comps[('pv_panels', 'contatore')].sum() * time_step
    assert math.isclose(electricity_from_pv, 27, abs_tol = 2)
    electricity_demand = df_comps[('electric_grid', 'contatore')].sum() * time_step
    electricity_from_grid = df_comps.loc[df_comps[('electric_grid', 'contatore')] > 0, ('electric_grid', 'contatore')].sum() * time_step
    electricity_to_grid = -df_comps.loc[df_comps[('electric_grid', 'contatore')] < 0, ('electric_grid', 'contatore')].sum() * time_step
    assert math.isclose(electricity_demand, -15, abs_tol = 2)
    assert math.isclose(electricity_from_grid, 8, abs_tol = 2)
    assert math.isclose(electricity_to_grid, 24, abs_tol = 2)
    assert math.isclose(df_nodes.loc[10.0, 'hot_water_storage_thermal_node'], 325, abs_tol = 1)

def test_4():
    # Adding more complexity: a battery
    # Trying a more complex system, with PV panels 
    components = {
        "demand_DHW": IEAHotWaterDemand(name= "demand_DHW", thermal_node = "hot_water_storage_thermal_node", mass_node = "hot_water_storage_mass_node", reference_temperature = 40, profile_name='M'),
        "heat_pump": HeatPumpConstantEfficiency(name = 'heat_pump', thermal_node = "hot_water_storage_thermal_node", electrical_node = 'battery_electrical_node', Qdot_max = 1.5, COP = 3.2),
        "hot_water_storage": HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45),
        "pv_panels": PVpanelFromPVGIS(name = 'pv_panels', electrical_node='battery_electrical_node', installed_power=3.0, latitude=44.511, longitude=11.335, tilt=30, azimuth=90),
        "battery": Battery(name = 'battery', capacity = 2.0, SOC_0 = 0.5),
        "electric_grid": GenericUtility(name = 'electric_grid', node= 'battery_electrical_node', max_power = 5, type = 'bidirectional'),
        "water_grid": ColdWaterGrid(name = 'water_grid', nodes = ['hot_water_storage_mass_node', 'hot_water_storage_thermal_node'])
    }
    controllers = [
        HeaterControllerWithBandwidth('heat_pump_controller', 'hot_water_storage_thermal_node', 'heat_pump', 40, 10),
        Inverter('inverter', 'battery_electrical_node', 'electric_grid')
    ]

    env = Environment(nodes={}, components=components, controllers = controllers)  # dt = 60 s
    time_step = 0.5
    env.run(time_step = time_step, time_end = 24.0*7)  # simulate 6 hours
    df_nodes, df_comps = env.to_dataframe()
    heat_pump_energy_demand = -df_comps[('heat_pump', 'battery_electrical_node')].sum() * time_step
    assert math.isclose(heat_pump_energy_demand, 13, abs_tol = 2)
    electricity_from_pv = df_comps[('pv_panels', 'battery_electrical_node')].sum() * time_step
    assert math.isclose(electricity_from_pv, 27, abs_tol = 2)
    electricity_demand = df_comps[('electric_grid', 'battery_electrical_node')].sum() * time_step
    electricity_from_grid = df_comps.loc[df_comps[('electric_grid', 'battery_electrical_node')] > 0, ('electric_grid', 'battery_electrical_node')].sum() * time_step
    electricity_to_grid = -df_comps.loc[df_comps[('electric_grid', 'battery_electrical_node')] < 0, ('electric_grid', 'battery_electrical_node')].sum() * time_step
    assert math.isclose(electricity_demand, -13, abs_tol = 2)
    assert math.isclose(electricity_from_grid, 0, abs_tol = 2)
    assert math.isclose(electricity_to_grid, 13, abs_tol = 2)
    assert math.isclose(df_nodes.loc[10.0, 'hot_water_storage_thermal_node'], 325, abs_tol = 1)