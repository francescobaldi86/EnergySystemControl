from EnergySystemControl.environments.base_environment import Environment, ElectricalNode
from EnergySystemControl.environments.utilities import HeatPumpConstantEfficiency, BalancingUtility, ColdWaterGrid
from EnergySystemControl.environments.storage_units import  HotWaterStorage
from EnergySystemControl.environments.demands import IEAHotWaterDemand
from EnergySystemControl.controllers.base_controller import HeaterControllerWithBandwidth


def test():
    # One node: house thermal mass
    nodes = {"contatore": ElectricalNode("contatore")}

    components = {
        "demand_DHW": IEAHotWaterDemand(name= "demand_DHW", thermal_node = "hot_water_storage_thermal_node", mass_node = "hot_water_storage_mass_node", reference_temperature = 40, profile_name='M'),
        "heat_pump": HeatPumpConstantEfficiency(name = 'heat_pump', thermal_node = "hot_water_storage_thermal_node", electrical_node = 'contatore', Qdot_max = 1.5, COP = 3.2),
        "hot_water_storage": HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45),
        "electric_grid": BalancingUtility(name = 'electric_grid', nodes = ['contatore']),
        "water_grid": ColdWaterGrid(name = 'water_grid', nodes = ['hot_water_storage_mass_node', 'hot_water_storage_thermal_node'])
    }
    controllers = {
        'heat_pump_controller': HeaterControllerWithBandwidth('heat_pump_controller', 'hot_water_storage_thermal_node', 'heat_pump', 40, 10)
    }

    env = Environment(nodes=nodes, components=components, controllers = controllers)  # dt = 60 s
    env.run(time_step = 0.5, time_end = 24.0*7)  # simulate 6 hours
    df_nodes, df_comps = env.to_dataframe()
    print(df_nodes.head())
    print(df_comps.head())

test()