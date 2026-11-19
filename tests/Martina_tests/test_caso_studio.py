import energy_system_control as esc
import pytest
from energy_system_control.components.explicit_components.demands import HotWaterDemand, ElectricityDemand, CookingDemand
from energy_system_control.components.explicit_components.pv_panels import PVpanelFromPVGIS
from energy_system_control.components.controlled_components.heat_pumps import HeatPump
from energy_system_control.components.controlled_components.other_heat_sources import InductionStove
from energy_system_control.components.composite_components.inverters import Inverter
from energy_system_control.components.base import ImplicitComponent
from energy_system_control.components.grids.grids import ElectricityGrid

class HeatExchanger(ImplicitComponent):
    """
    Scambiatore di calore semplificato:
    riceve calore da una HeatPort e lo trasferisce
    a una FluidPort, assegnando una temperatura.
    """
    def __init__(self, name: str, outlet_temperature: float = 45):

        self.heat_input_port_name = f"{name}_heat_input_port"
        self.fluid_output_port_name = f"{name}_fluid_output_port"

        self.outlet_temperature = outlet_temperature + 273.15

        super().__init__(name,{self.heat_input_port_name: "heat",self.fluid_output_port_name: "fluid"})

    def step(self, state, action=None):

        heat = self.ports[self.heat_input_port_name].flows["heat"]

        if heat is not None:
            fluid_port = self.ports[self.fluid_output_port_name]

            fluid_port.flows["heat"] = -heat
            fluid_port.T = self.outlet_temperature

    def set_inherited_port_values(self, state):
        fluid_port = self.ports[self.fluid_output_port_name]
        return {fluid_port.connected_port.name: {'T': self.outlet_temperature}}


def test_caso_tutto_elettrico():

    components = [PVpanelFromPVGIS(name="pv_panel", installed_power=5.0, latitude=35.50, longitude=12.60, tilt=30, azimuth=0, loss=14, years=[2023], time_alignment="yearly"),
        esc.HeatPumpConstantEfficiency(name="heat_pump", Qdot_design=8, COP_design=4.0),
        esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45, convection_coefficient_losses = 0.0),
        InductionStove(name="induction_stove",efficiency=0.9,max_power=2.0,source_type="electricity"),
        # HeatExchanger(name="heat_exchanger",outlet_temperature=45),
        HotWaterDemand.from_iea(name="hot_water_demand",reference_temperature=40, profile_name = 'M'),
        CookingDemand.from_value(name="cooking_demand",value=500/8760, var_type='power', var_unit = 'kW'),
        ElectricityDemand.from_value(name="electricity_demand",value=2000/8760, var_type='power', var_unit = 'kW'),
        Inverter(name = 'inverter', number_of_ac_output_ports = 3),
        esc.ColdWaterGrid(name = 'water_grid', utility_type = 'fluid'),
        ElectricityGrid(name="electricity_grid",cost_of_electricity_purchased=0.25,value_of_electricity_sold=0.10)]

    controllers = [esc.HeaterControllerWithBandwidth('heat_pump_controller', 'heat_pump', 'hot_water_storage_sensor', 40.0, 5.0)]
    sensors = [esc.TankTemperatureSensor('hot_water_storage_sensor', 'hot_water_storage'),
               esc.PowerSensor('PV_power_sensor', 'inverter_PV_input_port_0', 'electricity'),
               esc.PowerSensor('grid_power_sensor', 'inverter_grid_input_port', 'electricity')]

    connections = [("pv_panel_electricity_port","inverter_PV_input_port_0"),
        ("inverter_AC_output_port_0","heat_pump_electricity_input_port"),
        ("inverter_AC_output_port_1","induction_stove_electricity_input_port"),
        ("inverter_AC_output_port_2","electricity_demand_electricity_port"),
        ("heat_pump_heat_output_port","hot_water_storage_main_heat_input_port"),
        ("hot_water_storage_hot_water_output_port","hot_water_demand_fluid_port"),
        ("induction_stove_heat_output_port","cooking_demand_heat_port"),
        ("hot_water_storage_cold_water_input_port","water_grid_fluid_port"),
        ("inverter_grid_input_port","electricity_grid_electricity_port")]

    env = esc.Environment(components=components,controllers=controllers,sensors=sensors,connections=connections)

    sim_config = esc.SimulationConfig(simulation_end_h=8760,time_step_h=1/60)

    sim = esc.Simulator(env, sim_config)

    results = sim.run()

    df_ports, df_controllers, df_sensors = results.to_dataframe()

    print('Simulazione completata, yey!')
  
    assert df_ports is not None

if __name__ == "__main__":
    test_caso_tutto_elettrico()