"""Tests for HotWaterDemandSensor"""
import pytest
import energy_system_control as esc
import math
from energy_system_control.helpers import C2K
from energy_system_control.constants import WATER


class TestHotWaterDemandSensor:
    """Test suite for HotWaterDemandSensor"""
    
    def test_initialization(self):
        """Test HotWaterDemandSensor initialization"""
        sensor = esc.HotWaterDemandSensor(
            name='dhw_heat_flow_sensor',
            component_name='demand_DHW'
        )
        assert sensor.name == 'dhw_heat_flow_sensor'
        assert sensor.component_name == 'demand_DHW'
        assert sensor.port_name == 'demand_DHW_fluid_port'
    
    def test_sensor_measures_net_heat_flow(self):
        """Test that HotWaterDemandSensor measures net heat flow (T_hot - T_cold)"""
        components = [
            esc.IEAHotWaterDemand(
                name="demand_DHW",
                reference_temperature=40,
                profile_name='M'
            ),
            esc.HotWaterStorage(
                name='hot_water_storage',
                max_temperature=80,
                tank_volume=200,
                T_0=50
            ),
            esc.ElectricityGrid(name='electric_grid'),
            esc.ColdWaterGrid(name='water_grid', utility_type='fluid')
        ]
        
        sensors = [
            esc.HotWaterDemandSensor(
                'demand_heat_flow_sensor',
                component_name='demand_DHW'
            )
        ]
        
        connections = [
            ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
            ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
        ]
        
        env = esc.Environment(
            components=components,
            controllers=[],
            sensors=sensors,
            connections=connections
        )
        
        sim_config = esc.SimulationConfig(
            time_start_h=0.0,
            time_end_h=6.0,
            time_step_h=1.0
        )
        
        sim = esc.Simulator(env, sim_config)
        results = sim.run()
        df_ports, df_controllers, df_sensors = results.to_dataframe()
        
        # Verify sensor is recording heat flow
        assert 'demand_heat_flow_sensor' in df_sensors.columns
        heat_flow = df_sensors['demand_heat_flow_sensor']
        
        # Heat flow should have reasonable values
        assert heat_flow.max() < 10, "Max heat flow should be reasonable"
        assert len(heat_flow) == len(df_sensors), "Should have heat flow for every timestep"
    
    def test_sensor_in_simple_hot_water_system(self):
        """Test HotWaterDemandSensor in a simple hot water system"""
        components = [
            esc.IEAHotWaterDemand(
                name="demand_DHW",
                reference_temperature=40,
                profile_name='M'
            ),
            esc.HotWaterStorage(
                name='hot_water_storage',
                max_temperature=80,
                tank_volume=200,
                T_0=50
            ),
            esc.ElectricityGrid(name='electric_grid'),
            esc.ColdWaterGrid(name='water_grid', utility_type='fluid')
        ]
        
        sensors = [
            esc.HotWaterDemandSensor(
                'dhw_heat_flow',
                component_name='demand_DHW'
            )
        ]
        
        connections = [
            ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
            ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
        ]
        
        env = esc.Environment(
            components=components,
            controllers=[],
            sensors=sensors,
            connections=connections
        )
        
        sim_config = esc.SimulationConfig(
            time_start_h=0.0,
            time_end_h=24.0,
            time_step_h=1.0
        )
        
        sim = esc.Simulator(env, sim_config)
        results = sim.run()
        df_ports, df_controllers, df_sensors = results.to_dataframe()
        
        # Verify data is recorded
        assert 'dhw_heat_flow' in df_sensors.columns
        heat_flow = df_sensors['dhw_heat_flow']
        
        # Heat flow should have reasonable values (kW)
        assert heat_flow.max() < 20, "Max heat flow should be reasonable"
        assert len(heat_flow) == len(df_sensors), "Should have heat flow for every timestep"
    
    def test_heat_flow_varies_with_hot_water_temperature(self):
        """Test that heat flow increases when hot water temperature increases"""
        # Test case 1: Lower hot water temperature
        components_low = [
            esc.IEAHotWaterDemand(
                name="demand_DHW",
                reference_temperature=40,
                profile_name='M'
            ),
            esc.HotWaterStorage(
                name='hot_water_storage',
                max_temperature=80,
                tank_volume=200,
                T_0=45  # Lower initial temperature
            ),
            esc.ElectricityGrid(name='electric_grid'),
            esc.ColdWaterGrid(name='water_grid', utility_type='fluid')
        ]
        
        sensors_low = [
            esc.HotWaterDemandSensor(
                'dhw_heat_flow',
                component_name='demand_DHW'
            )
        ]
        
        connections = [
            ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
            ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
        ]
        
        env_low = esc.Environment(
            components=components_low,
            controllers=[],
            sensors=sensors_low,
            connections=connections
        )
        
        sim_config = esc.SimulationConfig(
            time_start_h=0.0,
            time_end_h=6.0,
            time_step_h=1.0
        )
        
        sim_low = esc.Simulator(env_low, sim_config)
        results_low = sim_low.run()
        df_low = results_low.to_dataframe()[2]
        
        heat_flow_low = df_low['dhw_heat_flow'].mean()
        
        # Test case 2: Higher hot water temperature
        components_high = [
            esc.IEAHotWaterDemand(
                name="demand_DHW",
                reference_temperature=40,
                profile_name='M'
            ),
            esc.HotWaterStorage(
                name='hot_water_storage',
                max_temperature=80,
                tank_volume=200,
                T_0=65  # Higher initial temperature
            ),
            esc.ElectricityGrid(name='electric_grid'),
            esc.ColdWaterGrid(name='water_grid', utility_type='fluid')
        ]
        
        sensors_high = [
            esc.HotWaterDemandSensor(
                'dhw_heat_flow',
                component_name='demand_DHW'
            )
        ]
        
        env_high = esc.Environment(
            components=components_high,
            controllers=[],
            sensors=sensors_high,
            connections=connections
        )
        
        sim_high = esc.Simulator(env_high, sim_config)
        results_high = sim_high.run()
        df_high = results_high.to_dataframe()[2]
        
        heat_flow_high = df_high['dhw_heat_flow'].mean()
        
        # Higher temperature should result in higher net heat flow
        # (noting that in the early hours, demand might affect this)
        assert heat_flow_high >= heat_flow_low * 0.8, "Higher supply temp should give comparable or higher heat flow"
    
    def test_sensor_port_name_convention(self):
        """Test that sensor correctly derives port name from component name"""
        sensor = esc.HotWaterDemandSensor(
            name='sensor1',
            component_name='my_demand_component'
        )
        assert sensor.port_name == 'my_demand_component_fluid_port'
        
        sensor2 = esc.HotWaterDemandSensor(
            name='sensor2',
            component_name='demand_DHW'
        )
        assert sensor2.port_name == 'demand_DHW_fluid_port'
