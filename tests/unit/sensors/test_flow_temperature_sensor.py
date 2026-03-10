"""Tests for FlowTemperatureSensor"""
import pytest
import energy_system_control as esc
import math


class TestFlowTemperatureSensor:
    """Test suite for FlowTemperatureSensor"""
    
    def test_initialization(self):
        """Test FlowTemperatureSensor initialization"""
        sensor = esc.FlowTemperatureSensor(
            name='test_temp_sensor',
            port_name='test_port'
        )
        assert sensor.name == 'test_temp_sensor'
        assert sensor.port_name == 'test_port'
    
    def test_sensor_in_system_with_hot_water_storage(self):
        """Test FlowTemperatureSensor measuring temperature at hot water storage output"""
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
                T_0=45
            ),
            esc.BalancingUtility(name='electric_grid', utility_type='electricity'),
            esc.ColdWaterGrid(name='water_grid', utility_type='fluid')
        ]
        
        sensors = [
            esc.TankTemperatureSensor(
                'storage_tank_temperature_sensor',
                component_name='hot_water_storage'
            ),
            esc.FlowTemperatureSensor(
                'hot_water_output_temp_sensor',
                'hot_water_storage_hot_water_output_port'
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
        
        # Verify that the temperature sensor readings are recorded
        assert 'hot_water_output_temp_sensor' in df_sensors.columns
        # Temperature should be in a reasonable range (should be around 40-80 K + 273)
        temp_k = df_sensors['hot_water_output_temp_sensor'].mean()
        assert 300 < temp_k < 360  # Between 27°C and 87°C
    
    def test_sensor_measures_port_temperature(self):
        """Test that FlowTemperatureSensor correctly measures port temperature"""
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
            esc.BalancingUtility(name='electric_grid', utility_type='electricity'),
            esc.ColdWaterGrid(name='water_grid', utility_type='fluid')
        ]
        
        sensors = [
            esc.FlowTemperatureSensor(
                'output_flow_temp',
                'hot_water_storage_hot_water_output_port'
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
            time_end_h=1.0,
            time_step_h=0.1
        )
        
        sim = esc.Simulator(env, sim_config)
        results = sim.run()
        df_ports, df_controllers, df_sensors = results.to_dataframe()
        
        # Verify sensor data exists
        assert len(df_sensors) > 0
        assert 'output_flow_temp' in df_sensors.columns
