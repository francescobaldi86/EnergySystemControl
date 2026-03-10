"""Tests for TankTemperatureSensor"""
import pytest
import energy_system_control as esc
import math
from energy_system_control.helpers import C2K


class TestTankTemperatureSensor:
    """Test suite for TankTemperatureSensor"""
    
    def test_initialization(self):
        """Test TankTemperatureSensor initialization"""
        sensor = esc.TankTemperatureSensor(
            name='tank_temp_sensor',
            component_name='hot_water_storage'
        )
        assert sensor.name == 'tank_temp_sensor'
        assert sensor.component_name == 'hot_water_storage'
    
    def test_temperature_sensor_in_hot_water_system(self):
        """Test TankTemperatureSensor in hot water system"""
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
            esc.TankTemperatureSensor(
                'storage_tank_temperature_sensor',
                component_name='hot_water_storage'
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
        
        # Verify temperature is recorded and in reasonable range
        assert 'storage_tank_temperature_sensor' in df_sensors.columns
        temp_k = df_sensors['storage_tank_temperature_sensor']
        
        # Temperature should be in Kelvin between ~30°C and ~85°C
        assert temp_k.min() > 300, "Tank temperature should be above 300K (~27°C)"
        assert temp_k.max() < 360, "Tank temperature should be below 360K (~87°C)"
    
    def test_temperature_sensor_reads_initial_temperature(self):
        """Test that temperature sensor reads initial tank temperature"""
        initial_temp_c = 55.0
        
        components = [
            esc.HotWaterStorage(
                name='hot_water_storage',
                max_temperature=80,
                tank_volume=200,
                T_0=initial_temp_c
            ),
            esc.ColdWaterGrid(name='water_grid', utility_type='fluid')
        ]
        
        sensors = [
            esc.TankTemperatureSensor(
                'tank_temp',
                component_name='hot_water_storage'
            )
        ]
        
        connections = [
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
            time_end_h=0.5,
            time_step_h=0.5
        )
        
        sim = esc.Simulator(env, sim_config)
        results = sim.run()
        df_ports, df_controllers, df_sensors = results.to_dataframe()
        
        # Initial temperature should be close to 55°C = 328.15 K
        initial_temp_k = df_sensors['tank_temp'].iloc[0]
        expected_temp_k = C2K(initial_temp_c)
        assert math.isclose(initial_temp_k, expected_temp_k, abs_tol=0.5)
    
    def test_temperature_sensor_with_heat_input(self):
        """Test temperature sensor reading with heat input from hot water demand"""
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
                T_0=40
            ),
            esc.BalancingUtility(name='electric_grid', utility_type='electricity'),
            esc.ColdWaterGrid(name='water_grid', utility_type='fluid')
        ]
        
        sensors = [
            esc.TankTemperatureSensor(
                'storage_tank_temperature_sensor',
                component_name='hot_water_storage'
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
        
        temp_k = df_sensors['storage_tank_temperature_sensor']
        
        # Temperature should be recorded for all timesteps
        assert len(temp_k) > 0, "Should have temperature readings"
        assert temp_k.min() > 300, "Temperature should be in reasonable range"
    
    def test_temperature_sensor_with_height_specification(self):
        """Test TankTemperatureSensor initialization with sensor height"""
        sensor = esc.TankTemperatureSensor(
            name='tank_temp_middle',
            component_name='hot_water_storage',
            sensor_height=1.0  # 1 meter height
        )
        assert sensor.sensor_height == 1.0
        assert sensor.component_name == 'hot_water_storage'
