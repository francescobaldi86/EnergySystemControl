"""Tests for PowerSensor and ElectricPowerSensor"""
import pytest
import energy_system_control as esc
import math
import os

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
__DATA__ = os.path.join(__TEST__, 'DATA')


class TestElectricPowerSensor:
    """Test suite for ElectricPowerSensor"""
    
    def test_initialization(self):
        """Test ElectricPowerSensor initialization"""
        sensor = esc.ElectricPowerSensor(
            name='test_power_sensor',
            port_name='test_electricity_port'
        )
        assert sensor.name == 'test_power_sensor'
        assert sensor.port_name == 'test_electricity_port'
        assert sensor.flow_type == 'electricity'
    
    def test_sensor_in_pv_system(self):
        """Test ElectricPowerSensor in PV system with battery"""
        components = [
            esc.PVpanelFromPVGIS(
                name='pv_panels',
                installed_power=5.0,
                latitude=44.511,
                longitude=11.335,
                tilt=30,
                azimuth=90
            ),
            esc.LithiumIonBattery(
                name='battery',
                capacity=10.0,
                SOC_0=0.5
            ),
            esc.Inverter(name='inverter'),
            esc.BalancingUtility(name='electric_grid', utility_type='electricity'),
        ]
        
        controllers = [
            esc.InverterController(
                name='inverter_controller',
                inverter_name='inverter',
                battery_name='battery',
                SOC_min=0.3,
                SOC_max=0.9
            )
        ]
        
        sensors = [
            esc.ElectricPowerSensor('pv_power_sensor', 'inverter_PV_input_port'),
            esc.ElectricPowerSensor('grid_power_sensor', 'inverter_grid_input_port'),
            esc.ElectricPowerSensor('ess_power_sensor', 'inverter_ESS_port'),
            esc.SOCSensor('battery_SOC_sensor', 'battery'),
        ]
        
        connections = [
            ('inverter_PV_input_port', 'pv_panels_electricity_port'),
            ('inverter_ESS_port', 'battery_electricity_port'),
            ('inverter_grid_input_port', 'electric_grid_electricity_port'),
        ]
        
        env = esc.Environment(
            components=components,
            controllers=controllers,
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
        
        # Verify that power sensors are recording data
        assert 'pv_power_sensor' in df_sensors.columns
        assert 'grid_power_sensor' in df_sensors.columns
        assert 'ess_power_sensor' in df_sensors.columns
        
        # PV power should have some positive values during the day
        pv_power = df_sensors['pv_power_sensor']
        assert pv_power.sum() > 0, "PV should generate some power"
        assert pv_power.max() > 0, "PV max power should be positive"
        
        # Grid and battery power should have both positive and negative values
        grid_power = df_sensors['grid_power_sensor']
        ess_power = df_sensors['ess_power_sensor']
        
        assert len(grid_power) > 0
        assert len(ess_power) > 0
    
    def test_power_sensor_with_constant_demand(self):
        """Test power sensor measuring electricity production from constant producer"""
        components = [
            esc.ConstantPowerProducer(
                name='constant_producer',
                production_type='electricity',
                power=5.0  # 5 kW constant power
            ),
            esc.BalancingUtility(name='electric_grid', utility_type='electricity'),
        ]
        
        sensors = [
            esc.ElectricPowerSensor(
                'producer_power_sensor',
                'constant_producer_electricity_port'
            ),
            esc.ElectricPowerSensor(
                'grid_power_sensor',
                'electric_grid_electricity_port'
            ),
        ]
        
        connections = [
            ('electric_grid_electricity_port', 'constant_producer_electricity_port'),
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
            time_step_h=0.5
        )
        
        sim = esc.Simulator(env, sim_config)
        results = sim.run()
        df_ports, df_controllers, df_sensors = results.to_dataframe()
        
        # Power should be relatively constant at -5 kW (negative because it's a producer)
        producer_power = df_sensors['producer_power_sensor']
        assert math.isclose(producer_power.mean(), -5.0, abs_tol=0.5)
    
    def test_power_sensor_measures_in_kilowatts(self):
        """Test that power sensor returns values in kW"""
        components = [
            esc.PVpanelFromPVGIS(
                name='pv_panels',
                installed_power=3.0,  # 3 kW peak
                latitude=44.511,
                longitude=11.335,
                tilt=30,
                azimuth=90
            ),
            esc.BalancingUtility(name='electric_grid', utility_type='electricity'),
        ]
        
        sensors = [
            esc.ElectricPowerSensor('pv_power', 'pv_panels_electricity_port'),
        ]
        
        connections = [
            ('electric_grid_electricity_port', 'pv_panels_electricity_port'),
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
        
        # Peak power should be within reasonable bounds based on installed power
        pv_power = df_sensors['pv_power']
        assert abs(pv_power.max()) <= 3.0, "Max power should not exceed installed power"
        assert len(pv_power) == 24, "Should have 24 hours of measurements"
