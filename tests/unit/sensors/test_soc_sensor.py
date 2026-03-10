"""Tests for SOCSensor (State of Charge Sensor)"""
import pytest
import energy_system_control as esc
import math


class TestSOCSensor:
    """Test suite for SOCSensor"""
    
    def test_initialization(self):
        """Test SOCSensor initialization"""
        sensor = esc.SOCSensor(
            name='battery_soc_sensor',
            component_name='battery'
        )
        assert sensor.name == 'battery_soc_sensor'
        assert sensor.component_name == 'battery'
    
    def test_soc_sensor_with_battery(self):
        """Test SOCSensor measuring battery state of charge"""
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
                SOC_0=0.5  # Start at 50% SOC
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
            esc.SOCSensor('battery_soc', 'battery'),
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
        
        # Verify SOC is recorded
        assert 'battery_soc' in df_sensors.columns
        soc = df_sensors['battery_soc']
        
        # SOC should always be between 0 and 1
        assert soc.min() >= 0.0, "SOC should not go below 0"
        assert soc.max() <= 1.0, "SOC should not go above 1"
        
        # SOC should vary during the simulation
        assert soc.std() > 0.0, "SOC should change during simulation"
    
    def test_soc_sensor_initial_value(self):
        """Test that SOC sensor reads the initial SOC correctly"""
        components = [
            esc.LithiumIonBattery(
                name='battery',
                capacity=5.0,
                SOC_0=0.75  # Start at 75% SOC
            ),
            esc.BalancingUtility(name='electric_grid', utility_type='electricity'),
        ]
        
        sensors = [
            esc.SOCSensor('battery_soc', 'battery'),
        ]
        
        connections = []
        
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
        
        # Initial SOC should be close to 0.75
        initial_soc = df_sensors['battery_soc'].iloc[0]
        assert math.isclose(initial_soc, 0.75, abs_tol=0.05)
    
    def test_soc_sensor_respects_bounds(self):
        """Test that battery SOC stays within min/max bounds"""
        components = [
            esc.PVpanelFromPVGIS(
                name='pv_panels',
                installed_power=10.0,
                latitude=44.511,
                longitude=11.335,
                tilt=30,
                azimuth=90
            ),
            esc.LithiumIonBattery(
                name='battery',
                capacity=5.0,
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
                SOC_min=0.2,  # Minimum 20%
                SOC_max=0.8   # Maximum 80%
            )
        ]
        
        sensors = [
            esc.SOCSensor('battery_soc', 'battery'),
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
        
        soc = df_sensors['battery_soc']
        
        # SOC should respect the controller bounds (with some tolerance for numerical precision)
        assert soc.min() >= 0.15, "SOC should not go significantly below min bound"
        assert soc.max() <= 0.85, "SOC should not go significantly above max bound"
