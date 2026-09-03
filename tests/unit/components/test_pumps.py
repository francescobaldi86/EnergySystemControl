"""
Comprehensive test suite for pump components (SimplePump and Pump).

Tests cover:
- SimplePump initialization and flow control
- Pump base class initialization with/without electricity input
- Flow rate modulation based on control actions
- Energy/heat flow conservation
- Port management and naming
- Integration with simulation environments
- Edge cases and boundary conditions
"""

import pytest
import numpy as np
import math
from energy_system_control.components.controlled_components.pumps import SimplePump, Pump
from energy_system_control.sim.state import SimulationState
from energy_system_control.constants import WATER
import energy_system_control as esc


class TestSimplePumpInitialization:
    """Test suite for SimplePump initialization."""
    
    def test_simple_pump_creation_basic(self):
        """Test basic creation of SimplePump."""
        pump = SimplePump(name='pump_1', mass_flow_rate=1.0)
        
        assert pump.name == 'pump_1'
        assert pump.mass_flow_rate == 1.0
        assert pump.input_water_port_name == 'pump_1_water_input_port'
        assert pump.output_water_port_name == 'pump_1_water_output_port'
    
    def test_simple_pump_port_names(self):
        """Test that port names are generated correctly."""
        pump = SimplePump(name='my_pump', mass_flow_rate=0.5)
        
        assert 'my_pump' in pump.input_water_port_name
        assert 'my_pump' in pump.output_water_port_name
        assert 'water_input_port' in pump.input_water_port_name
        assert 'water_output_port' in pump.output_water_port_name
    
    def test_simple_pump_different_flow_rates(self):
        """Test SimplePump creation with different mass flow rates."""
        flow_rates = [0.1, 0.5, 1.0, 5.0, 10.0]
        
        for flow_rate in flow_rates:
            pump = SimplePump(name=f'pump_{flow_rate}', mass_flow_rate=flow_rate)
            assert pump.mass_flow_rate == flow_rate
    
    def test_simple_pump_ports_creation(self):
        """Test that ports are created correctly."""
        pump = SimplePump(name='test_pump', mass_flow_rate=1.0)
        pump.create_ports()
        
        assert pump.input_water_port_name in pump.ports
        assert pump.output_water_port_name in pump.ports


class TestSimplePumpOperation:
    """Test suite for SimplePump operation."""
    
    def test_simple_pump_full_flow(self):
        """Test SimplePump at full flow (action=1.0)."""
        pump = SimplePump(name='pump_1', mass_flow_rate=1.0)
        pump.create_ports()
        
        # Set inlet conditions
        inlet_temp = 50.0 + 273.15
        pump.ports[pump.input_water_port_name].T = inlet_temp
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        # Run pump at full flow
        state = SimulationState(time=0.0, time_step=60.0)
        pump.step(state, action=1.0)
        
        # Check output flow
        assert pump.ports[pump.output_water_port_name].flows['mass'] == 1.0
    
    def test_simple_pump_half_flow(self):
        """Test SimplePump at half flow (action=0.5)."""
        pump = SimplePump(name='pump_1', mass_flow_rate=1.0)
        pump.create_ports()
        
        pump.ports[pump.input_water_port_name].T = 50.0 + 273.15
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        state = SimulationState(time=0.0, time_step=60.0)
        pump.step(state, action=0.5)
        
        assert pump.ports[pump.output_water_port_name].flows['mass'] == 0.5
    
    def test_simple_pump_zero_flow(self):
        """Test SimplePump at zero flow (action=0.0)."""
        pump = SimplePump(name='pump_1', mass_flow_rate=1.0)
        pump.create_ports()
        
        pump.ports[pump.input_water_port_name].T = 50.0 + 273.15
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        state = SimulationState(time=0.0, time_step=60.0)
        pump.step(state, action=0.0)
        
        assert pump.ports[pump.output_water_port_name].flows['mass'] == 0.0
    
    def test_simple_pump_temperature_passthrough(self):
        """Test that SimplePump passes through temperature unchanged."""
        pump = SimplePump(name='pump_1', mass_flow_rate=1.0)
        pump.create_ports()
        
        inlet_temp = 75.0 + 273.15
        pump.ports[pump.input_water_port_name].T = inlet_temp
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        state = SimulationState(time=0.0, time_step=60.0)
        pump.step(state, action=1.0)
        
        outlet_temp = pump.ports[pump.output_water_port_name].T
        assert math.isclose(outlet_temp, inlet_temp, rel_tol=1e-6)
    
    def test_simple_pump_heat_flow_calculation(self):
        """Test that heat flow is calculated correctly."""
        pump = SimplePump(name='pump_1', mass_flow_rate=1.0)
        pump.create_ports()
        
        inlet_temp = 50.0 + 273.15
        mfr = 1.0
        pump.ports[pump.input_water_port_name].T = inlet_temp
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        state = SimulationState(time=0.0, time_step=60.0)
        pump.step(state, action=1.0)
        
        # Heat flow = mass_flow_rate * cp * temperature
        expected_heat = mfr * WATER.cp * inlet_temp
        actual_heat = pump.ports[pump.output_water_port_name].flows['heat']
        
        assert math.isclose(actual_heat, expected_heat, rel_tol=1e-6)
    
    def test_simple_pump_variable_action(self):
        """Test SimplePump with different action values."""
        pump = SimplePump(name='pump_1', mass_flow_rate=2.0)
        pump.create_ports()
        
        pump.ports[pump.input_water_port_name].T = 50.0 + 273.15
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        actions = [0.0, 0.25, 0.5, 0.75, 1.0]
        expected_flows = [a * 2.0 for a in actions]
        
        for action, expected_flow in zip(actions, expected_flows):
            pump.step(SimulationState(time=0.0, time_step=60.0), action=action)
            assert math.isclose(pump.ports[pump.output_water_port_name].flows['mass'], 
                              expected_flow, rel_tol=1e-6)


class TestPumpInitialization:
    """Test suite for Pump (base class) initialization."""
    
    def test_pump_creation_without_electricity(self):
        """Test Pump creation without electricity input."""
        # Create a concrete subclass for testing
        pump = ConcretePump(name='pump_1', include_electricity_input=False)
        
        assert pump.name == 'pump_1'
        assert pump.input_water_port_name == 'pump_1_water_input_port'
        assert pump.output_water_port_name == 'pump_1_water_output_port'
        assert pump.electricity_port_name is None
    
    def test_pump_creation_with_electricity(self):
        """Test Pump creation with electricity input."""
        pump = ConcretePump(name='pump_1', include_electricity_input=True)
        
        assert pump.electricity_port_name == 'pump_1_electricity_port'
    
    def test_pump_port_names(self):
        """Test that Pump port names are generated correctly."""
        pump = ConcretePump(name='my_pump', include_electricity_input=True)
        
        assert 'my_pump' in pump.input_water_port_name
        assert 'my_pump' in pump.output_water_port_name
        assert 'my_pump' in pump.electricity_port_name
    
    def test_pump_ports_creation_without_electricity(self):
        """Test port creation without electricity."""
        pump = ConcretePump(name='pump_1', include_electricity_input=False)
        pump.create_ports()
        
        assert pump.input_water_port_name in pump.ports
        assert pump.output_water_port_name in pump.ports
        assert pump.electricity_port_name not in pump.ports
    
    def test_pump_ports_creation_with_electricity(self):
        """Test port creation with electricity."""
        pump = ConcretePump(name='pump_1', include_electricity_input=True)
        pump.create_ports()
        
        assert pump.input_water_port_name in pump.ports
        assert pump.output_water_port_name in pump.ports
        assert pump.electricity_port_name in pump.ports


class TestPumpOperation:
    """Test suite for Pump operation."""
    
    def test_pump_step_without_electricity(self):
        """Test Pump step method without electricity input."""
        pump = ConcretePump(name='pump_1', include_electricity_input=False)
        pump.create_ports()
        
        pump.ports[pump.input_water_port_name].T = 50.0 + 273.15
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        state = SimulationState(time=0.0, time_step=60.0)
        pump.step(state, action=1.0)
        
        # Check that flow is set correctly
        assert pump.ports[pump.output_water_port_name].flows['mass'] == 1.0
        # Check that electricity port is not modified
        assert pump.electricity_port_name is None
    
    def test_pump_step_with_electricity(self):
        """Test Pump step method with electricity input."""
        pump = ConcretePump(name='pump_1', include_electricity_input=True)
        pump.create_ports()
        
        pump.ports[pump.input_water_port_name].T = 50.0 + 273.15
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        state = SimulationState(time=0.0, time_step=60.0)
        pump.step(state, action=1.0)
        
        # Check that electricity consumption is calculated
        # Expected: flow * pressure_rise / efficiency = 1.0 * 10000 / 0.85
        expected_power = (1.0 * 10000) / 0.85
        actual_power = pump.ports[pump.electricity_port_name].flows['electricity']
        
        assert math.isclose(actual_power, expected_power, rel_tol=1e-6)
    
    def test_pump_temperature_passthrough(self):
        """Test that Pump passes through temperature unchanged."""
        pump = ConcretePump(name='pump_1', include_electricity_input=True)
        pump.create_ports()
        
        inlet_temp = 65.0 + 273.15
        pump.ports[pump.input_water_port_name].T = inlet_temp
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        state = SimulationState(time=0.0, time_step=60.0)
        pump.step(state, action=1.0)
        
        outlet_temp = pump.ports[pump.output_water_port_name].T
        assert math.isclose(outlet_temp, inlet_temp, rel_tol=1e-6)
    
    def test_pump_heat_flow_conservation(self):
        """Test that heat flow is conserved through the pump."""
        pump = ConcretePump(name='pump_1', include_electricity_input=True)
        pump.create_ports()
        
        inlet_temp = 50.0 + 273.15
        mfr = 1.0
        pump.ports[pump.input_water_port_name].T = inlet_temp
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        state = SimulationState(time=0.0, time_step=60.0)
        pump.step(state, action=1.0)
        
        inlet_heat = pump.ports[pump.input_water_port_name].flows['heat']
        outlet_heat = pump.ports[pump.output_water_port_name].flows['heat']
        
        # Since temperature doesn't change, heat flow should be conserved
        assert math.isclose(inlet_heat, outlet_heat, rel_tol=1e-6)


class TestPumpElectricityConsumption:
    """Test suite for pump electricity consumption calculations."""
    
    def test_electricity_proportional_to_flow(self):
        """Test that electricity consumption is proportional to flow rate."""
        pump = ConcretePump(name='pump_1', include_electricity_input=True)
        pump.create_ports()
        
        pump.ports[pump.input_water_port_name].T = 50.0 + 273.15
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        state = SimulationState(time=0.0, time_step=60.0)
        
        # Test different flow rates
        actions = [0.5, 1.0]
        powers = []
        
        for action in actions:
            pump.step(state, action=action)
            power = pump.ports[pump.electricity_port_name].flows['electricity']
            powers.append(power)
        
        # Power should scale with flow rate
        assert powers[1] > powers[0]
        ratio = powers[1] / powers[0]
        expected_ratio = actions[1] / actions[0]
        assert math.isclose(ratio, expected_ratio, rel_tol=1e-6)
    
    def test_zero_flow_zero_power(self):
        """Test that zero flow results in zero power consumption."""
        pump = ConcretePump(name='pump_1', include_electricity_input=True)
        pump.create_ports()
        
        pump.ports[pump.input_water_port_name].T = 50.0 + 273.15
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        state = SimulationState(time=0.0, time_step=60.0)
        pump.step(state, action=0.0)
        
        power = pump.ports[pump.electricity_port_name].flows['electricity']
        assert power == 0.0


class TestPumpEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_very_high_inlet_temperature(self):
        """Test pump with very high inlet temperature."""
        pump = SimplePump(name='pump_1', mass_flow_rate=1.0)
        pump.create_ports()
        
        high_temp = 100.0 + 273.15
        pump.ports[pump.input_water_port_name].T = high_temp
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        state = SimulationState(time=0.0, time_step=60.0)
        pump.step(state, action=1.0)
        
        assert not np.isnan(pump.ports[pump.output_water_port_name].T)
        assert math.isclose(pump.ports[pump.output_water_port_name].T, high_temp)
    
    def test_very_low_inlet_temperature(self):
        """Test pump with very low inlet temperature."""
        pump = SimplePump(name='pump_1', mass_flow_rate=1.0)
        pump.create_ports()
        
        low_temp = 0.0 + 273.15
        pump.ports[pump.input_water_port_name].T = low_temp
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        state = SimulationState(time=0.0, time_step=60.0)
        pump.step(state, action=1.0)
        
        assert not np.isnan(pump.ports[pump.output_water_port_name].T)
        assert math.isclose(pump.ports[pump.output_water_port_name].T, low_temp)
    
    def test_very_small_mass_flow_rate(self):
        """Test SimplePump with very small mass flow rate."""
        pump = SimplePump(name='pump_1', mass_flow_rate=0.001)
        pump.create_ports()
        
        pump.ports[pump.input_water_port_name].T = 50.0 + 273.15
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        state = SimulationState(time=0.0, time_step=60.0)
        pump.step(state, action=1.0)
        
        assert pump.ports[pump.output_water_port_name].flows['mass'] == 0.001
    
    def test_very_large_mass_flow_rate(self):
        """Test SimplePump with very large mass flow rate."""
        pump = SimplePump(name='pump_1', mass_flow_rate=1000.0)
        pump.create_ports()
        
        pump.ports[pump.input_water_port_name].T = 50.0 + 273.15
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        state = SimulationState(time=0.0, time_step=60.0)
        pump.step(state, action=1.0)
        
        assert pump.ports[pump.output_water_port_name].flows['mass'] == 1000.0
    
    def test_action_greater_than_one(self):
        """Test SimplePump with action > 1.0 (over-modulation)."""
        pump = SimplePump(name='pump_1', mass_flow_rate=1.0)
        pump.create_ports()
        
        pump.ports[pump.input_water_port_name].T = 50.0 + 273.15
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        state = SimulationState(time=0.0, time_step=60.0)
        pump.step(state, action=1.5)
        
        # Pump should produce flow greater than nominal
        assert pump.ports[pump.output_water_port_name].flows['mass'] == 1.5


class TestPumpIntegration:
    """Integration tests for pumps in simulation environments."""
    
    def test_simple_pump_with_environment(self):
        """Test SimplePump in a simulation environment."""
        # This is a basic integration test
        pump = SimplePump(name='test_pump', mass_flow_rate=1.0)
        pump.create_ports()
        
        # Basic sanity checks
        assert pump.input_water_port_name in pump.ports
        assert pump.output_water_port_name in pump.ports
        
        pump.ports[pump.input_water_port_name].T = 50.0 + 273.15
        pump.ports[pump.input_water_port_name].flows = {'mass': None, 'heat': 0.0}
        
        state = SimulationState(time=0.0, time_step=60.0)
        pump.step(state, action=0.8)
        
        # Verify output
        assert pump.ports[pump.output_water_port_name].flows['mass'] == 0.8


# Concrete implementation of Pump for testing
class ConcretePump(Pump):
    """Concrete implementation of Pump for testing purposes."""
    
    def get_flow(self, state: SimulationState, action):
        """Return a constant flow rate based on action."""
        return 1.0 * (action if action is not None else 1.0)
    
    def get_pressure_rise(self, state: SimulationState, action):
        """Return a constant pressure rise of 10 kPa."""
        return 10000.0  # Pa
    
    def get_efficiency(self, state: SimulationState, action):
        """Return a constant efficiency of 0.85."""
        return 0.85
