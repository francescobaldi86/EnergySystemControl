# tests/unit/components/test_utilities.py
import pytest
import numpy as np
from energy_system_control.components.composite_components.inverters import Inverter
from energy_system_control.sim.state import SimulationState


class TestInverterInitialization:
    """Test Inverter instantiation and basic properties."""
    
    def test_inverter_default_efficiency(self):
        """Test inverter creation with default efficiency."""
        inverter = Inverter(name="test_inverter")
        assert inverter.name == "test_inverter"
        assert inverter.efficiency == 0.92
    
    def test_inverter_custom_efficiency(self):
        """Test inverter creation with custom efficiency."""
        inverter = Inverter(name="test_inverter", efficiency=0.95)
        assert inverter.efficiency == 0.95
    
    def test_inverter_port_names(self):
        """Test that all required ports are created with correct names."""
        inverter = Inverter(name="test_inverter")
        assert inverter.PV_port_name == "test_inverter_PV_input_port"
        assert inverter.AC_output_port_name == "test_inverter_output_port"
        assert inverter.ESS_port_name == "test_inverter_ESS_port"
        assert inverter.grid_port_name == "test_inverter_grid_input_port"
    
    def test_inverter_ports_creation(self):
        """Test that ports are created and stored correctly."""
        inverter = Inverter(name="test_inverter")
        inverter.create_ports()
        
        assert inverter.ESS_port_name in inverter.ports
        assert inverter.PV_port_name in inverter.ports
        assert inverter.AC_output_port_name in inverter.ports
        assert inverter.grid_port_name in inverter.ports
        
        # All ports should be electricity ports
        for port in inverter.ports.values():
            assert 'electricity' in port.layers


class TestInverterMethods:
    """Test Inverter methods."""
    
    @pytest.fixture
    def inverter_with_ports(self):
        """Fixture to create inverter with initialized ports."""
        inverter = Inverter(name="test_inverter", efficiency=0.90)
        inverter.create_ports()
        return inverter
    
    @pytest.fixture
    def simulation_state(self):
        """Fixture to create a basic simulation state."""
        return SimulationState(time=0, time_step=900, time_id=0)  # 15 minutes
    
    def test_get_efficiency(self, inverter_with_ports):
        """Test get_efficiency method."""
        assert inverter_with_ports.get_efficiency() == 0.90
    
    def test_step_only_pv_input_no_load_no_battery(self, inverter_with_ports, simulation_state):
        """Test step when only PV input is available, no AC load, no battery action."""
        pv_power = 5.0  # 5 kW
        inverter_with_ports.ports[inverter_with_ports.PV_port_name].flow['electricity'] = pv_power
        inverter_with_ports.ports[inverter_with_ports.AC_output_port_name].flow['electricity'] = 0.0
        
        inverter_with_ports.step(simulation_state, action=0.0)
        
        # AC_input = (PV_power + 0) * efficiency + 0
        expected_ac_input = pv_power * 0.90
        expected_grid_flow = -expected_ac_input
        
        assert np.isclose(inverter_with_ports.ports[inverter_with_ports.grid_port_name].flow['electricity'], expected_grid_flow)
        assert inverter_with_ports.ports[inverter_with_ports.ESS_port_name].flow['electricity'] == 0.0
    
    def test_step_pv_with_ac_load(self, inverter_with_ports, simulation_state):
        """Test step when PV input and AC load are present."""
        pv_power = 3.0  # 3 kW
        ac_load = 2.0   # 2 kW load (positive flow)
        efficiency = 0.90
        
        inverter_with_ports.ports[inverter_with_ports.PV_port_name].flow['electricity'] = pv_power
        inverter_with_ports.ports[inverter_with_ports.AC_output_port_name].flow['electricity'] = ac_load
        
        inverter_with_ports.step(simulation_state, action=0.0)
        
        # AC_input = (PV_power + 0) * efficiency + AC_load
        expected_ac_input = pv_power * efficiency + ac_load
        expected_grid_flow = -expected_ac_input
        
        assert np.isclose(inverter_with_ports.ports[inverter_with_ports.grid_port_name].flow['electricity'], expected_grid_flow)
    
    def test_step_pv_with_battery_discharge(self, inverter_with_ports, simulation_state):
        """Test step with battery discharging."""
        pv_power = 2.0      # 2 kW
        battery_discharge = 1.0  # 1 kW discharge
        efficiency = 0.90
        
        inverter_with_ports.ports[inverter_with_ports.PV_port_name].flow['electricity'] = pv_power
        inverter_with_ports.ports[inverter_with_ports.AC_output_port_name].flow['electricity'] = 0.0
        
        inverter_with_ports.step(simulation_state, action=battery_discharge)
        
        # AC_input = (PV_power + battery_discharge) * efficiency + 0
        expected_ac_input = (pv_power + battery_discharge) * efficiency
        expected_grid_flow = -expected_ac_input
        expected_ess_flow = battery_discharge
        
        assert np.isclose(inverter_with_ports.ports[inverter_with_ports.grid_port_name].flow['electricity'], expected_grid_flow)
        assert np.isclose(inverter_with_ports.ports[inverter_with_ports.ESS_port_name].flow['electricity'], expected_ess_flow)
    
    def test_step_pv_with_battery_charge(self, inverter_with_ports, simulation_state):
        """Test step with battery charging (negative action)."""
        pv_power = 5.0       # 5 kW
        battery_charge = -1.0  # -1 kW charge
        efficiency = 0.90
        ac_load = 1.0
        
        inverter_with_ports.ports[inverter_with_ports.PV_port_name].flow['electricity'] = pv_power
        inverter_with_ports.ports[inverter_with_ports.AC_output_port_name].flow['electricity'] = ac_load
        
        inverter_with_ports.step(simulation_state, action=battery_charge)
        
        # AC_input = (PV_power + battery_charge) * efficiency + AC_load
        expected_ac_input = (pv_power + battery_charge) * efficiency + ac_load
        expected_grid_flow = -expected_ac_input
        expected_ess_flow = battery_charge
        
        assert np.isclose(inverter_with_ports.ports[inverter_with_ports.grid_port_name].flow['electricity'], expected_grid_flow)
        assert np.isclose(inverter_with_ports.ports[inverter_with_ports.ESS_port_name].flow['electricity'], expected_ess_flow)
    
    def test_step_no_pv_battery_discharge(self, inverter_with_ports, simulation_state):
        """Test step with no PV, only battery discharge to supply AC load."""
        battery_discharge = 3.0  # 3 kW discharge
        ac_load = 2.0
        efficiency = 0.90
        
        inverter_with_ports.ports[inverter_with_ports.PV_port_name].flow['electricity'] = 0.0
        inverter_with_ports.ports[inverter_with_ports.AC_output_port_name].flow['electricity'] = ac_load
        
        inverter_with_ports.step(simulation_state, action=battery_discharge)
        
        # AC_input = (0 + battery_discharge) * efficiency + AC_load
        expected_ac_input = battery_discharge * efficiency + ac_load
        expected_grid_flow = -expected_ac_input
        
        assert np.isclose(inverter_with_ports.ports[inverter_with_ports.grid_port_name].flow['electricity'], expected_grid_flow)
    
    def test_step_zero_action(self, inverter_with_ports, simulation_state):
        """Test step with zero action (no battery exchange)."""
        pv_power = 4.0
        efficiency = 0.90
        
        inverter_with_ports.ports[inverter_with_ports.PV_port_name].flow['electricity'] = pv_power
        inverter_with_ports.ports[inverter_with_ports.AC_output_port_name].flow['electricity'] = 0.0
        
        inverter_with_ports.step(simulation_state, action=0.0)
        
        expected_grid_flow = -pv_power * efficiency
        assert np.isclose(inverter_with_ports.ports[inverter_with_ports.grid_port_name].flow['electricity'], expected_grid_flow)
        assert inverter_with_ports.ports[inverter_with_ports.ESS_port_name].flow['electricity'] == 0.0


class TestInverterEnergyBalance:
    """Test energy balance properties of the inverter."""
    
    @pytest.fixture
    def inverter_with_ports(self):
        inverter = Inverter(name="test_inverter", efficiency=0.92)
        inverter.create_ports()
        return inverter
    
    @pytest.fixture
    def simulation_state(self):
        return SimulationState(time=0, time_step=900, time_id=0)
    
    def test_efficiency_impact_on_balance(self, inverter_with_ports, simulation_state):
        """Verify that efficiency correctly impacts the energy balance."""
        pv_power = 10.0
        efficiency = 0.92
        
        inverter_with_ports.ports[inverter_with_ports.PV_port_name].flow['electricity'] = pv_power
        inverter_with_ports.ports[inverter_with_ports.AC_output_port_name].flow['electricity'] = 0.0
        
        inverter_with_ports.step(simulation_state, action=0.0)
        
        grid_flow = inverter_with_ports.ports[inverter_with_ports.grid_port_name].flow['electricity']
        
        # Efficiency means that lower AC output for the same PV input
        assert grid_flow == -pv_power * efficiency
    
    def test_combined_pv_and_battery_supply(self, inverter_with_ports, simulation_state):
        """Test energy balance when both PV and battery supply AC load."""
        pv_power = 2.0
        battery_power = 1.5
        ac_load = 3.0
        efficiency = 0.92
        
        inverter_with_ports.ports[inverter_with_ports.PV_port_name].flow['electricity'] = pv_power
        inverter_with_ports.ports[inverter_with_ports.AC_output_port_name].flow['electricity'] = ac_load
        
        inverter_with_ports.step(simulation_state, action=battery_power)
        
        # AC_input = (PV + Battery) * efficiency + AC_load
        expected_ac_input = (pv_power + battery_power) * efficiency + ac_load
        expected_grid_flow = -expected_ac_input
        
        assert np.isclose(
            inverter_with_ports.ports[inverter_with_ports.grid_port_name].flow['electricity'],
            expected_grid_flow
        )