# tests/unit/components/test_utilities.py
"""
Comprehensive tests for the Inverter composite component and its subcomponents:
- Inverter (CompositeComponent)
- DC Bus (Bus)
- AC Bus (Bus)
- InverterConverter (ImplicitComponent)
- FixedEfficiencyInverterConverter
"""

import pytest
import numpy as np
from energy_system_control.components.composite_components.inverters import (
    Inverter,
    InverterConverter,
    FixedEfficiencyInverterConverter,
)
from energy_system_control.components.base import Bus
from energy_system_control.sim.state import SimulationState
from energy_system_control.core.base_classes import InitContext


class TestInverterInitialization:
    """Test Inverter instantiation and basic properties."""

    def test_inverter_basic_creation(self):
        """Test that inverter can be created with default parameters."""
        inverter = Inverter(name="test_inverter")
        assert inverter.name == "test_inverter"
        
    def test_inverter_custom_design_efficiency(self):
        """Test inverter creation with custom design efficiency."""
        inverter = Inverter(name="test_inverter", design_efficiency=0.95)
        assert inverter.converter.design_efficiency == 0.95

    def test_inverter_default_design_efficiency(self):
        """Test that default efficiency is 0.92."""
        inverter = Inverter(name="test_inverter")
        assert inverter.converter.design_efficiency == 0.92

    def test_inverter_port_names_defined(self):
        """Test that port name attributes are correctly set."""
        inverter = Inverter(name="test_inverter")
        assert inverter.PV_port_name == "test_inverter_PV_input_port"
        assert inverter.AC_output_port_name == "test_inverter_AC_output_port"
        assert inverter.grid_port_name == "test_inverter_grid_input_port"
        assert inverter.ESS_port_name == "test_inverter_ESS_port"

    def test_inverter_internal_components_structure(self):
        """Test that inverter has the correct internal components."""
        inverter = Inverter(name="test_inverter")
        internal_components = inverter.get_internal_components()
        
        assert "test_inverter_dc_bus" in internal_components
        assert "test_inverter_ac_bus" in internal_components
        assert "test_inverter_inverter" in internal_components
        
        # Verify component types
        assert isinstance(internal_components["test_inverter_dc_bus"], Bus)
        assert isinstance(internal_components["test_inverter_ac_bus"], Bus)
        assert isinstance(internal_components["test_inverter_inverter"], FixedEfficiencyInverterConverter)

    def test_inverter_internal_connections(self):
        """Test that internal connections are properly defined."""
        inverter = Inverter(name="test_inverter")
        connections = inverter.get_internal_connections()
        
        assert len(connections) == 2
        # Should have connections from buses to inverter converter
        connection_set = set(connections)
        expected_connections = {
            ("test_inverter_dc_bus_internal_port", "test_inverter_inverter_dc_port"),
            ("test_inverter_ac_bus_internal_port", "test_inverter_inverter_ac_port"),
        }
        assert connection_set == expected_connections

    def test_inverter_efficiency_type_fixed(self):
        """Test that efficiency_type='fixed' creates FixedEfficiencyInverterConverter."""
        inverter = Inverter(
            name="test_inverter", 
            design_efficiency=0.90, 
            efficiency_type="fixed"
        )
        assert isinstance(inverter.converter, FixedEfficiencyInverterConverter)
        assert inverter.converter.design_efficiency == 0.90


class TestDCBusComponent:
    """Test DC Bus functionality within the Inverter."""

    def test_dc_bus_creation(self):
        """Test that DC bus is correctly created with expected ports."""
        inverter = Inverter(name="test_inverter")
        dc_bus = inverter.dc_bus
        dc_bus.create_ports()
        
        assert dc_bus.name == "test_inverter_dc_bus"
        # DC bus should have 3 ports: PV input, ESS, and internal port
        assert len(dc_bus.ports) == 3
        
    def test_dc_bus_port_structure(self):
        """Test that DC bus has the correct ports defined."""
        inverter = Inverter(name="test_inverter")
        dc_bus = inverter.dc_bus
        dc_bus.create_ports()
        
        expected_ports = {
            "test_inverter_PV_input_port",
            "test_inverter_ESS_port",
            "test_inverter_dc_bus_internal_port",
        }
        assert set(dc_bus.ports.keys()) == expected_ports

    def test_dc_bus_port_layers(self):
        """Test that all DC bus ports have electricity layer."""
        inverter = Inverter(name="test_inverter")
        dc_bus = inverter.dc_bus
        dc_bus.create_ports()
        
        for port in dc_bus.ports.values():
            assert "electricity" in port.layers

    def test_dc_bus_balance_with_missing_port(self):
        """Test DC bus balance when one port flow is missing."""
        inverter = Inverter(name="test_inverter")
        dc_bus = inverter.dc_bus
        dc_bus.create_ports()
        
        # Set flows for PV and internal port
        dc_bus.ports["test_inverter_PV_input_port"].flows["electricity"] = 5.0
        dc_bus.ports["test_inverter_dc_bus_internal_port"].flows["electricity"] = 2.0
        # ESS port flow is None - should be calculated
        
        is_solved, updated_ports = dc_bus.balance(SimulationState(time=0, time_step=900, time_id=0))
        
        assert is_solved is True
        assert "test_inverter_ESS_port" in updated_ports
        # Balance: 5.0 + 2.0 + ESS = 0 => ESS = -7.0
        assert dc_bus.ports["test_inverter_ESS_port"].flows["electricity"] == -7.0

    def test_dc_bus_balance_flow_conservation(self):
        """Test that DC bus respects flow conservation law."""
        inverter = Inverter(name="test_inverter")
        dc_bus = inverter.dc_bus
        dc_bus.create_ports()
        
        pv_flow = 3.5
        ess_flow = -1.5
        
        dc_bus.ports["test_inverter_PV_input_port"].flows["electricity"] = pv_flow
        dc_bus.ports["test_inverter_ESS_port"].flows["electricity"] = ess_flow
        # Internal port should be calculated
        
        is_solved, updated_ports = dc_bus.balance(SimulationState(time=0, time_step=900, time_id=0))
        
        assert is_solved is True
        # Sum should be zero: pv_flow + ess_flow + internal = 0
        internal_flow = dc_bus.ports["test_inverter_dc_bus_internal_port"].flows["electricity"]
        total_flow = pv_flow + ess_flow + internal_flow
        assert np.isclose(total_flow, 0.0)


class TestACBusComponent:
    """Test AC Bus functionality within the Inverter."""

    def test_ac_bus_creation(self):
        """Test that AC bus is correctly created with expected ports."""
        inverter = Inverter(name="test_inverter")
        ac_bus = inverter.ac_bus
        ac_bus.create_ports()
        
        assert ac_bus.name == "test_inverter_ac_bus"
        # AC bus should have 3 ports: AC output, grid, and internal port
        assert len(ac_bus.ports) == 3

    def test_ac_bus_port_structure(self):
        """Test that AC bus has the correct ports defined."""
        inverter = Inverter(name="test_inverter")
        ac_bus = inverter.ac_bus
        ac_bus.create_ports()
        
        expected_ports = {
            "test_inverter_AC_output_port",
            "test_inverter_grid_input_port",
            "test_inverter_ac_bus_internal_port",
        }
        assert set(ac_bus.ports.keys()) == expected_ports

    def test_ac_bus_balance_with_missing_port(self):
        """Test AC bus balance when one port flow is missing."""
        inverter = Inverter(name="test_inverter")
        ac_bus = inverter.ac_bus
        ac_bus.create_ports()
        
        # Set flows for AC output and internal port
        ac_bus.ports["test_inverter_AC_output_port"].flows["electricity"] = 2.0
        ac_bus.ports["test_inverter_ac_bus_internal_port"].flows["electricity"] = 1.5
        # Grid port flow is None - should be calculated
        
        is_solved, updated_ports = ac_bus.balance(SimulationState(time=0, time_step=900, time_id=0))
        
        assert is_solved is True
        assert "test_inverter_grid_input_port" in updated_ports
        # Balance: 2.0 + 1.5 + grid = 0 => grid = -3.5
        assert ac_bus.ports["test_inverter_grid_input_port"].flows["electricity"] == -3.5


class TestInverterConverterComponent:
    """Test InverterConverter (DC/AC conversion logic)."""

    def test_inverter_converter_creation(self):
        """Test InverterConverter initialization."""
        converter = InverterConverter(name="test_converter", design_efficiency=0.90)
        
        assert converter.name == "test_converter"
        assert converter.design_efficiency == 0.90
        assert converter.ac_port_name == "test_converter_ac_port"
        assert converter.dc_port_name == "test_converter_dc_port"

    def test_inverter_converter_ports(self):
        """Test that InverterConverter has AC and DC ports."""
        converter = InverterConverter(name="test_converter", design_efficiency=0.90)
        converter.create_ports()
        
        assert len(converter.ports) == 2
        assert "test_converter_ac_port" in converter.ports
        assert "test_converter_dc_port" in converter.ports

    def test_inverter_converter_both_ports_none(self):
        """Test balance when both ports have no flow (unsolvable)."""
        converter = InverterConverter(name="test_converter", design_efficiency=0.90)
        converter.create_ports()
        state = SimulationState(time=0, time_step=900, time_id=0)
        
        # Both ports have None flow
        is_solved, updated_ports = converter.balance(state)
        
        assert is_solved is False
        assert updated_ports == []

    def test_inverter_converter_dc_to_ac_conversion(self):
        """Test DC to AC conversion (positive DC flow, negative AC flow)."""
        converter = FixedEfficiencyInverterConverter(name="test_converter", design_efficiency=0.90)
        converter.create_ports()
        state = SimulationState(time=0, time_step=900, time_id=0)
        
        # Set DC flow (positive means DC supply)
        converter.ports["test_converter_dc_port"].flows["electricity"] = 10.0
        # AC port is None and should be calculated
        
        is_solved, updated_ports = converter.balance(state)
        
        assert is_solved is True
        assert "test_converter_ac_port" in updated_ports
        # DC to AC: AC_flow = -DC_flow * efficiency
        expected_ac_flow = -10.0 * 0.90
        assert converter.ports["test_converter_ac_port"].flows["electricity"] == expected_ac_flow

    def test_inverter_converter_ac_to_dc_conversion(self):
        """Test AC to DC conversion (positive AC flow, negative DC flow)."""
        converter = FixedEfficiencyInverterConverter(name="test_converter", design_efficiency=0.90)
        converter.create_ports()
        state = SimulationState(time=0, time_step=900, time_id=0)
        
        # Set AC flow (positive means AC supply going to DC side)
        converter.ports["test_converter_ac_port"].flows["electricity"] = 8.0
        # DC port is None and should be calculated
        
        is_solved, updated_ports = converter.balance(state)
        
        assert is_solved is True
        assert "test_converter_dc_port" in updated_ports
        # AC to DC with efficiency loss: DC_flow = -AC_flow * efficiency
        expected_dc_flow = -8.0 * 0.90
        assert np.isclose(converter.ports["test_converter_dc_port"].flows["electricity"], expected_dc_flow)

    def test_inverter_converter_negative_dc_flow(self):
        """Test negative DC flow (battery charging from AC)."""
        converter = FixedEfficiencyInverterConverter(name="test_converter", design_efficiency=0.90)
        converter.create_ports()
        state = SimulationState(time=0, time_step=900, time_id=0)
        
        # Set negative DC flow (charging)
        converter.ports["test_converter_dc_port"].flows["electricity"] = -5.0
        
        is_solved, updated_ports = converter.balance(state)
        
        assert is_solved is True
        # Negative DC flow: AC_flow = -DC_flow / efficiency
        expected_ac_flow = -(-5.0) / 0.90
        assert np.isclose(converter.ports["test_converter_ac_port"].flows["electricity"], expected_ac_flow)

    def test_inverter_converter_negative_ac_flow(self):
        """Test negative AC flow (battery discharging to AC)."""
        converter = FixedEfficiencyInverterConverter(name="test_converter", design_efficiency=0.90)
        converter.create_ports()
        state = SimulationState(time=0, time_step=900, time_id=0)
        
        # Set negative AC flow (discharging)
        converter.ports["test_converter_ac_port"].flows["electricity"] = -6.0
        
        is_solved, updated_ports = converter.balance(state)
        
        assert is_solved is True
        # Negative AC flow: DC_flow = -AC_flow / efficiency
        expected_dc_flow = -(-6.0) / 0.90
        assert np.isclose(converter.ports["test_converter_dc_port"].flows["electricity"], expected_dc_flow)


class TestFixedEfficiencyInverterConverter:
    """Test FixedEfficiencyInverterConverter specific functionality."""

    def test_fixed_efficiency_converter_creation(self):
        """Test creation of FixedEfficiencyInverterConverter."""
        converter = FixedEfficiencyInverterConverter(name="test_converter", design_efficiency=0.95)
        assert isinstance(converter, InverterConverter)
        assert isinstance(converter, FixedEfficiencyInverterConverter)

    def test_fixed_efficiency_getter(self):
        """Test that get_efficiency returns design_efficiency."""
        converter = FixedEfficiencyInverterConverter(name="test_converter", design_efficiency=0.88)
        assert converter.get_efficiency() == 0.88

    def test_fixed_efficiency_values(self):
        """Test efficiency with various values."""
        for eff_val in [0.80, 0.85, 0.90, 0.95]:
            converter = FixedEfficiencyInverterConverter(
                name="test_converter", design_efficiency=eff_val
            )
            assert converter.get_efficiency() == eff_val

    def test_fixed_efficiency_conversion_accuracy(self):
        """Test that conversion calculations use correct efficiency."""
        efficiency = 0.85
        converter = FixedEfficiencyInverterConverter(
            name="test_converter", design_efficiency=efficiency
        )
        converter.create_ports()
        state = SimulationState(time=0, time_step=900, time_id=0)
        
        dc_flow = 20.0
        converter.ports["test_converter_dc_port"].flows["electricity"] = dc_flow
        
        converter.balance(state)
        
        ac_flow = converter.ports["test_converter_ac_port"].flows["electricity"]
        expected_ac_flow = -dc_flow * efficiency
        assert np.isclose(ac_flow, expected_ac_flow)


class TestInverterIntegration:
    """Integration tests for complete Inverter systems."""

    def test_inverter_complete_structure(self):
        """Test that a complete inverter has all components properly connected."""
        inverter = Inverter(
            name="main_inverter",
            design_efficiency=0.92,
            efficiency_type="fixed"
        )
        
        # Verify structure
        assert inverter.name == "main_inverter"
        assert hasattr(inverter, "dc_bus")
        assert hasattr(inverter, "ac_bus")
        assert hasattr(inverter, "converter")
        
        internal_comps = inverter.get_internal_components()
        assert len(internal_comps) == 3
        
        connections = inverter.get_internal_connections()
        assert len(connections) == 2

    def test_inverter_with_different_efficiencies(self):
        """Test creating multiple inverters with different efficiencies."""
        efficiencies = [0.80, 0.85, 0.90, 0.95]
        
        inverters = [
            Inverter(name=f"inv_{eff}", design_efficiency=eff)
            for eff in efficiencies
        ]
        
        for inverter, expected_eff in zip(inverters, efficiencies):
            assert inverter.converter.design_efficiency == expected_eff
            assert inverter.converter.get_efficiency() == expected_eff

    def test_inverter_port_independence(self):
        """Test that multiple inverters have independent port names."""
        inv1 = Inverter(name="inverter_1")
        inv2 = Inverter(name="inverter_2")
        
        # Verify port names are independent
        assert inv1.PV_port_name != inv2.PV_port_name
        assert inv1.AC_output_port_name != inv2.AC_output_port_name
        assert inv1.grid_port_name != inv2.grid_port_name
        assert inv1.ESS_port_name != inv2.ESS_port_name

    def test_inverter_internal_component_independence(self):
        """Test that internal components of different inverters are independent."""
        inv1 = Inverter(name="inverter_1", design_efficiency=0.90)
        inv2 = Inverter(name="inverter_2", design_efficiency=0.95)
        
        # Verify they are different objects
        assert inv1.dc_bus is not inv2.dc_bus
        assert inv1.ac_bus is not inv2.ac_bus
        assert inv1.converter is not inv2.converter
        
        # Verify different names
        assert inv1.dc_bus.name != inv2.dc_bus.name
        assert inv1.ac_bus.name != inv2.ac_bus.name
        assert inv1.converter.name != inv2.converter.name


class TestInverterConverterEdgeCases:
    """Test edge cases and error conditions."""

    def test_both_ports_have_flow_raises_error(self):
        """Test that having flow on both ports raises an error."""
        converter = FixedEfficiencyInverterConverter(name="test_converter", design_efficiency=0.90)
        converter.create_ports()
        state = SimulationState(time=0, time_step=900, time_id=0)
        
        # Set flows on both ports
        converter.ports["test_converter_ac_port"].flows["electricity"] = 5.0
        converter.ports["test_converter_dc_port"].flows["electricity"] = 5.0
        
        with pytest.raises(ValueError, match="Both ports of the inverter"):
            converter.balance(state)

    def test_high_efficiency_conversion(self):
        """Test conversion with very high efficiency."""
        converter = FixedEfficiencyInverterConverter(name="test_converter", design_efficiency=0.99)
        converter.create_ports()
        state = SimulationState(time=0, time_step=900, time_id=0)
        
        dc_flow = 100.0
        converter.ports["test_converter_dc_port"].flows["electricity"] = dc_flow
        
        is_solved, _ = converter.balance(state)
        assert is_solved is True
        
        ac_flow = converter.ports["test_converter_ac_port"].flows["electricity"]
        expected_ac_flow = -100.0 * 0.99
        assert np.isclose(ac_flow, expected_ac_flow)

    def test_low_efficiency_conversion(self):
        """Test conversion with realistic low efficiency."""
        converter = FixedEfficiencyInverterConverter(name="test_converter", design_efficiency=0.75)
        converter.create_ports()
        state = SimulationState(time=0, time_step=900, time_id=0)
        
        dc_flow = 50.0
        converter.ports["test_converter_dc_port"].flows["electricity"] = dc_flow
        
        is_solved, _ = converter.balance(state)
        assert is_solved is True
        
        ac_flow = converter.ports["test_converter_ac_port"].flows["electricity"]
        expected_ac_flow = -50.0 * 0.75
        assert np.isclose(ac_flow, expected_ac_flow)

    def test_zero_flow_handling(self):
        """Test handling of zero flow."""
        converter = FixedEfficiencyInverterConverter(name="test_converter", design_efficiency=0.90)
        converter.create_ports()
        state = SimulationState(time=0, time_step=900, time_id=0)
        
        converter.ports["test_converter_dc_port"].flows["electricity"] = 0.0
        
        is_solved, updated_ports = converter.balance(state)
        assert is_solved is True
        assert converter.ports["test_converter_ac_port"].flows["electricity"] == 0.0

    def test_very_small_flows(self):
        """Test handling of very small flows."""
        converter = FixedEfficiencyInverterConverter(name="test_converter", design_efficiency=0.90)
        converter.create_ports()
        state = SimulationState(time=0, time_step=900, time_id=0)
        
        small_flow = 1e-6
        converter.ports["test_converter_dc_port"].flows["electricity"] = small_flow
        
        is_solved, _ = converter.balance(state)
        assert is_solved is True
        
        ac_flow = converter.ports["test_converter_ac_port"].flows["electricity"]
        expected_ac_flow = -small_flow * 0.90
        assert np.isclose(ac_flow, expected_ac_flow)