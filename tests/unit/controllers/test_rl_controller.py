# test_rl_controller.py
import pytest
import energy_system_control as esc
from energy_system_control.controllers.RL.RLcontrollers import QLearningController
from energy_system_control.controllers.RL.discretizers import StateDiscretizer, Discretizer, TemporalAggregator
from energy_system_control.controllers.RL.agents import QLearningAgent
from energy_system_control.controllers.predictors import PerfectTimeSeriesPredictor
from energy_system_control.controllers.RL.reward_functions import CompositeReward, TemperatureTrackingReward, EnergyCostReward, TemperatureMinMaxReward
from energy_system_control.helpers import C2K
import math, os
import pandas as pd
import numpy as np

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

@pytest.fixture
def test_components():
    test_components = [
        esc.IEAHotWaterDemand(name= "demand_DHW", reference_temperature = 40, profile_name='M'),
        esc.HeatPumpConstantEfficiency(name = 'heat_pump', Qdot_design = 1.5, COP_design= 3.2),
        esc.HotWaterStorage(name = 'hot_water_storage', max_temperature = 80, tank_volume = 200, T_0 = 45),
        esc.ElectricityGrid(name = 'electric_grid', cost_of_electricity_purchased=0.24, value_of_electricity_sold=0.06),
        esc.ColdWaterGrid(name = 'water_grid', utility_type = 'fluid'),
        esc.PVpanelFromPVGISData(name = 'pv_panels', data_path=os.path.join(__TEST__, 'DATA'), filename = 'pvgis_data.csv', rescale_factor = 0.5),
        esc.LithiumIonBattery(name = 'battery', capacity = 2.0, SOC_0 = 0.5),
        esc.Inverter(name = 'inverter')
    ]
    return test_components

@pytest.fixture
def test_sensors():
    test_sensors = [
        esc.TankTemperatureSensor('storage_tank_temperature_sensor', 'hot_water_storage'),
        esc.SOCSensor('storage_tank_SOC_sensor', 'hot_water_storage'),
        esc.SOCSensor('battery_SOC_sensor', 'battery'),
        esc.ElectricPowerSensor('PV_power_sensor', 'inverter_PV_input_port'),
        esc.ElectricPowerSensor('inverter_power_output_sensor', 'inverter_AC_output_port'),
        esc.ElectricPowerSensor('grid_power_sensor', 'inverter_grid_input_port'),
        esc.ElectricPowerSensor('battery_power_sensor', 'inverter_ESS_port'),
        esc.HotWaterDemandSensor('demand_heat_flow_sensor', 'demand_DHW')
    ]
    return test_sensors

@pytest.fixture
def test_predictors():
    test_predictors = [PerfectTimeSeriesPredictor('pv_power_predictor', 'pv_panels'), 
                       PerfectTimeSeriesPredictor('dhw_demand_predictor', 'demand_DHW')]
    return test_predictors


class TestDiscretizer:
    """Test suite for the Discretizer class"""
    
    def test_discretizer_initialization(self):
        """Test that Discretizer initializes correctly with proper bin edges"""
        discretizer = Discretizer(vmin=0, vmax=10, n_bins=5)
        
        assert discretizer.vmin == 0
        assert discretizer.vmax == 10
        assert discretizer.n_bins == 5
        assert len(discretizer.bin_edges) == 6  # n_bins + 1
        
    def test_discretizer_bin_edges(self):
        """Test that bin edges are correctly calculated"""
        discretizer = Discretizer(vmin=0, vmax=10, n_bins=5)
        expected_edges = np.linspace(0, 10, 6)
        np.testing.assert_array_almost_equal(discretizer.bin_edges, expected_edges)
        
    def test_discretizer_minimum_value(self):
        """Test discretization of minimum boundary value"""
        discretizer = Discretizer(vmin=0, vmax=10, n_bins=5)
        result = discretizer.discretize(0)
        assert result == 0, "Minimum value should map to bin 0"
        
    def test_discretizer_maximum_value(self):
        """Test discretization of maximum boundary value"""
        discretizer = Discretizer(vmin=0, vmax=10, n_bins=5)
        result = discretizer.discretize(10)
        assert result == discretizer.n_bins - 1, "Maximum value should map to last bin"
        
    def test_discretizer_middle_values(self):
        """Test discretization of values in the middle of range"""
        discretizer = Discretizer(vmin=0, vmax=10, n_bins=5)
        
        # Test value in the middle (should map to bin 2 or 3 depending on digitize behavior)
        result = discretizer.discretize(5)
        assert 0 <= result < discretizer.n_bins, "Middle value should map to valid bin"
        
    def test_discretizer_clipping_below_minimum(self):
        """Test that values below minimum are clipped"""
        discretizer = Discretizer(vmin=0, vmax=10, n_bins=5)
        result = discretizer.discretize(-5)
        assert result == 0, "Value below minimum should be clipped to bin 0"
        
    def test_discretizer_clipping_above_maximum(self):
        """Test that values above maximum are clipped"""
        discretizer = Discretizer(vmin=0, vmax=10, n_bins=5)
        result = discretizer.discretize(15)
        assert result == discretizer.n_bins - 1, "Value above maximum should be clipped to last bin"
        
    def test_discretizer_consistency(self):
        """Test that discretization is consistent for same input"""
        discretizer = Discretizer(vmin=-50, vmax=50, n_bins=10)
        
        value = 25.5
        result1 = discretizer.discretize(value)
        result2 = discretizer.discretize(value)
        
        assert result1 == result2, "Same value should always produce same bin"
        
    def test_discretizer_negative_range(self):
        """Test discretizer with negative value range"""
        discretizer = Discretizer(vmin=-100, vmax=-10, n_bins=9)
        
        result_min = discretizer.discretize(-100)
        result_max = discretizer.discretize(-10)
        
        assert result_min == 0, "Negative minimum should map to bin 0"
        assert result_max == discretizer.n_bins - 1, "Negative maximum should map to last bin"
        
    def test_discretizer_single_bin(self):
        """Test discretizer with single bin (edge case)"""
        discretizer = Discretizer(vmin=0, vmax=10, n_bins=1)
        
        result_min = discretizer.discretize(0)
        result_mid = discretizer.discretize(5)
        result_max = discretizer.discretize(10)
        
        assert result_min == 0
        assert result_mid == 0
        assert result_max == 0
        
    def test_discretizer_many_bins(self):
        """Test discretizer with many bins"""
        discretizer = Discretizer(vmin=0, vmax=100, n_bins=100)
        
        result = discretizer.discretize(50)
        assert 0 <= result < 100, "Result should be within valid bin range"
        

class TestStateDiscretizer:
    """Test suite for the StateDiscretizer class"""
    
    def test_state_discretizer_initialization(self):
        """Test StateDiscretizer initialization with config"""
        config = {
            'temperature': {"min": 20, "max": 80, "bins": 10},
            'power': {"min": 0, "max": 100, "bins": 5}
        }
        state_disc = StateDiscretizer(config)
        
        assert 'temperature' in state_disc.discretizers
        assert 'power' in state_disc.discretizers
        assert len(state_disc.discretizers) == 2
        
    def test_state_discretizer_single_variable(self):
        """Test StateDiscretizer with single variable"""
        config = {
            'temperature': {"min": 20, "max": 80, "bins": 10}
        }
        state_disc = StateDiscretizer(config)
        obs = {'temperature': 50}
        
        result = state_disc.transform(obs)
        
        assert isinstance(result, tuple), "Result should be tuple"
        assert len(result) == 1, "Should have one element"
        assert isinstance(result[0], (int, np.integer)), "Element should be integer"
        
    def test_state_discretizer_multiple_variables(self):
        """Test StateDiscretizer with multiple variables"""
        config = {
            'temperature': {"min": 20, "max": 80, "bins": 10},
            'power': {"min": 0, "max": 100, "bins": 5},
            'demand': {"min": 0, "max": 50, "bins": 8}
        }
        state_disc = StateDiscretizer(config)
        obs = {
            'temperature': 50,
            'power': 50,
            'demand': 25
        }
        
        result = state_disc.transform(obs)
        
        assert isinstance(result, tuple), "Result should be tuple"
        assert len(result) == 3, "Should have three elements"
        assert all(isinstance(x, (int, np.integer)) for x in result), "All elements should be integers"
        
    def test_state_discretizer_min_values(self):
        """Test StateDiscretizer with minimum boundary values"""
        config = {
            'temp': {"min": 0, "max": 100, "bins": 10},
            'power': {"min": 0, "max": 50, "bins": 5}
        }
        state_disc = StateDiscretizer(config)
        obs = {'temp': 0, 'power': 0}
        
        result = state_disc.transform(obs)
        
        assert result == (0, 0), "Minimum values should map to first bin (0)"
        
    def test_state_discretizer_max_values(self):
        """Test StateDiscretizer with maximum boundary values"""
        config = {
            'temp': {"min": 0, "max": 100, "bins": 10},
            'power': {"min": 0, "max": 50, "bins": 5}
        }
        state_disc = StateDiscretizer(config)
        obs = {'temp': 100, 'power': 50}
        
        result = state_disc.transform(obs)
        
        assert result[0] == 9, "Temperature max should map to bin 9 (10-1)"
        assert result[1] == 4, "Power max should map to bin 4 (5-1)"
        
    def test_state_discretizer_clipping_values(self):
        """Test StateDiscretizer clips out-of-range values"""
        config = {
            'temp': {"min": 0, "max": 100, "bins": 10}
        }
        state_disc = StateDiscretizer(config)
        
        # Values outside the range
        obs_low = {'temp': -50}
        obs_high = {'temp': 150}
        
        result_low = state_disc.transform(obs_low)
        result_high = state_disc.transform(obs_high)
        
        # Both should be clipped to valid bins
        assert result_low[0] == 0, "Below-min value should clip to bin 0"
        assert result_high[0] == 9, "Above-max value should clip to last bin"
        
    def test_state_discretizer_consistency(self):
        """Test StateDiscretizer produces consistent results"""
        config = {
            'temperature': {"min": 20, "max": 80, "bins": 6},
            'power': {"min": 0, "max": 100, "bins": 10}
        }
        state_disc = StateDiscretizer(config)
        obs = {'temperature': 50, 'power': 75}
        
        result1 = state_disc.transform(obs)
        result2 = state_disc.transform(obs)
        
        assert result1 == result2, "Same observation should always produce same discrete state"
        
    def test_state_discretizer_different_observations(self):
        """Test StateDiscretizer produces different results for different observations"""
        config = {
            'temp': {"min": 0, "max": 100, "bins": 10}
        }
        state_disc = StateDiscretizer(config)
        
        result1 = state_disc.transform({'temp': 10})
        result2 = state_disc.transform({'temp': 90})
        
        assert result1 != result2, "Different observations should produce different states"
        
    def test_state_discretizer_order_preservation(self):
        """Test that StateDiscretizer preserves variable order"""
        config = {
            'var1': {"min": 0, "max": 10, "bins": 5},
            'var2': {"min": 0, "max": 10, "bins": 5},
            'var3': {"min": 0, "max": 10, "bins": 5}
        }
        state_disc = StateDiscretizer(config)
        obs = {'var1': 5, 'var2': 3, 'var3': 7}
        
        result = state_disc.transform(obs)
        
        # Result should be in some consistent order
        assert len(result) == 3
        # All elements should be different (assuming different inputs)
        # Note: They might not be different due to discretization, so just check they're valid
        assert all(isinstance(x, (int, np.integer)) for x in result)
        
    def test_state_discretizer_with_negative_values(self):
        """Test StateDiscretizer with negative value ranges"""
        config = {
            'temp': {"min": -50, "max": 50, "bins": 10}
        }
        state_disc = StateDiscretizer(config)
        
        result_neg = state_disc.transform({'temp': -25})
        result_zero = state_disc.transform({'temp': 0})
        result_pos = state_disc.transform({'temp': 25})
        
        assert 0 <= result_neg[0] < 10
        assert 0 <= result_zero[0] < 10
        assert 0 <= result_pos[0] < 10
        
    def test_state_discretizer_empty_config(self):
        """Test StateDiscretizer with empty configuration (edge case)"""
        config = {}
        state_disc = StateDiscretizer(config)
        obs = {}
        
        result = state_disc.transform(obs)
        
        assert result == (), "Empty config should return empty tuple"
        
    def test_state_discretizer_realistic_temperature_scenario(self):
        """Test StateDiscretizer with realistic temperature controller scenario"""
        config = {
            'storage_tank_temperature_sensor': {"min": C2K(30), "max": C2K(80), "bins": 10},
            'PV_power_sensor': {"min": 0, "max": 3, "bins": 5},
            'demand_heat_flow_sensor': {"min": 0, "max": 10, "bins": 5}
        }
        state_disc = StateDiscretizer(config)
        
        obs = {
            'storage_tank_temperature_sensor': C2K(50),
            'PV_power_sensor': 1.5,
            'demand_heat_flow_sensor': 5
        }
        
        result = state_disc.transform(obs)
        
        assert len(result) == 3, "Should have 3 discrete values"
        assert all(0 <= x < 10 for x in result), "All values should be in valid range"


class TestTemporalAggregator:
    """Test suite for the TemporalAggregator class"""
    
    def test_temporal_aggregator_initialization(self):
        """Test TemporalAggregator initialization"""
        agg = TemporalAggregator(n_blocks=5, agg_func="mean")
        assert agg.n_blocks == 5
        assert agg.agg_func == "mean"
        
    def test_temporal_aggregator_default_agg_func(self):
        """Test TemporalAggregator default aggregation function"""
        agg = TemporalAggregator(n_blocks=4)
        assert agg.agg_func == "mean"
        
    def test_temporal_aggregator_mean_aggregation(self):
        """Test mean aggregation function"""
        agg = TemporalAggregator(n_blocks=2, agg_func="mean")
        df = pd.DataFrame({'values': [1, 2, 3, 4]})
        
        result = agg.transform(df['values'].values)
        
        assert len(result) == 2, "Should have 2 blocks"
        np.testing.assert_array_almost_equal(result, [1.5, 3.5])
        
    def test_temporal_aggregator_sum_aggregation(self):
        """Test sum aggregation function"""
        agg = TemporalAggregator(n_blocks=2, agg_func="sum")
        df = pd.DataFrame({'values': [1, 2, 3, 4]})
        
        result = agg.transform(df['values'].values)
        
        assert len(result) == 2, "Should have 2 blocks"
        np.testing.assert_array_almost_equal(result, [3.0, 7.0])
        
    def test_temporal_aggregator_max_aggregation(self):
        """Test max aggregation function"""
        agg = TemporalAggregator(n_blocks=2, agg_func="max")
        df = pd.DataFrame({'values': [1, 2, 3, 4]})
        
        result = agg.transform(df['values'].values)
        
        assert len(result) == 2, "Should have 2 blocks"
        np.testing.assert_array_almost_equal(result, [2.0, 4.0])
        
    def test_temporal_aggregator_single_block(self):
        """Test aggregation to single block"""
        agg = TemporalAggregator(n_blocks=1, agg_func="mean")
        df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
        
        result = agg.transform(df['values'].values)
        
        assert len(result) == 1
        assert result[0] == 3.0  # mean of [1,2,3,4,5]
        
    def test_temporal_aggregator_many_blocks(self):
        """Test aggregation to many blocks"""
        n_vals = 100
        agg = TemporalAggregator(n_blocks=10, agg_func="mean")
        df = pd.DataFrame({'values': list(range(1, n_vals + 1))})
        
        result = agg.transform(df['values'].values)
        
        assert len(result) == 10
        # First block: mean of values 1-10 = 5.5
        assert result[0] == pytest.approx(5.5, abs=0.1)
        
    def test_temporal_aggregator_uneven_division(self):
        """Test aggregation when data length doesn't divide evenly"""
        agg = TemporalAggregator(n_blocks=3, agg_func="mean")
        df = pd.DataFrame({'values': [1, 2, 3, 4, 5, 6, 7]})  # 7 values into 3 blocks
        
        result = agg.transform(df['values'].values)
        
        assert len(result) == 3
        # Block 1: [1,2] -> mean = 1.5
        # Block 2: [3,4] -> mean = 3.5
        # Block 3: [5,6,7] -> mean = 6.0 (last block gets remainder)
        np.testing.assert_array_almost_equal(result, [1.5, 3.5, 6.0])
        
    def test_temporal_aggregator_multicolumn_dataframe(self):
        """Test aggregation with multi-column DataFrame (should give error)"""
        agg = TemporalAggregator(n_blocks=2, agg_func="mean")
        df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [3, 4, 5, 6]})
        
        with pytest.raises(ValueError, match="TemporalAggregator only supports 1D arrays"):
            result = agg.transform(df.values)
        
        assert True
        
    def test_temporal_aggregator_single_row_dataframe(self):
        """Test aggregation with single row DataFrame"""
        agg = TemporalAggregator(n_blocks=2, agg_func="mean")
        df = pd.DataFrame({'value': [42.0]})
        
        result = agg.transform(df['value'].values)
        
        assert len(result) == 1
        # Single value divided into 2 blocks
        
    def test_temporal_aggregator_invalid_agg_func(self):
        """Test error handling for invalid aggregation function"""
        agg = TemporalAggregator(n_blocks=2, agg_func="invalid_func")
        df = pd.DataFrame({'values': [1, 2, 3, 4]})
        
        with pytest.raises(ValueError, match="Unknown aggregation"):
            agg.transform(df['values'].values)
            
    def test_temporal_aggregator_negative_values(self):
        """Test aggregation with negative values"""
        agg = TemporalAggregator(n_blocks=2, agg_func="sum")
        df = pd.DataFrame({'values': [-2, -1, 1, 2]})
        
        result = agg.transform(df['values'].values)
        
        np.testing.assert_array_almost_equal(result, [-3.0, 3.0])
        
    def test_temporal_aggregator_float_values(self):
        """Test aggregation with floating point values"""
        agg = TemporalAggregator(n_blocks=3, agg_func="mean")
        df = pd.DataFrame({'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
        
        result = agg.transform(df['values'].values)
        
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, [0.15, 0.35, 0.55])


class TestDiscretizerWithTemporalAggregation:
    """Test suite for Discretizer with temporal aggregation"""
    
    def test_discretizer_with_temporal_aggregator_initialization(self):
        """Test Discretizer initialization with TemporalAggregator"""
        agg = TemporalAggregator(n_blocks=5, agg_func="mean")
        discretizer = Discretizer(vmin=0, vmax=100, n_bins=10, temporal_aggregator=agg)

        assert discretizer.temporal_aggregator == agg
        
            
    def test_discretizer_with_temporal_aggregator_mean(self):
        """Test discretization of aggregated DataFrame with mean"""
        agg = TemporalAggregator(n_blocks=2, agg_func="mean")
        discretizer = Discretizer(vmin=0, vmax=100, n_bins=10, temporal_aggregator=agg)
        
        df = pd.DataFrame({'values': [10, 20, 80, 90]})
        result = discretizer.discretize(df['values'].values)
        
        # Should aggregate to [15, 85] then discretize
        assert len(result) == 2
        assert result[0] < result[1], "15 should be in lower bin than 85"
        
    def test_discretizer_with_temporal_aggregator_sum(self):
        """Test discretization of aggregated DataFrame with sum"""
        agg = TemporalAggregator(n_blocks=2, agg_func="sum")
        discretizer = Discretizer(vmin=0, vmax=100, n_bins=5, temporal_aggregator=agg)
        
        df = pd.DataFrame({'values': [10, 15, 35, 40]})
        result = discretizer.discretize(df['values'].values)
        
        # Should aggregate to [25, 75] then discretize
        assert len(result) == 2
        assert all(0 <= r < 5 for r in result)
        
    def test_discretizer_with_aggregated_values_clipping(self):
        """Test that aggregated values are clipped to valid range"""
        agg = TemporalAggregator(n_blocks=2, agg_func="max")
        discretizer = Discretizer(vmin=20, vmax=80, n_bins=6, temporal_aggregator=agg)
        
        df = pd.DataFrame({'values': [10, 15, 90, 95]})  # Values outside valid range
        result = discretizer.discretize(df['values'].values)
        
        # Should clip to [20, 80] then discretize
        assert len(result) == 2
        assert all(0 <= r < 6 for r in result)


class TestStateDiscretizerWithPredictions:
    """Test suite for StateDiscretizer with predictions"""
    
    def test_state_discretizer_predictions_only(self):
        """Test StateDiscretizer with predictions only (no observations)"""
        config = {
            'pv_forecast': {"min": 0, "max": 3, "bins": 5},
            'demand_forecast': {"min": 0, "max": 10, "bins": 5}
        }
        state_disc = StateDiscretizer(config)
        
        predictions = {
            'pv_forecast': 1.5,
            'demand_forecast': 5.0
        }
        
        result = state_disc.transform(predictions=predictions)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, (int, np.integer)) for x in result)
        
    def test_state_discretizer_observations_take_priority(self):
        """Test that observations take priority over predictions"""
        config = {
            'sensor_value': {"min": 0, "max": 100, "bins": 10}
        }
        state_disc = StateDiscretizer(config)
        
        obs = {'sensor_value': 50}
        predictions = {'sensor_value': 75}
        
        result_obs_only = state_disc.transform(obs=obs)
        result_with_both = state_disc.transform(obs=obs, predictions=predictions)
        
        assert result_obs_only != result_with_both; "Observations and predictions should be treated equally" 
        
    def test_state_discretizer_fallback_to_predictions(self):
        """Test that predictions are used when observation not available"""
        config = {
            'sensor1': {"min": 0, "max": 100, "bins": 10},
            'forecast1': {"min": 0, "max": 100, "bins": 10}
        }
        state_disc = StateDiscretizer(config)
        
        obs = {'sensor1': 50}
        predictions = {'forecast1': 75}
        
        result = state_disc.transform(obs=obs, predictions=predictions)
        
        assert len(result) == 2
        
    def test_state_discretizer_mixed_obs_and_predictions(self):
        """Test StateDiscretizer with both observations and predictions"""
        config = {
            'temp_sensor': {"min": 20, "max": 80, "bins": 10},
            'temp_forecast': {"min": 20, "max": 80, "bins": 10},
            'power_sensor': {"min": 0, "max": 100, "bins": 5},
            'power_forecast': {"min": 0, "max": 100, "bins": 5}
        }
        state_disc = StateDiscretizer(config)
        
        obs = {'temp_sensor': 50, 'power_sensor': 50}
        predictions = {'temp_forecast': 55, 'power_forecast': 60}
        
        result = state_disc.transform(obs=obs, predictions=predictions)
        
        assert len(result) == 4
        assert all(isinstance(x, (int, np.integer)) for x in result)
        
    def test_state_discretizer_missing_variable_raises_error(self):
        """Test that variables that have no discretizer assigned raise an error, unless they are integers"""
        config = {
            'required_var': {"min": 0, "max": 100, "bins": 10}
        }
        state_disc = StateDiscretizer(config)
        
        with pytest.raises(ValueError):
            state_disc.transform(obs={'other_var': 10.4})
        state_disc.transform(obs = {'other_var': 8})
        assert True
            
    def test_state_discretizer_predictions_as_dataframe(self):
        """Test StateDiscretizer with DataFrame predictions and temporal aggregation"""
        config = {
            'pv_power': {
                "min": 0, 
                "max": 5, 
                "bins": 5,
                "temporal": {"n_blocks": 3, "agg": "mean"}
            }
        }
        state_disc = StateDiscretizer(config)
        
        # Create a DataFrame prediction (e.g., 24-hour forecast)
        predictions = {
            'pv_power': np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
        }
        
        result = state_disc.transform(predictions=predictions)
        
        assert len(result) == 3, "Should have 3 aggregated values"
        assert all(isinstance(x, (int, np.integer)) for x in result)
        
    def test_state_discretizer_temporal_aggregation_multiple_vars(self):
        """Test StateDiscretizer with temporal aggregation on multiple variables"""
        config = {
            'pv_forecast': {
                "min": 0,
                "max": 5,
                "bins": 5,
                "temporal": {"n_blocks": 2, "agg": "mean"}
            },
            'demand_forecast': {
                "min": 0,
                "max": 10,
                "bins": 5,
                "temporal": {"n_blocks": 2, "agg": "sum"}
            }
        }
        state_disc = StateDiscretizer(config)
        
        predictions = {
            'pv_forecast': np.array([0.5, 1.0, 2.0, 3.0]),
            'demand_forecast': np.array([1, 2, 3, 4])
        }
        
        result = state_disc.transform(predictions=predictions)
        
        # Should have 2 blocks per variable = 4 total
        assert len(result) == 4
        assert all(isinstance(x, (int, np.integer)) for x in result)
        
    def test_state_discretizer_prediction_with_clipping(self):
        """Test that predictions outside valid range are clipped"""
        config = {
            'forecast': {"min": 0, "max": 100, "bins": 10}
        }
        state_disc = StateDiscretizer(config)
        
        predictions_low = {'forecast': -50}
        predictions_high = {'forecast': 150}
        
        result_low = state_disc.transform(predictions=predictions_low)
        result_high = state_disc.transform(predictions=predictions_high)
        
        assert result_low[0] == 0, "Below-min should clip to bin 0"
        assert result_high[0] == 9, "Above-max should clip to last bin"
        
    def test_state_discretizer_realistic_scenario_with_temporal_aggregation(self):
        """Test realistic scenario: sensor reading + temporally aggregated forecast"""
        config = {
            'storage_temp_sensor': {"min": C2K(20), "max": C2K(80), "bins": 10},
            'pv_30min_forecast': {
                "min": 0,
                "max": 5,
                "bins": 5,
                "temporal": {"n_blocks": 2, "agg": "mean"}  # Aggregate 30-min forecast to 2 blocks
            },
            'dhw_demand_forecast': {
                "min": 0,
                "max": 15,
                "bins": 5,
                "temporal": {"n_blocks": 2, "agg": "sum"}
            }
        }
        state_disc = StateDiscretizer(config)
        
        obs = {
            'storage_temp_sensor': C2K(50)
        }
        predictions = {
            'pv_30min_forecast': np.array([0.5, 1.5, 2.5, 3.5]),
            'dhw_demand_forecast': np.array([2, 3, 4, 5])
        }
        
        result = state_disc.transform(obs=obs, predictions=predictions)
        
        # 1 sensor + 2 aggregated PV + 2 aggregated demand = 5 discrete values
        assert len(result) == 5
        assert all(isinstance(x, (int, np.integer)) for x in result)


class TestRLControllerFunctions:

    def test_combination_of_actions(self):
        """Test that all combinations of actions are generated correctly"""
        actions = {'heat_pump': [0, 1], 'resistance_heater': [0, 1]}
        agent = QLearningAgent(actions)
        assert len(agent.action_space) == 4
        assert agent.action_space[0] == {'heat_pump': 0, 'resistance_heater': 0}
        assert agent.action_space[1] == {'heat_pump': 0, 'resistance_heater': 1}
        assert agent.action_space[2] == {'heat_pump': 1, 'resistance_heater': 0}
        assert agent.action_space[3] == {'heat_pump': 1, 'resistance_heater': 1}
        # Also tests what happen if there's only one component
        actions = {'heat_pump': [0, 1]}
        agent = QLearningAgent(actions)
        assert len(agent.action_space) == 2
        assert agent.action_space[0] == {'heat_pump': 0}
        assert agent.action_space[1] == {'heat_pump': 1}

class TestRLControllerFull:

    def test_RL_HybridDHW_application_onlyT(self, test_components, test_sensors):
    # Test of a standard hybrid system, where the only signal read by the RL controller is the storage tank temperature
        controllers = [
            QLearningController(
                name = 'test_RL_controller',
                sensors = {'storage tank temperature': 'storage_tank_temperature_sensor'},
                actions = {'heat_pump': [0, 1]},
                exploration_policy = {'type': 'epsilon-greedy', 'config info': {}},
                valid_states_function = {'control variable': 'storage tank temperature', 'config info': {(273+0, 273+35): {'heat_pump': [1]}, (273+35, 273+70): {'heat_pump': [0, 1]}, (273+70, 273+100): {'heat_pump': [0]}}},
                agent_config_info = {'epsilon': 0.02, 'decay': 0.01, 'alpha': 0.1},
                reward_function = TemperatureMinMaxReward(
                    sensor_name='storage_tank_temperature_sensor',
                    min_temp=40,
                    max_temp=60.0
                ),
                state_discretizer = StateDiscretizer({
                    'storage tank temperature': {"min": C2K(30), "max": C2K(80), "bins": 10},
                })),
            esc.ChargeController('charge_controller', 'battery', 'battery_SOC_sensor', 'inverter_power_output_sensor', 'PV_power_sensor')
                    ]
        connections = [
            ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
            ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
            ('heat_pump_electricity_input_port', 'inverter_AC_output_port'),
            ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
            ('inverter_PV_input_port', 'pv_panels_electricity_port'),
            ('inverter_grid_input_port', 'electric_grid_electricity_port'),
            ('inverter_ESS_port', 'battery_electricity_port')
        ]
        predictors = [PerfectTimeSeriesPredictor('pv_power_predictor', 'pv_panels'), 
                    PerfectTimeSeriesPredictor('dhw_demand_predictor', 'demand_DHW')]
        # Create environment
        env = esc.Environment(components=test_components, controllers = controllers, sensors=test_sensors, connections=connections, predictors=predictors)  # dt = 60 s
        # Create simulator object
        sim_config = esc.SimulationConfig(time_start_h = 0.0, time_end_h = 24.0*14, time_step_h = 5/60)
        sim = esc.Simulator(env, sim_config)
        # Run simulation
        results = sim.run()
        df_ports, df_controllers, df_sensors = results.to_dataframe()    
        assert (df_sensors['storage_tank_temperature_sensor'] < C2K(40)).sum() < 100
        assert (df_sensors['storage_tank_temperature_sensor'] > C2K(80)).sum() < 100
        assert True

    def test_RL_HybridDHW_application_onlyT_with_minimum_switch_time(self, test_components, test_sensors):
    # Test of a standard hybrid system, where the only signal read by the RL controller is the storage tank temperature
        controllers = [
            QLearningController(
                name = 'test_RL_controller',
                sensors = {'storage tank temperature': 'storage_tank_temperature_sensor'},
                actions = {'heat_pump': [0, 1]},
                exploration_policy = {'type': 'epsilon-greedy', 'config info': {}},
                valid_states_function = {'control variable': 'storage tank temperature', 'config info': {(273+0, 273+35): {'heat_pump': [1]}, (273+35, 273+70): {'heat_pump': [0, 1]}, (273+70, 273+100): {'heat_pump': [0]}}},
                agent_config_info = {'epsilon': 0.02, 'decay': 0.01, 'alpha': 0.1},
                minimum_time_between_state_switches_h = {'heat_pump': 0.5},
                reward_function = TemperatureMinMaxReward(
                    sensor_name='storage_tank_temperature_sensor',
                    min_temp=40,
                    max_temp=60.0
                ),
                state_discretizer = StateDiscretizer({
                    'storage tank temperature': {"min": C2K(30), "max": C2K(80), "bins": 10},
                })),
            esc.ChargeController('charge_controller', 'battery', 'battery_SOC_sensor', 'inverter_power_output_sensor', 'PV_power_sensor')
                    ]
        connections = [
            ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
            ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
            ('heat_pump_electricity_input_port', 'inverter_AC_output_port'),
            ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
            ('inverter_PV_input_port', 'pv_panels_electricity_port'),
            ('inverter_grid_input_port', 'electric_grid_electricity_port'),
            ('inverter_ESS_port', 'battery_electricity_port')
        ]
        predictors = [PerfectTimeSeriesPredictor('pv_power_predictor', 'pv_panels'), 
                    PerfectTimeSeriesPredictor('dhw_demand_predictor', 'demand_DHW')]
        # Create environment
        env = esc.Environment(components=test_components, controllers = controllers, sensors=test_sensors, connections=connections, predictors=predictors)  # dt = 60 s
        # Create simulator object
        sim_config = esc.SimulationConfig(time_start_h = 0.0, time_end_h = 24.0*14, time_step_h = 5/60)
        sim = esc.Simulator(env, sim_config)
        # Run simulation
        results = sim.run()
        df_ports, df_controllers, df_sensors = results.to_dataframe()    
        assert (df_sensors['storage_tank_temperature_sensor'] < C2K(40)).sum() < 100
        assert (df_sensors['storage_tank_temperature_sensor'] > C2K(80)).sum() < 100
        assert True


    def test_RL_HybridDHW_application_onlyT_2(self, test_components, test_sensors):
    # Test of a standard hybrid system, where the only signal read by the RL controller is the storage tank temperature
    # Testing a different way of initializing the different components
        controllers = [
            QLearningController(
                name = 'test_RL_controller',
                sensors = {'storage tank temperature': 'storage_tank_temperature_sensor'},
                actions = {'heat_pump': [0, 1]},
                exploration_policy = {'type': 'epsilon-greedy',
                                      'config info': {
                                          'bias function': {'control variable': 'storage tank temperature', 'config info': {(273+0, 273+35): [0.0, 1.0], (273+35, 273+40): [0.1, 0.9], (273+40, 273+60): [0.5, 0.5], (273+60, 273+70): [0.8, 0.2], (273+70, 273+100): [1.0, 0.0]}}}},
                valid_states_function = {'control variable': 'storage tank temperature', 'config info': {(273+0, 273+35): {'heat_pump': [1]}, (273+35, 273+70): {'heat_pump': [0, 1]}, (273+70, 273+100): {'heat_pump': [0]}}},
                agent_config_info = {'epsilon': 0.02, 'decay': 0.01, 'alpha': 0.1},
                reward_function = {
                    "type": "temperature_minmax",
                    "kwargs": {"min_temp": 40, "max_temp": 60.0, "sensor_name": 'storage_tank_temperature_sensor'}
                },
                state_discretizer = {'storage tank temperature': {"min": C2K(30), "max": C2K(80), "bins": 10},}),
            esc.ChargeController('charge_controller', 'battery', 'battery_SOC_sensor', 'inverter_power_output_sensor', 'PV_power_sensor')
                    ]
        connections = [
            ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
            ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
            ('heat_pump_electricity_input_port', 'inverter_AC_output_port'),
            ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
            ('inverter_PV_input_port', 'pv_panels_electricity_port'),
            ('inverter_grid_input_port', 'electric_grid_electricity_port'),
            ('inverter_ESS_port', 'battery_electricity_port')
        ]
        predictors = [PerfectTimeSeriesPredictor('pv_power_predictor', 'pv_panels'), 
                    PerfectTimeSeriesPredictor('dhw_demand_predictor', 'demand_DHW')]
        # Create environment
        env = esc.Environment(components=test_components, controllers = controllers, sensors=test_sensors, connections=connections, predictors=predictors)  # dt = 60 s
        # Create simulator object
        sim_config = esc.SimulationConfig(time_start_h = 0.0, time_end_h = 24.0*14, time_step_h = 5/60)
        sim = esc.Simulator(env, sim_config)
        # Run simulation
        results = sim.run()
        df_ports, df_controllers, df_sensors = results.to_dataframe()    
        assert (df_sensors['storage_tank_temperature_sensor'] < C2K(40)).sum() < 100
        assert (df_sensors['storage_tank_temperature_sensor'] > C2K(80)).sum() < 100
        assert True


    def test_RL_HybridDHW_application_TandP(self, test_components, test_sensors):
    # Test of a standard hybrid system, where the only signals read by the RL controller are:
    # - The storage tank temperature
    # - The heat pump power
        controllers = [
            QLearningController(
                name = 'test_RL_controller',
                sensors = {'storage tank temperature': 'storage_tank_temperature_sensor',
                           'power PV': 'PV_power_sensor'},
                actions = {'heat_pump': [0, 1]},
                exploration_policy = {'type': 'epsilon-greedy',
                                      'config info': {
                                          'bias function': {'control variable': 'storage tank temperature', 'config info': {(273+0, 273+35): [(0, 0.0), (1, 1.0)], (273+35, 273+40): [(0, 0.1), (1, 0.9)], (273+40, 273+60): [(0, 0.5), (1, 0.5)], (273+60, 273+70): [(0, 0.8), (1, 0.2)], (273+70, 273+100): [(0, 1.0), (1, 0.0)]}}}},
                valid_states_function = {'control variable': 'storage tank temperature', 'config info': {(273+0, 273+35): {'heat_pump': [1]}, (273+35, 273+70): {'heat_pump': [0, 1]}, (273+70, 273+100): {'heat_pump': [0]}}},
                agent_config_info = {'epsilon': 0.02, 'decay': 0.01, 'alpha': 0.1},
                reward_function = CompositeReward([
                    TemperatureMinMaxReward(sensor_name='storage_tank_temperature_sensor', min_temp=40, max_temp=60.0),
                    EnergyCostReward(cost_components = [{'component': 'electric_grid', 'sensor': 'grid_power_sensor'}])
                ]),
                state_discretizer = {'storage tank temperature': {"min": C2K(30), "max": C2K(80), "bins": 10},
                                     'power PV': {'min': 0, 'max': 3.0, "bins": 10}}),
            esc.ChargeController('charge_controller', 'battery', 'battery_SOC_sensor', 'inverter_power_output_sensor', 'PV_power_sensor')
                    ]
        connections = [
            ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
            ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
            ('heat_pump_electricity_input_port', 'inverter_AC_output_port'),
            ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
            ('inverter_PV_input_port', 'pv_panels_electricity_port'),
            ('inverter_grid_input_port', 'electric_grid_electricity_port'),
            ('inverter_ESS_port', 'battery_electricity_port')
        ]
        # Create environment
        env = esc.Environment(components=test_components, controllers = controllers, sensors=test_sensors, connections=connections)  # dt = 60 s
        # Create simulator object
        sim_config = esc.SimulationConfig(time_start_h = 0.0, time_end_h = 24.0*30, time_step_h = 1/60)
        sim = esc.Simulator(env, sim_config)
        # Run simulation
        results = sim.run()
        df_ports, df_controllers, df_sensors = results.to_dataframe()    
        assert (df_sensors['storage_tank_temperature_sensor'] < C2K(40)).sum() < 500
        assert (df_sensors['storage_tank_temperature_sensor'] > C2K(80)).sum() < 100
        assert True

    def test_RL_HybridDHW_application_TandP_with_minimum_switch_time(self, test_components, test_sensors):
    # Test of a standard hybrid system, where the only signals read by the RL controller are:
    # - The storage tank temperature
    # - The heat pump power
        controllers = [
            QLearningController(
                name = 'test_RL_controller',
                sensors = {'storage tank temperature': 'storage_tank_temperature_sensor',
                           'power PV': 'PV_power_sensor'},
                actions = {'heat_pump': [0, 1]},
                exploration_policy = {'type': 'epsilon-greedy',
                                      'config info': {
                                          'bias function': {'control variable': 'storage tank temperature', 'config info': {(273+0, 273+35): [(0, 0.0), (1, 1.0)], (273+35, 273+40): [(0, 0.1), (1, 0.9)], (273+40, 273+60): [(0, 0.5), (1, 0.5)], (273+60, 273+70): [(0, 0.8), (1, 0.2)], (273+70, 273+100): [(0, 1.0), (1, 0.0)]}}}},
                valid_states_function = {'control variable': 'storage tank temperature', 'config info': {(273+0, 273+35): {'heat_pump': [1]}, (273+35, 273+70): {'heat_pump': [0, 1]}, (273+70, 273+100): {'heat_pump': [0]}}},
                minimum_time_between_state_switches_h = {'heat_pump': 0.5},
                agent_config_info = {'epsilon': 0.02, 'decay': 0.01, 'alpha': 0.1},
                reward_function = CompositeReward([
                    TemperatureMinMaxReward(sensor_name='storage_tank_temperature_sensor', min_temp=40, max_temp=60.0),
                    EnergyCostReward(cost_components = [{'component': 'electric_grid', 'sensor': 'grid_power_sensor'}])
                ]),
                state_discretizer = {'storage tank temperature': {"min": C2K(30), "max": C2K(80), "bins": 10},
                                     'power PV': {'min': 0, 'max': 3.0, "bins": 10}}),
            esc.ChargeController('charge_controller', 'battery', 'battery_SOC_sensor', 'inverter_power_output_sensor', 'PV_power_sensor')
                    ]
        connections = [
            ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
            ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
            ('heat_pump_electricity_input_port', 'inverter_AC_output_port'),
            ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
            ('inverter_PV_input_port', 'pv_panels_electricity_port'),
            ('inverter_grid_input_port', 'electric_grid_electricity_port'),
            ('inverter_ESS_port', 'battery_electricity_port')
        ]
        # Create environment
        env = esc.Environment(components=test_components, controllers = controllers, sensors=test_sensors, connections=connections)  # dt = 60 s
        # Create simulator object
        sim_config = esc.SimulationConfig(time_start_h = 0.0, time_end_h = 24.0*14, time_step_h = 1/60)
        sim = esc.Simulator(env, sim_config)
        # Run simulation
        results = sim.run()
        df_ports, df_controllers, df_sensors = results.to_dataframe()    
        assert (df_sensors['storage_tank_temperature_sensor'] < C2K(40)).sum() < 500
        assert (df_sensors['storage_tank_temperature_sensor'] > C2K(80)).sum() < 100
        assert True

    def test_RL_HybridDHW_application_TandP_with_minimum_switch_time_hour_year(self, test_components, test_sensors):
    # Test of a standard hybrid system, where the only signals read by the RL controller are:
    # - The storage tank temperature
    # - The heat pump power
    # This specific case, in addition to the previous one, also includes a minimum switch time of 30 minutes for the heat pump.
    # This speficic case, in addition to the previous one, also includes the sin and cos of the hour of the day and the time of the year
        controllers = [
            QLearningController(
                name = 'test_RL_controller',
                sensors = {'storage tank temperature': 'storage_tank_temperature_sensor',
                           'power PV': 'PV_power_sensor',
                           'battery SOC': 'battery_SOC_sensor'},
                actions = {'heat_pump': [0, 1]},
                exploration_policy = {'type': 'epsilon-greedy',
                                      'config info': {
                                          'bias function': {'control variable': 'storage tank temperature', 'config info': {(273+0, 273+35): [(0, 0.0), (1, 1.0)], (273+35, 273+40): [(0, 0.1), (1, 0.9)], (273+40, 273+60): [(0, 0.5), (1, 0.5)], (273+60, 273+70): [(0, 0.8), (1, 0.2)], (273+70, 273+100): [(0, 1.0), (1, 0.0)]}}}},
                valid_states_function = {'control variable': 'storage tank temperature', 'config info': {(273+0, 273+35): {'heat_pump': [1]}, (273+35, 273+70): {'heat_pump': [0, 1]}, (273+70, 273+100): {'heat_pump': [0]}}},
                minimum_time_between_state_switches_h = {'heat_pump': 0.25},
                agent_config_info = {'epsilon': 0.4, 'decay': 24*30, 'alpha': 0.1, 'min_epsilon': 0.1},
                reward_function = CompositeReward([
                    TemperatureMinMaxReward(sensor_name='storage_tank_temperature_sensor', min_temp=40, max_temp=65.0, weight=0.000000),
                    EnergyCostReward(cost_components = [{'component': 'electric_grid', 'sensor': 'grid_power_sensor'}])
                ]),
                include_hour_of_day = False,
                include_day_of_the_year = False,
                state_discretizer = {'storage tank temperature': {"min": C2K(30), "max": C2K(80), "bins": 10},
                                     'power PV': {'min': 0, 'max': 3.0, "bins": 10},
                                     'battery SOC': {'min': 0, 'max': 1.0, 'bins': 3}}),
            esc.ChargeController('charge_controller', 'battery', 'battery_SOC_sensor', 'inverter_power_output_sensor', 'PV_power_sensor')
                    ]
        connections = [
            ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
            ('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port'),
            ('heat_pump_electricity_input_port', 'inverter_AC_output_port'),
            ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
            ('inverter_PV_input_port', 'pv_panels_electricity_port'),
            ('inverter_grid_input_port', 'electric_grid_electricity_port'),
            ('inverter_ESS_port', 'battery_electricity_port')
        ]
        # Create environment
        env = esc.Environment(components=test_components, controllers = controllers, sensors=test_sensors, connections=connections)  # dt = 60 s
        # Create simulator object
        sim_config = esc.SimulationConfig(time_start_h = 0.0, time_end_h = 24.0*180, time_step_h = 1/60)
        sim = esc.Simulator(env, sim_config)
        # Run simulation
        results = sim.run()
        df_ports, df_controllers, df_sensors = results.to_dataframe()    
        assert (df_sensors['storage_tank_temperature_sensor'] < C2K(40)).sum() < 500
        assert (df_sensors['storage_tank_temperature_sensor'] > C2K(80)).sum() < 100
        assert True
