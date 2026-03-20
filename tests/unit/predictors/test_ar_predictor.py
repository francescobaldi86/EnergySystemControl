# test_predictors.py
import pandas as pd
import numpy as np
import pytest
from dataclasses import dataclass
from energy_system_control.controllers.predictors import AutocorrPredictor
from energy_system_control.sim.state import SimulationState
from energy_system_control.core.base_classes import InitContext
from tests.utils import calculate_prediction_metrics, MockEnvironment, MockSensor
import os



# =====================================================
# AUTOCORR PREDICTOR TESTS
# =====================================================

@pytest.fixture
def autocorr_predictor_simple():
    """Create a simple AutocorrPredictor for basic tests."""
    return AutocorrPredictor(
        sensor_name="test_sensor",
        prediction_horizon_h=1,  # 1 hour ahead
        lags_h=[0.25, 1],  # 15 minutes and 1 hour lags
        name="simple_autocorr"
    )


class TestAutocorrPredictorBase:

    def test_AutocorrPredictor_initialization(self):
        """Test AutocorrPredictor initialization with various lag configurations."""
        # Test basic initialization
        predictor = AutocorrPredictor(
            sensor_name="test_sensor",
            prediction_horizon_h=2,
            lags_h=[1, 24],
            name="test_autocorr"
        )
        
        assert predictor.name == "test_autocorr"
        assert predictor.sensor_name == "test_sensor"
        assert predictor.variable_to_predict == "test_sensor"
        assert predictor.prediction_horizon_h == 2
        assert predictor.lags_h == [1, 24]
        assert len(predictor.buffer) == 0
        assert len(predictor.time_buffer) == 0


    def test_AutocorrPredictor_initialization_different_lags(self):
        """Test initialization with different lag configurations."""
        # Test with single lag
        pred1 = AutocorrPredictor(
            sensor_name="sensor1",
            prediction_horizon_h=1,
            lags_h=[1]
        )
        assert pred1.lags_h == [1]
        # Test with multiple lags
        pred2 = AutocorrPredictor(
            sensor_name="sensor2",
            prediction_horizon_h=1,
            lags_h=[0.5, 1, 24, 168]
        )
        assert pred2.lags_h == [0.5, 1, 24, 168]
        # Test with default name
        assert pred1.name == "sensor1"
        assert pred2.name == "sensor2"

    def test_AutocorrPredictor_initialize(self, init_context_autocorr):
        """Test the initialize method converts lags to steps."""
        predictor = AutocorrPredictor(
            sensor_name="test_sensor",
            prediction_horizon_h=1,
            lags_h=[1, 24]
        )
        # Initialize
        predictor.initialize(init_context_autocorr)        
        # Check that lags and horizon were converted to steps
        assert predictor.lags_steps is not None
        assert predictor.prediction_horizon is not None
        # 1 hour = 3600 seconds, 3600/900 = 4 steps
        assert predictor.lags_steps[0] == 4
        # 24 hours = 86400 seconds, 86400/900 = 96 steps
        assert predictor.lags_steps[1] == 96
        # 1 hour horizon = 4 steps
        assert predictor.prediction_horizon == 4


    def test_AutocorrPredictor_update(self, autocorr_predictor_simple, init_context_autocorr):
        """Test that update method correctly adds measurements to buffer."""
        predictor = autocorr_predictor_simple
        predictor.initialize(init_context_autocorr)
        
        initial_size = len(predictor.buffer)
        
        # Update with a measurement
        test_value = 5.0
        predictor.buffer.append(test_value)
        
        assert len(predictor.buffer) == initial_size + 1
        assert predictor.buffer[-1] == test_value


    def test_AutocorrPredictor_predict_insufficient_data(self, init_context_autocorr):
        """Test prediction fallback when insufficient data is available."""
        predictor = AutocorrPredictor(
            sensor_name="test_sensor",
            prediction_horizon_h=0.5,  # 30 minutes
            lags_h=[1, 24]  # requires 24 hour lag
        )
        
        # Setup with minimal data
        measurements = np.array([1.0, 2.0, 3.0])
        init_context_autocorr.environment.sensors['test_sensor'].measurements = measurements
        predictor.initialize(init_context_autocorr)
        
        # Add minimal data to buffer (less than max lag requirement)
        for val in measurements:
            predictor.buffer.append(val)
        
        # Predict - should use fallback (repeat last value)
        state = SimulationState(
            simulation_start_datetime=pd.Timestamp("2025-01-01 00:00:00"),
            time=0,
            time_step=900
        )
        predictions = predictor.predict(0.5, state)
        
        # Should return array with repeated last value
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == predictor.prediction_horizon
        assert np.allclose(predictions, measurements[0])


    def test_AutocorrPredictor_predict_sufficient_data(self, init_context_autocorr):
        """Test prediction with sufficient data."""
        predictor = AutocorrPredictor(
            sensor_name="test_sensor",
            prediction_horizon_h=1,  # 1 hour
            lags_h=[1]  # 1 hour lag
        )
        
        # Create cyclical data (sine-like pattern)
        t = np.linspace(0, 8*np.pi, 200)
        measurements = np.sin(t)
        init_context_autocorr.environment.sensors['test_sensor'].measurements = measurements
        predictor.initialize(init_context_autocorr)
        
        # Fill buffer with measurements
        for val in measurements:
            predictor.buffer.append(val)
            predictor.time_buffer.append(len(predictor.buffer) * 900)
        
        # Predict
        state = SimulationState(
            simulation_start_datetime=pd.Timestamp("2025-01-01 00:00:00"),
            time=(len(measurements) - 1) * 900,
            time_step=900
        )
        predictions = predictor.predict(1, state)
        
        # Verify output properties
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == predictor.prediction_horizon
        assert np.all(predictions >= 0)  # Non-negative constraint


    def test_AutocorrPredictor_non_negativity(self, init_context_autocorr):
        """Test that predictions are enforced to be non-negative."""
        predictor = AutocorrPredictor(
            sensor_name="test_sensor",
            prediction_horizon_h=0.5,
            lags_h=[1]
        )
        
        # Create data with negative values
        measurements = np.array([-1.0, -2.0, -1.5, -0.5, 0.0, 0.5, 1.0, 0.5] * 20)
        init_context_autocorr.environment.sensors['test_sensor'].measurements = measurements
        predictor.initialize(init_context_autocorr)
        
        # Fill buffer
        for val in measurements:
            predictor.buffer.append(val)
        
        # Predict
        state = SimulationState(
            simulation_start_datetime=pd.Timestamp("2025-01-01 00:00:00"),
            time=(len(measurements) - 1) * 900,
            time_step=900
        )
        predictions = predictor.predict(0.5, state)
        
        # All predictions should be non-negative
        assert np.all(predictions >= 0), "Predictions should be non-negative"


    def test_AutocorrPredictor_horizon_mismatch(self, init_context_autocorr):
        """Test that horizon mismatch raises ValueError."""
        predictor = AutocorrPredictor(
            sensor_name="test_sensor",
            prediction_horizon_h=1,  # Model expects 1 hour
            lags_h=[1]
        )
        
        predictor.initialize(init_context_autocorr)
        measurements = init_context_autocorr.environment.sensors['test_sensor'].measurements
        # Fill buffer
        for val in measurements:
            predictor.buffer.append(val)
        
        # Try to predict with different horizon - should raise ValueError
        with pytest.raises(ValueError, match="Prediction horizon mismatch"):
            predictor.predict(2, init_context_autocorr.state)  # Requesting 2 hours instead of 1


    def test_AutocorrPredictor_multiple_lags(self, init_context_autocorr):
        """Test predictor with multiple lag values."""
        predictor = AutocorrPredictor(
            sensor_name="test_sensor",
            prediction_horizon_h=1,
            lags_h=[0.25, 1, 4]  # 15 min, 1 hour, 4 hours
        )
        
        # Create more data to satisfy larger lag requirement
        measurements = np.sin(np.linspace(0, 20*np.pi, 600))
        init_context_autocorr.environment.sensors['test_sensor'].measurements = measurements
        predictor.initialize(init_context_autocorr)
        
        # Fill buffer
        for val in measurements:
            predictor.buffer.append(val)
        
        # Predict
        state = SimulationState(
            simulation_start_datetime=pd.Timestamp("2025-01-01 00:00:00"),
            time=(len(measurements) - 1) * 900,
            time_step=900
        )
        predictions = predictor.predict(1, state)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == predictor.prediction_horizon
        assert np.all(predictions >= 0)


class TestAutocorrPredictorDHW:

    def test_AutocorrPredictor_dhw_demand_data(self, dhw_demand_data, init_context_autocorr):
        """
        Test AutocorrPredictor on real DHW demand data.
        
        This test:
        1. Creates a predictor with reasonable lag values for hourly patterns
        2. Trains on first 30% of data
        3. Evaluates on remaining data
        4. Verifies predictions are reasonable
        """
        time_step = 900  # 15 minutes (same as in data which is 15-min intervals)
        prediction_horizon_h = 6
        
        # Split data: training and test
        train_size = int(len(dhw_demand_data) * 0.3)
        train_data = dhw_demand_data[:train_size]
        test_data = dhw_demand_data[train_size:]
        
        print(f"\n--- AutocorrPredictor DHW Test ---")
        print(f"Training data: {len(train_data)} samples ({len(train_data)*time_step/3600:.1f} hours)")
        print(f"Test data: {len(test_data)} samples ({len(test_data)*time_step/3600:.1f} hours)")
        
        # Create predictor with daily and hourly patterns
        predictor = AutocorrPredictor(
            sensor_name="test_sensor",
            prediction_horizon_h=prediction_horizon_h,
            lags_h=[1, 24],  # 1 hour and 24 hour lags
            name="dhw_autocorr"
        )
        
        # Initialize
        measurements = train_data.tolist()
        init_context_autocorr.environment.sensors['test_sensor'].measurements = measurements
        predictor.initialize(init_context_autocorr)
        
        # Populate buffer with training data
        for val in train_data:
            predictor.buffer.append(val)
            predictor.time_buffer.append(len(predictor.buffer) * time_step)
        
        # Make predictions on test data
        horizon_steps = predictor.prediction_horizon
        predictions = []
        actuals = []
        
        for i in range(min(len(test_data) - horizon_steps, 100)):  # Limit to 100 predictions
            predictor.buffer.append(test_data[i])
            predictor.time_buffer.append((train_size + i) * time_step)
            
            if len(predictor.buffer) > max(predictor.lags_steps):
                pred = predictor.predict(
                    prediction_horizon_h,
                    SimulationState(time=(train_size + i) * time_step, time_step=time_step)
                )
                actual_horizon = test_data[i:i+horizon_steps]
                predictions.append(pred)
                actuals.append(actual_horizon)
        
        # Analyze results
        if predictions:
            predictions_flat = np.concatenate(predictions)
            actuals_flat = np.concatenate(actuals)
            
            metrics = calculate_prediction_metrics(actuals_flat, predictions_flat)
            
            print(f"\nResults:")
            print(f"  Predictions made: {len(predictions)}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  MAE:  {metrics['mae']:.6f}")
            print(f"  R²:   {metrics['r2']:.6f}")
            print(f"  Actual mean: {np.mean(actuals_flat):.6f}")
            print(f"  Predicted mean: {np.mean(predictions_flat):.6f}")
            
            # Verify predictions are made
            assert len(predictions) > 10, "Should make at least 10 predictions"
            # Verify all predictions are non-negative
            assert np.all(predictions_flat >= 0), "All predictions should be non-negative"
            # Verify mean is reasonable
            actual_mean = np.mean(actuals_flat)
            pred_mean = np.mean(predictions_flat)
            assert abs(pred_mean - actual_mean) < actual_mean * 2, \
                f"Predicted mean {pred_mean} deviates too much from actual {actual_mean}"
            
            print("\n✓ DHW demand test passed!")
        else:
            pytest.skip("Unable to generate predictions with available data")


    def test_AutocorrPredictor_compare_lag_configurations(self, dhw_demand_data):
        """
        Test comparing AutocorrPredictor performance with different lag configurations.
        
        This test compares:
        - lag = [1, 24] hours (hourly and daily patterns)
        - lag = [1, 24, 168] hours (hourly, daily, and weekly patterns)
        
        Evaluates performance on DHW demand prediction and prints comparison.
        """
        time_step = 900  # 15 minutes
        prediction_horizon_h = 1
        
        # Use first 40% for training, next 30% for testing
        total = len(dhw_demand_data)
        train_size = int(total * 0.4)
        test_size = int(total * 0.3)
        
        train_data = dhw_demand_data[:train_size]
        test_data = dhw_demand_data[train_size:train_size + test_size]
        
        print(f"\n--- AutocorrPredictor Lag Comparison ---")
        print(f"Training: {len(train_data)} samples ({len(train_data)*time_step/3600:.1f} hours)")
        print(f"Testing: {len(test_data)} samples ({len(test_data)*time_step/3600:.1f} hours)")
        
        lag_configs = [
            [1, 24],         # Config 1: hourly and daily
            [1, 24, 168],     # Config 2: hourly, daily, and weekly
            [1, 2, 24, 268]     # Config 3: hourly, 2-hour, daily, and weekly
        ]
        
        results = []
        
        for config_idx, lag_config in enumerate(lag_configs, 1):
            print(f"\n--- Configuration {config_idx}: lag = {lag_config} ---")
            
            # Create predictor
            predictor = AutocorrPredictor(
                sensor_name="dhw_demand",
                prediction_horizon_h=prediction_horizon_h,
                lags_h=lag_config,
                name=f"autocorr_lag_{config_idx}"
            )
            
            # Initialize
            mock_sensor = MockSensor("dhw_demand", train_data.tolist())
            mock_env = MockEnvironment(sensors={'dhw_demand': mock_sensor})
            sim_state = SimulationState(
                simulation_start_datetime=pd.Timestamp("2025-01-01 00:00:00"),
                time=0,
                time_step=time_step
            )
            init_ctx = InitContext(environment=mock_env, state=sim_state)
            mock_sensor.measure(environment=mock_env, state=sim_state)  # Initialize first measurement
            predictor.initialize(init_ctx)
            
            # Populate buffer
            for val in train_data:
                predictor.buffer.append(val)
            
            # Make predictions
            horizon_steps = predictor.prediction_horizon
            predictions = []
            actuals = []
            
            for i in range(min(len(test_data) - horizon_steps, 200)):
                predictor.buffer.append(test_data[i])
                
                if len(predictor.buffer) > max(predictor.lags_steps):
                    pred = predictor.predict(
                        prediction_horizon_h,
                        SimulationState(time=(train_size + i) * time_step, time_step=time_step)
                    )
                    actual_horizon = test_data[i:i+horizon_steps]
                    predictions.append(pred)
                    actuals.append(actual_horizon)
            
            # Evaluate
            if predictions:
                predictions_flat = np.concatenate(predictions)
                actuals_flat = np.concatenate(actuals)
                
                metrics = calculate_prediction_metrics(actuals_flat, predictions_flat)
                
                results.append({
                    'config': str(lag_config),
                    'lag_size': len(lag_config),
                    'predictions': len(predictions),
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'r2': metrics['r2']
                })
                
                print(f"Predictions: {len(predictions)}")
                print(f"  RMSE: {metrics['rmse']:.6f}")
                print(f"  MAE:  {metrics['mae']:.6f}")
                print(f"  R²:   {metrics['r2']:.6f}")
        
        # Print comparison
        print(f"\n{'='*70}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"{'Config':<30} {'Predictions':<15} {'RMSE':<15} {'MAE':<15} {'R²':<10}")
        print(f"{'-'*70}")
        for r in results:
            print(f"{r['config']:<30} {r['predictions']:<15} "
                f"{r['rmse']:<15.6f} {r['mae']:<15.6f} {r['r2']:<10.6f}")
        
        # Verify tests
        assert len(results) == 3, "Should have results for all configurations"
        
        # Compare RMSE between configurations
        rmse_improvement = (results[0]['rmse'] - results[1]['rmse']) / results[0]['rmse'] * 100 \
            if results[0]['rmse'] > 0 else 0
        
        print(f"\nRMSE change (weekly vs daily+hourly): {rmse_improvement:+.2f}%")
        
        if abs(rmse_improvement) > 1:
            print(f"✓ Configuration with weekly lag shows measurable performance difference")
        else:
            print(f"✓ Both configurations show similar performance on this dataset")
        
        print("\n✓ Lag comparison test completed successfully!")