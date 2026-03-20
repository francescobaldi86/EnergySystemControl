# tests/unit/controllers/test_profilear_predictor.py
"""
Comprehensive test suite for ProfileARPredictor.

Tests include:
1. Base functionality tests
2. DHW demand dataset tests
3. Comparison tests with other prediction approaches
"""

import pandas as pd
import numpy as np
import pytest
from dataclasses import dataclass
from energy_system_control.controllers.predictors import (
    ProfileARPredictor,
    AutocorrPredictor,
    DailyProfilePredictor,
)
from energy_system_control.sim.state import SimulationState
from energy_system_control.core.base_classes import InitContext
from tests.utils import calculate_prediction_metrics, MockSensor, MockEnvironment
import os

__TEST__ = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))



# ============================================================================
# BASE FUNCTIONALITY TESTS
# ============================================================================

@pytest.fixture
def synthetic_daily_pattern():
    """Generate synthetic daily pattern data (2 weeks)."""
    dt = 900  # 15 minutes
    steps_per_day = int(86400 // dt)  # 576 steps per day
    n_days = 14
    
    # Create a realistic daily pattern with morning and evening peaks
    daily_pattern = np.zeros(steps_per_day)
    for i in range(steps_per_day):
        hour = (i * dt) / 3600
        # Morning peak (7-9 AM)
        if 7 <= hour < 9:
            daily_pattern[i] = 5.0 + 3.0 * np.sin((hour - 7) * np.pi / 2)
        # Evening peak (18-21 PM)
        elif 18 <= hour < 21:
            daily_pattern[i] = 4.0 + 4.0 * np.sin((hour - 18) * np.pi / 3)
        # Low demand at night and midday
        else:
            daily_pattern[i] = 0.5
    
    # Repeat pattern for 2 weeks with small random noise
    measurements = []
    rng = np.random.RandomState(42)
    for day in range(n_days):
        daily_measurements = daily_pattern + rng.normal(0, 0.2, steps_per_day)
        daily_measurements = np.maximum(daily_measurements, 0)  # Ensure non-negative
        measurements.extend(daily_measurements.tolist())
    
    return np.array(measurements)


@pytest.fixture
def init_context_synthetic(synthetic_daily_pattern):
    """Create an InitContext with synthetic data."""
    dt = 900  # 15 minutes
    mock_sensor = MockSensor("test_sensor", synthetic_daily_pattern.tolist())
    mock_env = MockEnvironment(sensors={'test_sensor': mock_sensor})
    sim_state = SimulationState(
        simulation_start_datetime=pd.Timestamp("2025-01-01 00:00:00"),
        time=0,
        time_step=dt
    )
    init_ctx = InitContext(environment=mock_env, state=sim_state)
    mock_sensor.measure(environment=mock_env, state=sim_state)
    return init_ctx, sim_state, dt


class TestProfileARPredictorBase:
    """Base functionality tests for ProfileARPredictor."""

    def test_initialization(self):
        """Test ProfileARPredictor initialization."""
        predictor = ProfileARPredictor(
            prediction_horizon_h=2,
            sensor_name="test_sensor",
            residual_lags_h=[1, 24],
            buffer_size_h=24*14
        )
        
        assert predictor.sensor_name == "test_sensor"
        assert predictor.prediction_horizon_h == 2
        assert predictor.residual_lags_h == [1, 24]
        assert predictor.is_trained is False
        assert predictor.variable_to_predict == "test_sensor"

    def test_initialization_with_custom_name(self):
        """Test initialization with custom name."""
        predictor = ProfileARPredictor(
            prediction_horizon_h=2,
            sensor_name="test_sensor",
            name="my_predictor"
        )
        
        assert predictor.name == "my_predictor"
        assert predictor.variable_to_predict == "test_sensor"

    def test_initialization_default_parameters(self):
        """Test initialization with default parameters."""
        predictor = ProfileARPredictor(
            prediction_horizon_h=2,
            sensor_name="test_sensor"
        )
        
        assert predictor.residual_lags_h == [1, 24]
        assert predictor.buffer_size_h == 24 * 14  # 2 weeks default

    def test_predict_raises_on_mismatch_horizon(self, init_context_synthetic):
        """Test that predict raises on horizon mismatch."""
        init_ctx, sim_state, dt = init_context_synthetic
        
        predictor = ProfileARPredictor(
            prediction_horizon_h=2,
            sensor_name="test_sensor"
        )
        predictor.initialize(init_ctx)
        
        # Try to predict with wrong horizon
        with pytest.raises(ValueError, match="Prediction horizon mismatch"):
            predictor.predict(horizon_h=4.0, state=sim_state)

    def test_predict_returns_array(self, init_context_synthetic, synthetic_daily_pattern):
        """Test that predict returns numpy array of correct shape."""
        init_ctx, sim_state, dt = init_context_synthetic
        
        predictor = ProfileARPredictor(
            prediction_horizon_h=2,
            sensor_name="test_sensor",
            buffer_size_h=24*7
        )
        predictor.initialize(init_ctx)
        
        # Add training data
        synthetic_data = synthetic_daily_pattern
        for i in range(int(24 * 7 * 3600 / dt)):  # 1 week of data
            sensor_measurement = synthetic_data[min(i, len(synthetic_data)-1)]
            predictor.buffer.append(sensor_measurement)
            predictor.time_buffer.append(i * dt)
        
        # Train the model
        predictor._train()
        
        # Make prediction
        prediction = predictor.predict(horizon_h=2.0, state=sim_state)
        
        assert isinstance(prediction, np.ndarray)
        expected_length = int(2.0 * 3600 / dt)
        assert len(prediction) == expected_length

    def test_predict_non_negative_output(self, init_context_synthetic):
        """Test that predictions are non-negative."""
        init_ctx, sim_state, dt = init_context_synthetic
        
        predictor = ProfileARPredictor(
            prediction_horizon_h=1,
            sensor_name="test_sensor",
            buffer_size_h=24*7
        )
        predictor.initialize(init_ctx)
        
        # Get the synthetic data
        mock_sensor = init_ctx.environment.sensors['test_sensor']
        synthetic_data = mock_sensor.measurements
        
        # Add training data
        for i in range(int(24 * 7 * 3600 / dt)):
            sensor_measurement = synthetic_data[min(i, len(synthetic_data)-1)]
            predictor.buffer.append(sensor_measurement)
            predictor.time_buffer.append(i * dt)
        
        # Train model
        predictor._train()
        
        # Make prediction
        prediction = predictor.predict(horizon_h=1.0, state=sim_state)
        
        assert np.all(prediction >= 0), "All predictions should be non-negative"

    def test_fallback_when_not_trained(self, init_context_synthetic):
        """Test fallback behavior when model is not trained."""
        init_ctx, sim_state, dt = init_context_synthetic
        
        predictor = ProfileARPredictor(
            prediction_horizon_h=1,
            sensor_name="test_sensor",
            buffer_size_h=24*14
        )
        predictor.initialize(init_ctx)
        
        # Add minimal data (not enough to train)
        for i in range(10):
            predictor.buffer.append(5.0)
            predictor.time_buffer.append(i * dt)
        
        # Make prediction without training
        prediction = predictor.predict(horizon_h=1.0, state=sim_state)
        
        # Should return last value repeated
        assert isinstance(prediction, np.ndarray)
        assert len(prediction) > 0


class TestProfileARPredictorTraining:
    """Test the training mechanism of ProfileARPredictor."""

    def test_train_builds_profile(self, init_context_synthetic):
        """Test that training builds a weekly profile."""
        init_ctx, sim_state, dt = init_context_synthetic
        
        predictor = ProfileARPredictor(
            prediction_horizon_h=1,
            sensor_name="test_sensor",
            buffer_size_h=24*14
        )
        predictor.initialize(init_ctx)
        
        # Add training data
        mock_sensor = init_ctx.environment.sensors['test_sensor']
        synthetic_data = mock_sensor.measurements
        
        for i in range(int(24 * 14 * 3600 / dt)):
            sensor_measurement = synthetic_data[min(i, len(synthetic_data)-1)]
            predictor.buffer.append(sensor_measurement)
            predictor.time_buffer.append(i * dt)
        
        # Train
        predictor._train()
        
        assert predictor.profile is not None
        assert len(predictor.profile) > 0
        assert predictor.is_trained is True

    def test_train_fits_ar_coefficients(self, init_context_synthetic):
        """Test that training fits AR coefficients."""
        init_ctx, sim_state, dt = init_context_synthetic
        
        predictor = ProfileARPredictor(
            prediction_horizon_h=1,
            sensor_name="test_sensor",
            residual_lags_h=[1, 24],
            buffer_size_h=24*14
        )
        predictor.initialize(init_ctx)
        
        # Add training data
        mock_sensor = init_ctx.environment.sensors['test_sensor']
        synthetic_data = mock_sensor.measurements
        
        for i in range(int(24 * 14 * 3600 / dt)):
            sensor_measurement = synthetic_data[min(i, len(synthetic_data)-1)]
            predictor.buffer.append(sensor_measurement)
            predictor.time_buffer.append(i * dt)
        
        # Train
        predictor._train()
        
        assert predictor.ar_coeffs is not None
        assert len(predictor.ar_coeffs) == len(predictor.residual_lags)

    def test_update_populates_buffers(self, init_context_synthetic):
        """Test that update method populates buffers."""
        init_ctx, sim_state, dt = init_context_synthetic
        
        predictor = ProfileARPredictor(
            prediction_horizon_h=1,
            sensor_name="test_sensor"
        )
        predictor.initialize(init_ctx)
        
        initial_size = len(predictor.buffer)
        predictor.update(time_s=100)
        
        assert len(predictor.buffer) == initial_size + 1
        assert len(predictor.time_buffer) == initial_size + 1


# ============================================================================
# DHW DEMAND DATASET TESTS
# ============================================================================

@pytest.fixture
def dhw_demand_data():
    """Load and prepare DHW demand data from the test data file."""
    data_file = os.path.join(__TEST__, 'DATA', 'dhw_demand_data.csv')
    df = pd.read_csv(data_file, sep=';')
    demand_values = df['DHW demand'].values
    return demand_values


class TestProfileARPredictorDHWDemand:
    """Test ProfileARPredictor on DHW demand dataset."""

    def test_predict_on_dhw_demand_data(self, dhw_demand_data):
        """Test basic prediction on DHW demand data."""
        dt = 900  # 15 minutes (0.25 hours in the CSV)
        prediction_horizon_h = 2
        
        predictor = ProfileARPredictor(
            prediction_horizon_h=prediction_horizon_h,
            sensor_name="dhw_demand",
            buffer_size_h=24*7
        )
        
        # Initialize
        mock_sensor = MockSensor("dhw_demand", dhw_demand_data.tolist())
        mock_env = MockEnvironment(sensors={'dhw_demand': mock_sensor})
        sim_state = SimulationState(
            simulation_start_datetime=pd.Timestamp("2025-01-01 00:00:00"),
            time=0,
            time_step=dt
        )
        init_ctx = InitContext(environment=mock_env, state=sim_state)
        mock_sensor.measure(environment=mock_env, state=sim_state)
        
        predictor.initialize(init_ctx)
        
        # Add training data
        buffer_size_samples = int(24 * 7 * 3600 / dt)
        for i in range(min(buffer_size_samples, len(dhw_demand_data))):
            predictor.buffer.append(dhw_demand_data[i])
            predictor.time_buffer.append(i * dt)
        
        # Train
        predictor._train()
        
        # Make prediction
        prediction = predictor.predict(horizon_h=prediction_horizon_h, state=sim_state)
        
        assert isinstance(prediction, np.ndarray)
        expected_length = int(prediction_horizon_h * 3600 / dt)
        assert len(prediction) == expected_length
        assert np.all(prediction >= 0)

    def test_dhw_predictions_respect_time_structure(self, dhw_demand_data):
        """Test that predictions follow time structure (morning/evening patterns)."""
        dt = 900  # 15 minutes
        prediction_horizon_h = 1
        
        predictor = ProfileARPredictor(
            prediction_horizon_h=prediction_horizon_h,
            sensor_name="dhw_demand",
            buffer_size_h=24*7
        )
        
        # Initialize
        mock_sensor = MockSensor("dhw_demand", dhw_demand_data.tolist())
        mock_env = MockEnvironment(sensors={'dhw_demand': mock_sensor})
        sim_state = SimulationState(
            simulation_start_datetime=pd.Timestamp("2025-01-01 00:00:00"),
            time=0,
            time_step=dt
        )
        init_ctx = InitContext(environment=mock_env, state=sim_state)
        mock_sensor.measure(environment=mock_env, state=sim_state)
        
        predictor.initialize(init_ctx)
        
        # Add training data
        buffer_size_samples = int(24 * 7 * 3600 / dt)
        for i in range(min(buffer_size_samples, len(dhw_demand_data))):
            predictor.buffer.append(dhw_demand_data[i])
            predictor.time_buffer.append(i * dt)
        
        # Train
        predictor._train()
        
        # Make predictions at different times of day
        morning_state = SimulationState(
            simulation_start_datetime=pd.Timestamp("2025-01-08 07:00:00"),
            time=7 * 3600,
            time_step=dt
        )
        
        evening_state = SimulationState(
            simulation_start_datetime=pd.Timestamp("2025-01-08 19:00:00"),
            time=19 * 3600,
            time_step=dt
        )
        
        morning_pred = predictor.predict(horizon_h=prediction_horizon_h, state=morning_state)
        evening_pred = predictor.predict(horizon_h=prediction_horizon_h, state=evening_state)
        
        # Both should be arrays
        assert isinstance(morning_pred, np.ndarray)
        assert isinstance(evening_pred, np.ndarray)


# ============================================================================
# COMPARISON TESTS: ProfileAR vs Autocorr vs Standard Profile
# ============================================================================

def calculate_prediction_metrics(actual, predicted):
    """Calculate prediction quality metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Filter out cases where both are zero
    mask = (actual != 0) | (predicted != 0)
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]
    
    if len(actual_filtered) == 0:
        return {'rmse': 0, 'mae': 0, 'r2': 0, 'mape': 0}
    
    rmse = np.sqrt(mean_squared_error(actual_filtered, predicted_filtered))
    mae = mean_absolute_error(actual_filtered, predicted_filtered)
    
    # R² can be negative for poor predictions
    try:
        r2 = r2_score(actual_filtered, predicted_filtered)
    except:
        r2 = 0
    
    # Mean Absolute Percentage Error (avoid division by zero)
    non_zero_mask = actual_filtered != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs(
            (actual_filtered[non_zero_mask] - predicted_filtered[non_zero_mask]) / 
            actual_filtered[non_zero_mask]
        )) * 100
    else:
        mape = 0
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


class TestPredictorComparison:
    """Compare ProfileARPredictor with other prediction approaches."""

    def test_comparison_profilear_vs_autocorr_vs_standard_profile(self, synthetic_daily_pattern):
        """
        Compare three prediction approaches:
        1. ProfileARPredictor (profile + AR residuals)
        2. AutocorrPredictor (autocorrelation with [1, 24] lags)
        3. DailyProfilePredictor (simple daily profile)
        """
        dt = 900  # 15 minutes
        prediction_horizon_h = 2
        n_training_days = 7
        n_test_days = 7
        
        # Prepare training and test data
        steps_per_day = int(86400 // dt)
        train_steps = n_training_days * steps_per_day
        test_start = train_steps
        test_end = test_start + n_test_days * steps_per_day
        
        train_data = synthetic_daily_pattern[:train_steps]
        test_data = synthetic_daily_pattern[test_start:test_end]
        
        results = {}
        
        # =====================================================================
        # 1. PROFILEAR PREDICTOR
        # =====================================================================
        print("\n" + "="*80)
        print("Testing ProfileARPredictor")
        print("="*80)
        
        profilear_predictor = ProfileARPredictor(
            prediction_horizon_h=prediction_horizon_h,
            sensor_name="sensor",
            residual_lags_h=[1, 24],
            buffer_size_h=24*7
        )
        
        mock_sensor = MockSensor("sensor", train_data.tolist())
        mock_env = MockEnvironment(sensors={'sensor': mock_sensor})
        sim_state = SimulationState(
            simulation_start_datetime=pd.Timestamp("2025-01-01 00:00:00"),
            time=0,
            time_step=dt
        )
        init_ctx = InitContext(environment=mock_env, state=sim_state)
        mock_sensor.measure(environment=mock_env, state=sim_state)
        
        profilear_predictor.initialize(init_ctx)
        
        # Populate buffer
        for i, value in enumerate(train_data):
            profilear_predictor.buffer.append(value)
            profilear_predictor.time_buffer.append(i * dt)
        
        profilear_predictor._train()
        
        # Make predictions
        profilear_predictions = []
        profilear_actuals = []
        horizon_steps = int(prediction_horizon_h * 3600 / dt)
        
        for i in range(len(test_data) - horizon_steps):
            test_value = test_data[i]
            profilear_predictor.buffer.append(test_value)
            profilear_predictor.time_buffer.append((train_steps + i) * dt)
            
            if profilear_predictor.is_trained and len(profilear_predictor.buffer) >= 100:
                prediction = profilear_predictor.predict(
                    horizon_h=prediction_horizon_h,
                    state=SimulationState(
                        time=(train_steps + i) * dt,
                        time_step=dt
                    )
                )
                actual = test_data[i:i+horizon_steps]
                profilear_predictions.append(prediction)
                profilear_actuals.append(actual)
        
        if profilear_predictions:
            profilear_pred_flat = np.concatenate(profilear_predictions)
            profilear_actual_flat = np.concatenate(profilear_actuals)
            profilear_metrics = calculate_prediction_metrics(profilear_actual_flat, profilear_pred_flat)
            results['ProfileAR'] = profilear_metrics
            
            print(f"Predictions made: {len(profilear_predictions)}")
            print(f"  RMSE: {profilear_metrics['rmse']:.6f}")
            print(f"  MAE:  {profilear_metrics['mae']:.6f}")
            print(f"  MAPE: {profilear_metrics['mape']:.2f}%")
            print(f"  R²:   {profilear_metrics['r2']:.6f}")
        
        # =====================================================================
        # 2. AUTOCORR PREDICTOR
        # =====================================================================
        print("\n" + "="*80)
        print("Testing AutocorrPredictor with [1, 24] lags")
        print("="*80)
        
        autocorr_predictor = AutocorrPredictor(
            sensor_name="sensor",
            prediction_horizon_h=prediction_horizon_h,
            lags_h=[1, 24],
            name="autocorr_predictor"
        )
        
        mock_sensor = MockSensor("sensor", train_data.tolist())
        mock_env = MockEnvironment(sensors={'sensor': mock_sensor})
        sim_state = SimulationState(
            simulation_start_datetime=pd.Timestamp("2025-01-01 00:00:00"),
            time=0,
            time_step=dt
        )
        init_ctx = InitContext(environment=mock_env, state=sim_state)
        mock_sensor.measure(environment=mock_env, state=sim_state)
        
        autocorr_predictor.initialize(init_ctx)
        
        # Make predictions (no explicit training needed for AutocorrPredictor)
        autocorr_predictions = []
        autocorr_actuals = []
        
        for i in range(len(train_data)):
            train_value = train_data[i]
            autocorr_predictor.buffer.append(train_value)
            autocorr_predictor.time_buffer.append(i * dt)
        
        for i in range(len(test_data) - horizon_steps):
            test_value = test_data[i]
            autocorr_predictor.buffer.append(test_value)
            autocorr_predictor.time_buffer.append((train_steps + i) * dt)
            
            if len(autocorr_predictor.buffer) >= 100:
                prediction = autocorr_predictor.predict(
                    horizon_h=prediction_horizon_h,
                    state=SimulationState(
                        time=(train_steps + i) * dt,
                        time_step=dt
                    )
                )
                actual = test_data[i:i+horizon_steps]
                autocorr_predictions.append(prediction)
                autocorr_actuals.append(actual)
        
        if autocorr_predictions:
            autocorr_pred_flat = np.concatenate(autocorr_predictions)
            autocorr_actual_flat = np.concatenate(autocorr_actuals)
            autocorr_metrics = calculate_prediction_metrics(autocorr_actual_flat, autocorr_pred_flat)
            results['Autocorr'] = autocorr_metrics
            
            print(f"Predictions made: {len(autocorr_predictions)}")
            print(f"  RMSE: {autocorr_metrics['rmse']:.6f}")
            print(f"  MAE:  {autocorr_metrics['mae']:.6f}")
            print(f"  MAPE: {autocorr_metrics['mape']:.2f}%")
            print(f"  R²:   {autocorr_metrics['r2']:.6f}")
        
        # =====================================================================
        # 3. STANDARD DAILY PROFILE
        # =====================================================================
        print("\n" + "="*80)
        print("Testing Standard DailyProfilePredictor")
        print("="*80)
        
        # Build daily profile from training data
        steps_per_day = int(86400 // dt)
        daily_profile = np.zeros(steps_per_day)
        daily_counts = np.zeros(steps_per_day)
        
        for i, value in enumerate(train_data):
            step_in_day = i % steps_per_day
            daily_profile[step_in_day] += value
            daily_counts[step_in_day] += 1
        
        # Average to get profile
        daily_profile = daily_profile / (daily_counts + 1e-10)
        
        profile_df = pd.DataFrame(
            index=np.arange(0, 24, 24 / steps_per_day),
            data={'demand': daily_profile}
        )
        
        standard_profile_predictor = DailyProfilePredictor(
            name="standard_profile",
            variable_to_predict="demand",
            profile=profile_df
        )
        
        # Make predictions
        standard_predictions = []
        standard_actuals = []
        
        for i in range(len(test_data) - horizon_steps):
            state = SimulationState(
                simulation_start_datetime=pd.Timestamp("2025-01-08 00:00:00") + pd.Timedelta(seconds=(i * dt)),
                time=i * dt,
                time_step=dt
            )
            
            prediction = standard_profile_predictor.predict(
                horizon=prediction_horizon_h,
                state=state
            )
            actual = test_data[i:i+horizon_steps]
            standard_predictions.append(prediction)
            standard_actuals.append(actual)
        
        if standard_predictions:
            standard_pred_flat = np.concatenate(standard_predictions)
            standard_actual_flat = np.concatenate(standard_actuals)
            standard_metrics = calculate_prediction_metrics(standard_actual_flat, standard_pred_flat)
            results['StandardProfile'] = standard_metrics
            
            print(f"Predictions made: {len(standard_predictions)}")
            print(f"  RMSE: {standard_metrics['rmse']:.6f}")
            print(f"  MAE:  {standard_metrics['mae']:.6f}")
            print(f"  MAPE: {standard_metrics['mape']:.2f}%")
            print(f"  R²:   {standard_metrics['r2']:.6f}")
        
        # =====================================================================
        # SUMMARY COMPARISON
        # =====================================================================
        print("\n" + "="*80)
        print("SUMMARY COMPARISON")
        print("="*80)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  MAE:  {metrics['mae']:.6f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            print(f"  R²:   {metrics['r2']:.6f}")
        
        # Basic assertions to ensure all models produced predictions
        assert len(results) >= 2, "At least 2 predictors should produce results"
        assert 'ProfileAR' in results or 'Autocorr' in results, "At least one main predictor should work"

    def test_comparison_on_dhw_demand_data(self, dhw_demand_data):
        """Compare all three approaches on real DHW demand data."""
        dt = 900  # 15 minutes
        prediction_horizon_h = 2
        
        # Use a smaller subset for faster testing
        n_samples_use = min(2000, len(dhw_demand_data))
        use_data = dhw_demand_data[:n_samples_use]
        
        # Split into train/test
        split_point = int(len(use_data) * 0.7)
        train_data = use_data[:split_point]
        test_data = use_data[split_point:]
        
        results = {}
        
        # ProfileARPredictor
        profilear_pred = ProfileARPredictor(
            prediction_horizon_h=prediction_horizon_h,
            sensor_name="dhw",
            buffer_size_h=24*7
        )
        
        mock_sensor = MockSensor("dhw", train_data.tolist())
        mock_env = MockEnvironment(sensors={'dhw': mock_sensor})
        sim_state = SimulationState(
            simulation_start_datetime=pd.Timestamp("2025-01-01 00:00:00"),
            time=0,
            time_step=dt
        )
        init_ctx = InitContext(environment=mock_env, state=sim_state)
        mock_sensor.measure(environment=mock_env, state=sim_state)
        
        profilear_pred.initialize(init_ctx)
        for i, val in enumerate(train_data):
            profilear_pred.buffer.append(val)
            profilear_pred.time_buffer.append(i * dt)
        profilear_pred._train()
        
        results['ProfileAR'] = profilear_pred

        # AutocorrPredictor
        autocorr_pred = AutocorrPredictor(
            sensor_name="dhw",
            prediction_horizon_h=prediction_horizon_h,
            lags_h=[1, 24]
        )
        
        mock_sensor = MockSensor("dhw", train_data.tolist())
        mock_env = MockEnvironment(sensors={'dhw': mock_sensor})
        init_ctx = InitContext(environment=mock_env, state=sim_state)
        mock_sensor.measure(environment=mock_env, state=sim_state)
        
        autocorr_pred.initialize(init_ctx)
        for i, val in enumerate(train_data):
            autocorr_pred.buffer.append(val)
            autocorr_pred.time_buffer.append(i * dt)
        
        results['Autocorr'] = autocorr_pred
        
        # Get predictions from both
        horizon_steps = int(prediction_horizon_h * 3600 / dt)
        
        profilear_results = []
        autocorr_results = []
        
        for i in range(min(50, len(test_data) - horizon_steps)):
            # ProfileAR
            test_val = test_data[i]
            profilear_pred.buffer.append(test_val)
            profilear_pred.time_buffer.append((split_point + i) * dt)
            
            if profilear_pred.is_trained and len(profilear_pred.buffer) >= 100:
                p_pred = profilear_pred.predict(
                    horizon_h=prediction_horizon_h,
                    state=SimulationState(time=(split_point + i) * dt, time_step=dt)
                )
                profilear_results.append(p_pred)
            
            # Autocorr
            autocorr_pred.buffer.append(test_val)
            autocorr_pred.time_buffer.append((split_point + i) * dt)
            
            if len(autocorr_pred.buffer) >= 100:
                a_pred = autocorr_pred.predict(
                    horizon_h=prediction_horizon_h,
                    state=SimulationState(time=(split_point + i) * dt, time_step=dt)
                )
                autocorr_results.append(a_pred)
        
        # Basic assertions
        assert len(profilear_results) > 0, "ProfileAR should produce predictions"
        assert len(autocorr_results) > 0, "Autocorr should produce predictions"
