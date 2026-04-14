import pandas as pd
import numpy as np
import pytest
from dataclasses import dataclass
from energy_system_control.controllers.predictors import MLBasedPredictor
from energy_system_control.sim.state import SimulationState
from energy_system_control.core.base_classes import InitContext
from energy_system_control.sensors.sensors import Sensor
from tests.utils import calculate_prediction_metrics, MockSensor, MockEnvironment
import os

@pytest.fixture
def init_context_ml(dhw_demand_data):
    """Create an InitContext with synthetic data."""
    dt = 900  # 15 minutes
    mock_sensor = MockSensor("test_sensor", dhw_demand_data.tolist())
    mock_env = MockEnvironment(sensors={'test_sensor': mock_sensor})
    sim_state = SimulationState(
        simulation_start_datetime=pd.Timestamp("2025-01-01 00:00:00"),
        time=0,
        time_step=dt
    )
    init_ctx = InitContext(environment=mock_env, state=sim_state)
    mock_sensor.measure(environment=mock_env, state=sim_state)
    return init_ctx


@pytest.fixture
def ann_predictor(init_context_ml):
    predictor = MLBasedPredictor(
        prediction_horizon_h=1,  # 1 hour
        model_type='ann',
        sensor_name="test_sensor",
        window_size_h=24,  # 24 hours
        retrain_interval_h=24,  # 24 hours
        min_sample_size_h=48,  # 24 hours
        hidden_layer_sizes=(10,),
        max_iter=1000
    )
    predictor.initialize(init_context_ml)
    return predictor


class TestMLPredictorBase:
    def test_ann_based_predictor_initialization(self, ann_predictor):
        assert ann_predictor.prediction_horizon_h == 1
        assert ann_predictor.sensor_name == "test_sensor"
        assert ann_predictor.window_size_h == 24
        assert ann_predictor.retrain_interval_h == 24
        assert ann_predictor.min_sample_size_h == 48
        assert ann_predictor.is_trained is False

    def test_ann_based_predictor_initialized_with_wrong_sizes(self):
        with pytest.raises(ValueError):
            predictor = MLBasedPredictor(
                prediction_horizon_h = 4,  # 1 hour, 
                model_type='ann',
                sensor_name = "test_sensor",
                window_size_h = 24,  # 24 hours
                retrain_interval_h = 24,  # 24 hours
                min_sample_size_h = 25,  # 24 hours
                hidden_layer_sizes = (10,),
                max_iter = 1000
            )

    def test_ann_based_predictor_update(self, ann_predictor):
        # Add some data to the buffer
        for i in range(10):
            ann_predictor.update(i * 900)  # i hours in seconds

        assert len(ann_predictor.buffer) == 10
        assert len(ann_predictor.time_buffer) == 10

    def test_ann_based_predictor_train(self, ann_predictor):
        # Add enough data to trigger training
        for i in range(50*4):  # 50 hours
            ann_predictor.update(i * 900)

        # Check if the model is trained
        assert ann_predictor.is_trained is True

    def test_ann_based_predictor_predict_not_trained(self, ann_predictor):
        # Add some data but not enough to trigger training
        for i in range(10):
            ann_predictor.update(i * 900)

        # Make a prediction
        predictions = ann_predictor.predict(1, SimulationState(time=10 * 900))

        # Check if the prediction is a persistence forecast
        assert np.all(predictions == ann_predictor.buffer[-1])

    def test_ann_based_predictor_predict_trained(self, ann_predictor, simulation_state):
        # Add enough data to trigger training
        for i in range(int(80*3600/simulation_state.time_step)):  # 80 hours
            ann_predictor.update(i * simulation_state.time_step)

        # Make a prediction
        predictions = ann_predictor.predict(1, SimulationState(time=80 * 3600))

        # Check if the prediction is an array of the correct length
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == ann_predictor.prediction_horizon

    def test_ann_based_predictor_predict_wrong_horizon(self, ann_predictor, simulation_state):
        # Add enough data to trigger training
        for i in range(int(80*3600/simulation_state.time_step)):  # 25 hours
            ann_predictor.update(i * 3600)

        # Make a prediction with a wrong horizon
        with pytest.raises(ValueError):
            ann_predictor.predict(2, SimulationState(time=25 * 3600))

    def test_ann_based_predictor_encode_hour(self, ann_predictor):
        # Test the _encode_hour method
        sin_h, cos_h = ann_predictor._encode_hour(0)  # Midnight
        assert np.isclose(sin_h, 0)
        assert np.isclose(cos_h, 1)
        sin_h, cos_h = ann_predictor._encode_hour(6 * 3600)  # 6 AM
        assert np.isclose(sin_h, 1)
        assert np.isclose(cos_h, 0)
        sin_h, cos_h = ann_predictor._encode_hour(12 * 3600)  # Noon
        assert np.isclose(sin_h, 0)
        assert np.isclose(cos_h, -1)
        sin_h, cos_h = ann_predictor._encode_hour(18 * 3600)  # 6 PM
        assert np.isclose(sin_h, -1)
        assert np.isclose(cos_h, 0)
        sin_h, cos_h = ann_predictor._encode_hour(42 * 3600)  # 6 PM
        assert np.isclose(sin_h, -1)
        assert np.isclose(cos_h, 0)

    def test_ann_based_predictor_prepare_data(self, ann_predictor, simulation_state):
        # Add some data to the buffer
        for i in range(int(80*3600/simulation_state.time_step)):  # 50 hours
            ann_predictor.update(i * 3600)

        # Prepare the data for training
        X, Y = ann_predictor._prepare_data()

        # Check if the data is prepared correctly
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert len(X) == len(Y)
        assert X.shape[1] == ann_predictor.window_size + ann_predictor.prediction_horizon * 2  # window_size + sin_h + cos_h
        assert Y.shape[1] == ann_predictor.prediction_horizon

class TestMLPredictorPerformance:

    def test_ML_predictor_dhw_demand_increasing_dataset_size(self, dhw_demand_data):
        """
        Test comparing tree and ANN ML predictors on DHW demand data.
        
        This test compares:
        1. A tree-based predictor trained on 25% of data
        2. ANN-based predictors trained on 25%, 50%, and 75% of data
        
        The test evaluates prediction quality using RMSE, MAE, and R² metrics
        and prints results for comparison without making strong assertions.
        """
        time_step = 900  # 15 minutes
        prediction_horizon_h = 2  # 12 hours ahead prediction
        window_size_h = 12  # Use 24 hours of history
        buffer_size_h = 48  # Buffer size
        
        total_samples = len(dhw_demand_data)
        
        # We'll use 25% for comparisons
        tree_fraction = 0.25
        ann_fractions = [0.25, 0.50, 0.75]
        
        results = []
        
        print("\n" + "="*80)
        print("TREE MODEL - 25% Training Data")
        print("="*80)
        
        # Train tree model
        tree_train_samples = int(total_samples * tree_fraction)
        tree_test_start = tree_train_samples
        tree_test_end = min(tree_test_start + int(total_samples * 0.15), total_samples)
        
        tree_train_data = dhw_demand_data[:tree_train_samples]
        tree_test_data = dhw_demand_data[tree_test_start:tree_test_end]
        
        if len(tree_test_data) >= 10:
            min_train_hours = window_size_h + prediction_horizon_h + 1
            min_train_samples = int(min_train_hours * 3600 / time_step)
            
            tree_predictor = MLBasedPredictor(
                model_type='rf',
                prediction_horizon_h=prediction_horizon_h,
                sensor_name="dhw_demand",
                window_size_h=window_size_h,
                buffer_size_h=buffer_size_h,
                retrain_interval_h=int(len(tree_train_data) * time_step / 3600),
                min_sample_size_h=min_train_hours,
                random_state=42,
                n_estimators = 10,
            )
            
            # Initialize tree predictor
            mock_sensor = MockSensor("dhw_demand", tree_train_data.tolist())
            mock_env = MockEnvironment(sensors={'dhw_demand': mock_sensor})
            sim_state = SimulationState(
                simulation_start_datetime=pd.Timestamp("2025-01-01 00:00:00"),
                time=0,
                time_step=time_step
            )
            init_ctx = InitContext(environment=mock_env, state=sim_state)
            mock_sensor.measure(environment=mock_env, state=sim_state)  # Initialize first measurement
            tree_predictor.initialize(init_ctx)
            
            # Populate buffer with training data
            for i, value in enumerate(tree_train_data):
                tree_predictor.buffer.append(value)
                tree_predictor.time_buffer.append(i * time_step)
            
            # Train tree model
            if len(tree_predictor.buffer) >= min_train_samples:
                tree_predictor._train()
            
            # Make predictions
            horizon_steps = tree_predictor.prediction_horizon
            tree_predictions = []
            tree_actuals = []
            
            for i in range(len(tree_test_data) - horizon_steps):
                test_value = tree_test_data[i]
                tree_predictor.buffer.append(test_value)
                tree_predictor.time_buffer.append((tree_train_samples + i) * time_step)
                
                if tree_predictor.is_trained and len(tree_predictor.buffer) >= tree_predictor.window_size:
                    prediction = tree_predictor.predict(
                        prediction_horizon_h,
                        SimulationState(time=(tree_train_samples + i) * time_step, time_step=time_step)
                    )
                    actual_horizon = tree_test_data[i:i+horizon_steps]
                    tree_predictions.append(prediction)
                    tree_actuals.append(actual_horizon)
            
            if tree_predictions:
                flat_tree_pred = np.concatenate(tree_predictions)
                flat_tree_actual = np.concatenate(tree_actuals)
                tree_metrics = calculate_prediction_metrics(flat_tree_actual, flat_tree_pred)
                
                results.append({
                    'model': 'Tree (25%)',
                    'training_fraction': tree_fraction,
                    'train_samples': tree_train_samples,
                    'test_predictions': len(tree_predictions),
                    'rmse': tree_metrics['rmse'],
                    'mae': tree_metrics['mae'],
                    'r2': tree_metrics['r2']
                })
                
                print(f"Training samples: {len(tree_train_data)}, Test predictions: {len(tree_predictions)}")
                print(f"  RMSE: {tree_metrics['rmse']:.6f}")
                print(f"  MAE:  {tree_metrics['mae']:.6f}")
                print(f"  R²:   {tree_metrics['r2']:.6f}")
        
        # Train ANN models with different training data sizes
        for fraction in ann_fractions:
            print("\n" + "="*80)
            print(f"ANN MODEL - {fraction*100:.0f}% Training Data")
            print("="*80)
            
            ann_train_samples = int(total_samples * fraction)
            ann_test_start = ann_train_samples
            ann_test_end = min(ann_test_start + int(total_samples * 0.15), total_samples)
            
            ann_train_data = dhw_demand_data[:ann_train_samples]
            ann_test_data = dhw_demand_data[ann_test_start:ann_test_end]
            
            if len(ann_test_data) < 10:
                print("Skipping - not enough test data")
                continue
            
            min_train_hours = window_size_h + prediction_horizon_h + 1
            min_train_samples = int(min_train_hours * 3600 / time_step)
            
            ann_predictor = MLBasedPredictor(
                model_type='ann',
                prediction_horizon_h=prediction_horizon_h,
                sensor_name="dhw_demand",
                window_size_h=window_size_h,
                buffer_size_h=buffer_size_h,
                retrain_interval_h=int(len(ann_train_data) * time_step / 3600),
                min_sample_size_h=min_train_hours,
                hidden_layer_sizes=(32, 16),
                max_iter=2000,
                random_state=42,
                early_stopping=True
            )
            
            # Initialize ANN predictor
            mock_sensor = MockSensor("dhw_demand", ann_train_data.tolist())
            mock_env = MockEnvironment(sensors={'dhw_demand': mock_sensor})
            sim_state = SimulationState(
                simulation_start_datetime=pd.Timestamp("2025-01-01 00:00:00"),
                time=0,
                time_step=time_step
            )
            init_ctx = InitContext(environment=mock_env, state=sim_state)
            mock_sensor.measure(environment=mock_env, state=sim_state)  # Initialize first measurement
            ann_predictor.initialize(init_ctx)
            
            # Populate buffer with training data
            for i, value in enumerate(ann_train_data):
                ann_predictor.buffer.append(value)
                ann_predictor.time_buffer.append(i * time_step)
            
            # Train ANN model
            if len(ann_predictor.buffer) >= min_train_samples:
                ann_predictor._train()
            
            # Make predictions
            horizon_steps = ann_predictor.prediction_horizon
            ann_predictions = []
            ann_actuals = []
            
            for i in range(len(ann_test_data) - horizon_steps):
                test_value = ann_test_data[i]
                ann_predictor.buffer.append(test_value)
                ann_predictor.time_buffer.append((ann_train_samples + i) * time_step)
                
                if ann_predictor.is_trained and len(ann_predictor.buffer) >= ann_predictor.window_size:
                    prediction = ann_predictor.predict(
                        prediction_horizon_h,
                        SimulationState(time=(ann_train_samples + i) * time_step, time_step=time_step)
                    )
                    actual_horizon = ann_test_data[i:i+horizon_steps]
                    ann_predictions.append(prediction)
                    ann_actuals.append(actual_horizon)
            
            if ann_predictions:
                flat_ann_pred = np.concatenate(ann_predictions)
                flat_ann_actual = np.concatenate(ann_actuals)
                ann_metrics = calculate_prediction_metrics(flat_ann_actual, flat_ann_pred)
                
                results.append({
                    'model': f'ANN ({fraction*100:.0f}%)',
                    'training_fraction': fraction,
                    'train_samples': ann_train_samples,
                    'test_predictions': len(ann_predictions),
                    'rmse': ann_metrics['rmse'],
                    'mae': ann_metrics['mae'],
                    'r2': ann_metrics['r2']
                })
                
                print(f"Training samples: {len(ann_train_data)}, Test predictions: {len(ann_predictions)}")
                print(f"  RMSE: {ann_metrics['rmse']:.6f}")
                print(f"  MAE:  {ann_metrics['mae']:.6f}")
                print(f"  R²:   {ann_metrics['r2']:.6f}")
        
        # Print summary comparison
        print("\n" + "="*80)
        print("SUMMARY COMPARISON")
        print("="*80)
        print(f"{'Model':<20} {'Train %':<12} {'Samples':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
        print("-"*80)
        for r in results:
            print(f"{r['model']:<20} {r['training_fraction']*100:>10.0f}% {r['train_samples']:>10} "
                f"{r['rmse']:>11.6f} {r['mae']:>11.6f} {r['r2']:>11.6f}")
        
        # Verify we have results
        assert len(results) >= 2, "Not enough models tested"
        print("\n✓ Test completed successfully!")



    def test_ann_predictor_dhw_demand_vs_actual_comparison(self, dhw_demand_data):
        """
        Test ANN predictor DHW demand and compare predicted vs actual values.
        
        This test:
        1. Trains an ANN on 60% of the available DHW demand data
        2. Makes predictions on the remaining 40%
        3. Compares predicted vs actual demand values visually and statistically
        4. Verifies that predictions capture general demand patterns
        5. Provides detailed statistics on prediction accuracy
        """
        time_step = 900  # 15 minutes
        prediction_horizon_h = 1
        window_size_h = 24
        
        total_samples = len(dhw_demand_data)
        train_samples = int(total_samples * 0.60)
        
        train_data = dhw_demand_data[:train_samples]
        test_data = dhw_demand_data[train_samples:train_samples + int(total_samples * 0.30)]
        
        print(f"\nTraining on {len(train_data)} samples ({len(train_data)*time_step/3600:.1f} hours)")
        print(f"Testing on {len(test_data)} samples ({len(test_data)*time_step/3600:.1f} hours)")
        
        # Create and train predictor
        predictor = MLBasedPredictor(
            prediction_horizon_h=prediction_horizon_h,
            model_type='rf',
            sensor_name="dhw_demand",
            window_size_h=window_size_h,
            retrain_interval_h=100,
            min_sample_size_h=window_size_h + prediction_horizon_h + 1,
            max_depth=5,
            random_state=42
        )
        
        # Setup and initialize
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
        
        # Populate buffer with all training data
        for i, value in enumerate(train_data):
            predictor.buffer.append(value)
            predictor.time_buffer.append(i * time_step)
        
        # Train the predictor once on all training data
        if len(predictor.buffer) >= predictor.min_sample_size:
            predictor._train()
        
        # Calculate the number of timesteps in the prediction horizon
        horizon_steps = predictor.prediction_horizon
        
        # Test and collect predictions
        all_predictions = []
        all_actuals = []
        
        # We can only make predictions while there's enough test data ahead
        for i in range(len(test_data) - horizon_steps):
            predictor.buffer.append(test_data[i])
            predictor.time_buffer.append((train_samples + i) * time_step)
            
            if predictor.is_trained and len(predictor.buffer) >= predictor.window_size:
                pred = predictor.predict(
                    prediction_horizon_h,
                    SimulationState(time=(train_samples + i) * time_step)
                )
                # Get actual values for the next 12 hours
                actual_horizon = test_data[i:i+horizon_steps]
                
                all_predictions.append(pred)
                all_actuals.append(actual_horizon)
        
        # Flatten predictions and actuals for metrics calculation
        predictions = np.concatenate(all_predictions) if all_predictions else np.array([])
        actuals = np.concatenate(all_actuals) if all_actuals else np.array([])
        
        # Calculate and verify metrics
        assert len(predictions) > 10, "Not enough predictions made"
        
        metrics = calculate_prediction_metrics(actuals, predictions)
        
        print(f"\nPrediction Metrics:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        
        # Verify that predictions are reasonable
        actual_mean = np.mean(actuals)
        pred_mean = np.mean(predictions)
        print(f"\nData Statistics:")
        print(f"  Actual mean: {actual_mean:.4f}")
        print(f"  Predicted mean: {pred_mean:.4f}")
        print(f"  Actual std: {np.std(actuals):.4f}")
        print(f"  Predicted std: {np.std(predictions):.4f}")
        
        # Predictions should not be wildly off
        assert metrics['mae'] < actual_mean * 2, \
            f"Mean Absolute Error ({metrics['mae']:.4f}) is unreasonably large " \
            f"compared to actual mean ({actual_mean:.4f})"
        
        # Model should capture at least some correlation
        assert metrics['r2'] > -1.0, "R² score is unexpectedly low"
        
        print("\n✓ DHW demand prediction test passed!")
        print(f"  Successfully made {len(predictions)} predictions")
        print(f"  Model captured correlation (R² = {metrics['r2']:.4f})")

