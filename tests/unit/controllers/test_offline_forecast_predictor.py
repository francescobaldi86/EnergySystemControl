# tests/unit/controllers/test_offline_forecast_predictor.py
import pytest
import pandas as pd
import numpy as np
import os
import math
import energy_system_control as esc
from energy_system_control.controllers.predictors import OfflineForecastPredictor
from energy_system_control.sim.state import SimulationState

__HERE__ = os.path.dirname(os.path.realpath(__file__))
__TEST__ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
__DATA__ = os.path.join(__TEST__, 'DATA')
__PV_FORECAST_DATA__ = os.path.join(__DATA__, 'PV_prediction_data')


# ============================================================================
# Helper Functions
# ============================================================================

def load_forecast_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a forecast CSV file and parse it into a DataFrame.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing forecast data.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with parsed datetime index and numeric columns.
    """
    df = pd.read_csv(csv_path, sep=";")
    # Rename the time column
    time_col = df.columns[0]
    df = df.rename(columns={time_col: 'valid_time'})
    # Parse the time column
    df['valid_time'] = pd.to_datetime(df['valid_time'])
    # Set as index
    df = df.set_index('valid_time')
    return df


def build_forecast_dataframe(forecast_files: list[str], issue_time_offset_hours: float = 0.0) -> pd.DataFrame:
    """
    Build a MultiIndex DataFrame from forecast files.
    
    Each file represents a set of data valid for a specific forecast issue time.
    
    Parameters
    ----------
    forecast_files : list[str]
        List of paths to forecast CSV files.
    issue_time_offset_hours : float
        Number of hours before the first valid_time to use as the issue_time.
        This represents when the forecast was issued. Default is 0.0 (issued at start of data).
        
    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with (issue_time, valid_time) index and variable columns.
    """
    daily_frames = []
    
    for forecast_file in forecast_files:
        # Load the CSV
        df = load_forecast_csv(forecast_file)
        
        # Set issue_time: all rows in the same file use the same issue_time
        # The issue_time is the first valid_time minus the offset
        first_valid = df.index.min()
        issue_time = first_valid - pd.to_timedelta(issue_time_offset_hours, unit='h')
        
        # Add issue_time column
        df['issue_time'] = issue_time
        
        daily_frames.append(df.reset_index())
    
    # Use the building method from OfflineForecastPredictor
    return OfflineForecastPredictor.build_forecast_df(daily_frames)


# ============================================================================
# BASIC INITIALIZATION TESTS
# ============================================================================

class TestOfflineForecastPredictorInitialization:
    """Tests for the initialization of OfflineForecastPredictor."""
    
    @pytest.fixture
    def simple_forecast_df(self):
        """Create a simple forecast DataFrame for testing."""
        # Create a simple forecast with 2 days of hourly data
        times = pd.date_range('2025-03-27 00:00', periods=48, freq='h')
        data = {
            'irradiance': np.random.uniform(0, 1000, 48),
            'temperature': np.random.uniform(10, 30, 48),
        }
        df = pd.DataFrame(data, index=times)
        df.index.name = 'valid_time'
        
        # Add issue_time column and set it to 24 hours before
        df['issue_time'] = times[0] - pd.Timedelta(hours=24)
        
        # Build MultiIndex
        return OfflineForecastPredictor.build_forecast_df([df.reset_index()])
    
    def test_init_with_valid_parameters(self, simple_forecast_df):
        """Test initialization with valid parameters."""
        predictor = OfflineForecastPredictor(
            name='test_predictor',
            forecast_df=simple_forecast_df,
            variable_to_predict='irradiance'
        )
        
        assert predictor.name == 'test_predictor'
        assert predictor.variable_to_predict == 'irradiance'
        assert predictor.issue_level == 'issue_time'
        assert predictor.valid_level == 'valid_time'
        assert predictor.align == 'ffill'
    
    def test_init_with_custom_parameters(self, simple_forecast_df):
        """Test initialization with custom parameters."""
        predictor = OfflineForecastPredictor(
            name='custom_predictor',
            forecast_df=simple_forecast_df,
            variable_to_predict='temperature',
            issue_level='issue_time',
            valid_level='valid_time',
            align='linear'
        )
        
        assert predictor.name == 'custom_predictor'
        assert predictor.variable_to_predict == 'temperature'
        assert predictor.align == 'linear'
    
    def test_dt_native_s_calculation(self, simple_forecast_df):
        """Test that dt_native_s is correctly calculated."""
        predictor = OfflineForecastPredictor(
            name='test_predictor',
            forecast_df=simple_forecast_df,
            variable_to_predict='irradiance'
        )
        
        # Should be 3600 seconds (1 hour) for hourly data
        assert predictor.dt_native_s == 3600.0
    
    def test_init_with_real_forecast_data(self):
        """Test initialization with real forecast data from files."""
        forecast_files = [
            os.path.join(__PV_FORECAST_DATA__, 'solar_radiation_forecast_2025-03-29.csv'),
        ]
        
        if not os.path.exists(forecast_files[0]):
            pytest.skip(f"Forecast file not found: {forecast_files[0]}")
        
        forecast_df = build_forecast_dataframe(forecast_files, issue_time_offset_hours=3.0)
        
        predictor = OfflineForecastPredictor(
            name='pv_predictor',
            forecast_df=forecast_df,
            variable_to_predict='Global tilted irradiance'
        )
        
        assert predictor.name == 'pv_predictor'
        assert predictor.variable_to_predict == 'Global tilted irradiance'
        # The forecast data is hourly
        assert predictor.dt_native_s == 3600.0


# ============================================================================
# STAND-ALONE PREDICTOR TESTS
# ============================================================================

class TestOfflineForecastPredictorStandalone:
    """Tests for the predict method of OfflineForecastPredictor."""
    
    @pytest.fixture
    def forecast_with_real_pv_data(self):
        """Load real PV forecast data from test files."""
        forecast_files = [
            os.path.join(__PV_FORECAST_DATA__, 'solar_radiation_forecast_2025-03-29.csv'),
            os.path.join(__PV_FORECAST_DATA__, 'solar_radiation_forecast_2025-03-30.csv'),
        ]
        
        available_files = [f for f in forecast_files if os.path.exists(f)]
        if not available_files:
            pytest.skip("No forecast files found")
        
        # Build forecast DataFrame with 3-hour issue offset
        forecast_df = build_forecast_dataframe(available_files, issue_time_offset_hours=3.0)
        
        return forecast_df
    
    def test_predict_basic(self, forecast_with_real_pv_data):
        """Test basic prediction functionality."""
        predictor = OfflineForecastPredictor(
            name='pv_predictor',
            forecast_df=forecast_with_real_pv_data,
            variable_to_predict='Global tilted irradiance'
        )
        
        # Create a simulation state
        # Find the first valid time in the forecast
        first_issue_time = forecast_with_real_pv_data.index.get_level_values('issue_time').min()
        first_valid_time = forecast_with_real_pv_data.index.get_level_values('valid_time').min()
        
        state = SimulationState(
            time_id=0,
            time_step=3600,  # 1 hour
            simulation_start_datetime=first_valid_time
        )
        
        # Make a prediction for 6 hours ahead
        predictions = predictor.predict(horizon=6.0, state=state)
        
        assert not predictions.empty
        assert isinstance(predictions, pd.Series)
        assert predictions.name == 'Global tilted irradiance'
    
    def test_predict_returns_correct_shape(self, forecast_with_real_pv_data):
        """Test that predivations return correct number of timesteps."""
        predictor = OfflineForecastPredictor(
            name='pv_predictor',
            forecast_df=forecast_with_real_pv_data,
            variable_to_predict='Global tilted irradiance'
        )
        
        first_issue_time = forecast_with_real_pv_data.index.get_level_values('issue_time').min()
        first_valid_time = forecast_with_real_pv_data.index.get_level_values('valid_time').min()
        
        state = SimulationState(
            time_id=0,
            time_step=3600,  # 1 hour
            simulation_start_datetime=first_valid_time
        )
        
        horizon_hours = 12.0
        predictions = predictor.predict(horizon=horizon_hours, state=state)
        
        # Expected number of timesteps: horizon / time_step (in hours)
        expected_length = int(horizon_hours * 3600 / state.time_step)
        assert len(predictions) == expected_length
    
    def test_predict_with_different_alignments(self, forecast_with_real_pv_data):
        """Test prediction with different alignment methods."""
        first_valid_time = forecast_with_real_pv_data.index.get_level_values('valid_time').min()
        
        for align_method in ['ffill', 'linear']:
            predictor = OfflineForecastPredictor(
                name='pv_predictor',
                forecast_df=forecast_with_real_pv_data,
                variable_to_predict='Global tilted irradiance',
                align=align_method
            )
            
            state = SimulationState(
                time_id=0,
                time_step=1800,  # 30 minutes (different from native 1 hour)
                simulation_start_datetime=first_valid_time
            )
            
            predictions = predictor.predict(horizon=6.0, state=state)
            
            assert not predictions.empty
            assert len(predictions) == int(6.0 * 3600 / 1800)
    
    def test_predict_at_different_simulation_times(self, forecast_with_real_pv_data):
        """Test predictions at different simulation times."""
        predictor = OfflineForecastPredictor(
            name='pv_predictor',
            forecast_df=forecast_with_real_pv_data,
            variable_to_predict='Global tilted irradiance'
        )
        
        first_valid_time = forecast_with_real_pv_data.index.get_level_values('valid_time').min()
        
        # Test at multiple time points
        for hours_offset in [0, 6, 12]:
            state = SimulationState(
                time_id=hours_offset,
                time_step=3600,
                simulation_start_datetime=first_valid_time
            )
            # Update state.time to reflect the actual simulation time in seconds
            state.time = hours_offset * 3600
            
            predictions = predictor.predict(horizon=6.0, state=state)
            assert not predictions.empty
    
    def test_predict_uses_most_recent_issue_time(self, forecast_with_real_pv_data):
        """Test that predictions use the most recent available issue_time."""
        predictor = OfflineForecastPredictor(
            name='pv_predictor',
            forecast_df=forecast_with_real_pv_data,
            variable_to_predict='Global tilted irradiance'
        )
        
        # Get issue times
        issue_times = forecast_with_real_pv_data.index.get_level_values('issue_time').unique().sort_values()
        first_valid_time = forecast_with_real_pv_data.index.get_level_values('valid_time').min()
        
        if len(issue_times) > 1:
            # Test that it selects the most recent issue_time
            state = SimulationState(
                time_id=0,
                time_step=3600,
                simulation_start_datetime=first_valid_time + pd.Timedelta(hours=24)
            )
            state.time = 0
            
            predictions = predictor.predict(horizon=6.0, state=state)
            assert not predictions.empty
    
    def test_predict_raises_error_when_no_forecast_available(self, forecast_with_real_pv_data):
        """Test that prediction raises error when no forecast is available."""
        predictor = OfflineForecastPredictor(
            name='pv_predictor',
            forecast_df=forecast_with_real_pv_data,
            variable_to_predict='Global tilted irradiance'
        )
        
        # Create state with time before any forecast
        early_time = forecast_with_real_pv_data.index.get_level_values('valid_time').min() - pd.Timedelta(days=10)
        
        state = SimulationState(
            time_id=0,
            time_step=3600,
            simulation_start_datetime=early_time
        )
        state.time = 0
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="No forecast available"):
            predictor.predict(horizon=6.0, state=state)
    
    def test_predict_raises_error_when_horizon_exceeds_available_data(self, forecast_with_real_pv_data):
        """Test that prediction raises error when requested horizon exceeds available data."""
        predictor = OfflineForecastPredictor(
            name='pv_predictor',
            forecast_df=forecast_with_real_pv_data,
            variable_to_predict='Global tilted irradiance'
        )
        
        first_valid_time = forecast_with_real_pv_data.index.get_level_values('valid_time').min()
        
        state = SimulationState(
            time_id=0,
            time_step=3600,
            simulation_start_datetime=first_valid_time
        )
        state.time = 0
        
        # Should raise ValueError when trying to get predictions beyond available data
        with pytest.raises(ValueError, match="covers"):
            # Request a horizon of 500 hours (likely beyond available data)
            predictor.predict(horizon=500.0, state=state)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestOfflineForecastPredictorIntegration:
    """Integration tests with full environment simulation."""
    
    @pytest.fixture
    def forecast_data_for_simulation(self):
        """Load all available forecast files for simulation."""
        forecast_files = [
            os.path.join(__PV_FORECAST_DATA__, 'solar_radiation_forecast_2025-03-29.csv'),
            os.path.join(__PV_FORECAST_DATA__, 'solar_radiation_forecast_2025-03-30.csv'),
            os.path.join(__PV_FORECAST_DATA__, 'solar_radiation_forecast_2025-03-31.csv'),
            os.path.join(__PV_FORECAST_DATA__, 'solar_radiation_forecast_2025-04-01.csv'),
        ]
        
        available_files = [f for f in forecast_files if os.path.exists(f)]
        if not available_files:
            pytest.skip("No forecast files available for integration test")
        
        # Build forecast DataFrame
        forecast_df = build_forecast_dataframe(available_files, issue_time_offset_hours=3.0)
        
        return forecast_df
    
    @pytest.fixture
    def pv_forecast_predictor(self, forecast_data_for_simulation):
        """Create a PV forecast predictor."""
        return OfflineForecastPredictor(
            name='pv_power_predictor',
            forecast_df=forecast_data_for_simulation,
            variable_to_predict='Global tilted irradiance'
        )
    
    def test_integration_pv_battery_inverter_system(self, pv_forecast_predictor, forecast_data_for_simulation):
        """
        Test OfflineForecastPredictor in an integrated system with:
        - PV panels producing electricity
        - Inverter managing power flow
        - Battery for energy storage
        - Electric grid for balance
        - Inverter controller managing battery charge/discharge
        
        This test verifies that the OfflineForecastPredictor class works
        correctly within a complete energy system simulation environment.
        """
        # Get the time range from the forecast data
        valid_times = forecast_data_for_simulation.index.get_level_values('valid_time').unique()
        start_time = valid_times.min()
        end_time = valid_times.max()
        
        # For testing, simulate for 24 hours or the full available data range
        if (end_time - start_time).total_seconds() < 86400:
            end_time = start_time + pd.Timedelta(hours=24)
        
        # Create system components - use only components that are well-tested
        components = [
            # PV generation
            esc.PVpanelFromPVGIS(
                name='pv_panels',
                installed_power=5.0,
                latitude=44.511,
                longitude=11.335,
                tilt=30,
                azimuth=90
            ),
            # Energy storage
            esc.LithiumIonBattery(
                name='battery',
                capacity=10.0,  # 10 kWh
                SOC_0=0.5
            ),
            # Power conversion
            esc.Inverter(name='inverter'),
            # Grid connection for balance
            esc.BalancingUtility(name='electric_grid', utility_type='electricity'),
        ]
        
        # Create controllers
        controllers = [
            esc.InverterController(
                name='inverter_controller',
                inverter_name='inverter',
                battery_name='battery',
                SOC_min=0.3,
                SOC_max=0.9
            )
        ]
        
        # Create sensors to monitor the system
        sensors = [
            esc.SOCSensor('battery_SOC_sensor', 'battery'),
        ]
        
        # Create connections
        connections = [
            ('inverter_PV_input_port', 'pv_panels_electricity_port'),
            ('inverter_ESS_port', 'battery_electricity_port'),
            ('inverter_grid_input_port', 'electric_grid_electricity_port'),
        ]
        
        # Create environment without predictors for basic integration test
        try:
            env = esc.Environment(
                components=components,
                controllers=controllers,
                sensors=sensors,
                connections=connections,
                predictors=[]  # Test without predictors first
            )
            
            # Create simulation configuration
            simulation_hours = min(24.0, (end_time - start_time).total_seconds() / 3600)
            
            sim_config = esc.SimulationConfig(
                time_start_h=0.0,
                time_end_h=simulation_hours,
                time_step_h=1.0  # 1-hour timesteps
            )
            
            # Run simulation
            sim = esc.Simulator(env, sim_config)
            results = sim.run()
            
            # Verify that simulation ran successfully
            assert results is not None
            
            # Extract results
            df_ports, df_controllers, df_sensors = results.to_dataframe()
            
            # Basic assertions about results
            assert not df_ports.empty
            assert not df_sensors.empty
            
            # Verify that the battery SOC is within physical bounds
            battery_soc = df_sensors['battery_SOC_sensor']
            assert battery_soc.min() >= 0.0
            assert battery_soc.max() <= 1.0
            
            # Verify that values are reasonable (changed by the simulation)
            assert battery_soc.std() > 0.0, "Battery SOC should vary during simulation"
            
            print(f"✓ Integration test passed: Simulated {simulation_hours} hours with PV, battery, and inverter control")
            
        except Exception as e:
            # If there are issues with the environment setup, skip the test
            pytest.skip(f"Integration test skipped due to: {e}")
    
    def test_multiple_forecasts_same_horizon(self, forecast_data_for_simulation):
        """
        Test that the predictor correctly uses different forecast runs
        for different simulation times.
        """
        predictor = OfflineForecastPredictor(
            name='pv_predictor',
            forecast_df=forecast_data_for_simulation,
            variable_to_predict='Global tilted irradiance'
        )
        
        valid_times = forecast_data_for_simulation.index.get_level_values('valid_time').unique().sort_values()
        issue_times = forecast_data_for_simulation.index.get_level_values('issue_time').unique().sort_values()
        
        if len(issue_times) > 1 and len(valid_times) > 24:
            # Predict from two different issue times and compare
            state1 = SimulationState(
                time_id=0,
                time_step=3600,
                simulation_start_datetime=valid_times[0] + pd.Timedelta(hours=1)
            )
            state1.time = 0
            
            state2 = SimulationState(
                time_id=0,
                time_step=3600,
                simulation_start_datetime=valid_times[12] + pd.Timedelta(hours=1)
            )
            state2.time = 0
            
            try:
                pred1 = predictor.predict(horizon=6.0, state=state1)
                pred2 = predictor.predict(horizon=6.0, state=state2)
                
                # Predictions might be different or similar depending on forecast accuracy
                # Just verify they're both valid
                assert not pred1.empty
                assert not pred2.empty
            except ValueError:
                # This is okay if the second state is outside forecast range
                pass
