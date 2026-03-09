"""
Unit tests for environmental data providers.

Tests the functionality of CSVEnvironmentalProvider and APIEnvironmentalProvider,
including data loading, resampling, retrieval, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from energy_system_control.io.data_provider import (
    EnvironmentalDataProvider,
    CSVEnvironmentalProvider,
    APIEnvironmentalProvider,
)
from energy_system_control.sim.config import SimulationConfig
from energy_system_control.core.base_classes import EnvironmentalData


class TestEnvironmentalDataProvider:
    """Tests for the abstract EnvironmentalDataProvider base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Verify that EnvironmentalDataProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EnvironmentalDataProvider()

    def test_subclass_must_implement_get_environmental_data(self):
        """Verify that subclasses must implement get_environmental_data method."""
        
        class IncompleteProvider(EnvironmentalDataProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()


class TestCSVEnvironmentalProvider:
    """Tests for the CSVEnvironmentalProvider class."""

    @pytest.fixture
    def sample_csv_temp_file(self):
        """Create a temporary CSV file with sample environmental data."""
        # Create sample data for 2 days with 1-hour intervals
        dates = pd.date_range("2024-01-01", periods=48, freq="h")
        data = {
            "datetime": dates,
            "temperature_ambient": np.linspace(280, 300, 48) + np.random.randn(48) * 0.5,
            "temperature_cold_water": 288 + np.random.randn(48) * 0.2,
            "direct_irradiation": np.abs(np.sin(np.linspace(0, 2 * np.pi * 2, 48)) * 1000),
            "diffuse_irradiation": 100 + 50 * np.abs(np.sin(np.linspace(0, 2 * np.pi * 2, 48))),
        }
        df = pd.DataFrame(data)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name
            df.to_csv(f, index=False)

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    @pytest.fixture
    def simulation_config(self):
        """Create a basic simulation configuration."""
        return SimulationConfig(
            time_step_h=0.08333,  # ~5 minutes in hours (300 seconds)
            time_start_h=0,
            time_end_h=24,
        )

    def test_initialization(self, sample_csv_temp_file):
        """Test CSVEnvironmentalProvider initialization."""
        provider = CSVEnvironmentalProvider(sample_csv_temp_file)
        
        assert provider.csv_path == sample_csv_temp_file
        assert provider.var_types == {}
        assert provider.data == {}
        assert provider.datetime_index is None

    def test_initialization_with_var_types(self, sample_csv_temp_file):
        """Test CSVEnvironmentalProvider initialization with variable types."""
        var_types = {
            "temperature_ambient": "intensive",
            "direct_irradiation": "extensive",
        }
        provider = CSVEnvironmentalProvider(sample_csv_temp_file, var_types=var_types)
        
        assert provider.var_types == var_types

    def test_initialize_loads_data(self, sample_csv_temp_file, simulation_config):
        """Test that initialize method loads and resamples CSV data."""
        provider = CSVEnvironmentalProvider(sample_csv_temp_file)
        provider.initialize(simulation_config)

        # Check that data has been loaded
        assert len(provider.data) > 0
        assert "temperature_ambient" in provider.data
        assert "direct_irradiation" in provider.data
        assert "diffuse_irradiation" in provider.data

        # Check that all arrays have reasonable length (approximately 24h / 0.08333h = ~288 timesteps)
        # Allow some tolerance due to resampling behavior
        expected_length = int((simulation_config.time_end_h - simulation_config.time_start_h) / simulation_config.time_step_h)
        for key, array in provider.data.items():
            # Allow ±15 timesteps tolerance due to resampling
            assert abs(len(array) - expected_length) <= 15

    def test_get_environmental_data(self, sample_csv_temp_file, simulation_config):
        """Test retrieving environmental data."""
        provider = CSVEnvironmentalProvider(sample_csv_temp_file)
        provider.initialize(simulation_config)

        current_time = pd.Timestamp("2024-01-01 00:00:00")
        env_data = provider.get_environmental_data(time_id=0, current_time=current_time)

        assert isinstance(env_data, EnvironmentalData)
        assert env_data.temperature_ambient is not None
        assert env_data.direct_irradiation is not None
        assert env_data.diffuse_irradiation is not None

    def test_get_environmental_data_different_timesteps(
        self, sample_csv_temp_file, simulation_config
    ):
        """Test that different timesteps can be retrieved successfully."""
        provider = CSVEnvironmentalProvider(sample_csv_temp_file)
        provider.initialize(simulation_config)

        current_time = pd.Timestamp("2024-01-01 00:00:00")
        
        env_data_0 = provider.get_environmental_data(time_id=0, current_time=current_time)
        env_data_100 = provider.get_environmental_data(
            time_id=100, 
            current_time=current_time + pd.Timedelta(hours=8.33)
        )

        # Verify we can retrieve data at different timesteps
        assert isinstance(env_data_0, EnvironmentalData)
        assert isinstance(env_data_100, EnvironmentalData)
        assert env_data_0.temperature_ambient is not None
        assert env_data_100.temperature_ambient is not None

    def test_get_environmental_data_out_of_range(
        self, sample_csv_temp_file, simulation_config
    ):
        """Test behavior when requesting data beyond available range."""
        provider = CSVEnvironmentalProvider(sample_csv_temp_file)
        provider.initialize(simulation_config)

        current_time = pd.Timestamp("2024-01-02 00:00:00")
        
        # Should raise IndexError for out-of-range time_id
        with pytest.raises(IndexError):
            provider.get_environmental_data(time_id=10000, current_time=current_time)

    def test_missing_csv_file(self):
        """Test that missing CSV file raises appropriate error."""
        provider = CSVEnvironmentalProvider("/nonexistent/path/data.csv")
        cfg = SimulationConfig(time_step_h=0.08333, time_start_h=0, time_end_h=24)
        
        with pytest.raises(FileNotFoundError):
            provider.initialize(cfg)

    def test_var_types_intensive(self, sample_csv_temp_file, simulation_config):
        """Test that intensive variable types are correctly handled."""
        var_types = {
            "temperature_ambient": "intensive",
            "direct_irradiation": "intensive",
            "diffuse_irradiation": "intensive",
            "temperature_cold_water": "intensive",
        }
        provider = CSVEnvironmentalProvider(sample_csv_temp_file, var_types=var_types)
        provider.initialize(simulation_config)

        # Should initialize successfully with intensive types
        assert len(provider.data) > 0

    def test_var_types_extensive(self, sample_csv_temp_file, simulation_config):
        """Test that extensive variable types are correctly handled."""
        var_types = {
            "direct_irradiation": "extensive",
            "diffuse_irradiation": "extensive",
            "temperature_ambient": "intensive",
            "temperature_cold_water": "intensive",
        }
        provider = CSVEnvironmentalProvider(sample_csv_temp_file, var_types=var_types)
        provider.initialize(simulation_config)

        # Should initialize successfully with mixed types
        assert len(provider.data) > 0

    def test_environmental_data_values_reasonable(
        self, sample_csv_temp_file, simulation_config
    ):
        """Test that retrieved data contains reasonable physical values."""
        provider = CSVEnvironmentalProvider(sample_csv_temp_file)
        provider.initialize(simulation_config)

        # Sample several timesteps
        for time_id in [0, 50, 100, 150, 288]:
            if time_id < len(provider.data["temperature_ambient"]):
                current_time = pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=time_id * 300)
                env_data = provider.get_environmental_data(time_id=time_id, current_time=current_time)

                # Reasonable temperature ranges (in Kelvin)
                assert 250 < env_data.temperature_ambient < 320
                # Irradiation should be non-negative
                assert env_data.direct_irradiation >= 0
                assert env_data.diffuse_irradiation >= 0


class TestAPIEnvironmentalProvider:
    """Tests for the APIEnvironmentalProvider class."""

    def test_initialization(self):
        """Test APIEnvironmentalProvider initialization."""
        mock_api = Mock()
        latitude = 45.5
        longitude = 9.2
        
        provider = APIEnvironmentalProvider(latitude, longitude, mock_api)

        assert provider.latitude == latitude
        assert provider.longitude == longitude
        assert provider.api_client is mock_api

    def test_get_environmental_data_successful_call(self):
        """Test successful API call and data retrieval."""
        mock_api = Mock()
        mock_api.get_current_weather.return_value = {
            "temperature": 288.15,
            "dni": 500.0,
            "dhi": 150.0,
        }
        
        provider = APIEnvironmentalProvider(latitude=45.5, longitude=9.2, api_client=mock_api)
        current_time = pd.Timestamp("2024-01-01 12:00:00")
        
        env_data = provider.get_environmental_data(time_id=0, current_time=current_time)

        assert isinstance(env_data, EnvironmentalData)
        assert env_data.temperature_ambient == 288.15
        assert env_data.direct_irradiation == 500.0
        assert env_data.diffuse_irradiation == 150.0
        
        # Verify the API was called with correct parameters
        mock_api.get_current_weather.assert_called_once()
        call_kwargs = mock_api.get_current_weather.call_args[1]
        assert call_kwargs["lat"] == 45.5
        assert call_kwargs["lon"] == 9.2
        assert call_kwargs["time"] == current_time

    def test_api_called_with_correct_time(self):
        """Test that the API is called with the correct current time."""
        mock_api = Mock()
        mock_api.get_current_weather.return_value = {
            "temperature": 290,
            "dni": 600.0,
            "dhi": 120.0,
        }
        
        provider = APIEnvironmentalProvider(latitude=40.0, longitude=10.0, api_client=mock_api)
        
        # Test multiple different times
        times = [
            pd.Timestamp("2024-01-01 00:00:00"),
            pd.Timestamp("2024-01-01 12:00:00"),
            pd.Timestamp("2024-01-02 06:30:00"),
        ]
        
        for current_time in times:
            provider.get_environmental_data(time_id=0, current_time=current_time)
        
        # Verify each call had the correct time
        assert mock_api.get_current_weather.call_count == 3
        for i, current_time in enumerate(times):
            call_kwargs = mock_api.get_current_weather.call_args_list[i][1]
            assert call_kwargs["time"] == current_time

    def test_api_error_handling(self):
        """Test that API errors are propagated."""
        mock_api = Mock()
        mock_api.get_current_weather.side_effect = ConnectionError("API unavailable")
        
        provider = APIEnvironmentalProvider(latitude=45.5, longitude=9.2, api_client=mock_api)
        
        with pytest.raises(ConnectionError):
            provider.get_environmental_data(time_id=0, current_time=pd.Timestamp("2024-01-01"))

    def test_api_missing_response_fields(self):
        """Test handling of incomplete API responses."""
        mock_api = Mock()
        mock_api.get_current_weather.return_value = {
            "temperature": 288.15,
            # Missing "ni" and "dhi"
        }
        
        provider = APIEnvironmentalProvider(latitude=45.5, longitude=9.2, api_client=mock_api)
        
        with pytest.raises(KeyError):
            provider.get_environmental_data(time_id=0, current_time=pd.Timestamp("2024-01-01"))

    def test_multiple_sequential_calls(self):
        """Test multiple sequential API calls."""
        mock_api = Mock()
        mock_api.get_current_weather.return_value = {
            "temperature": 285.0,
            "dni": 400.0,
            "dhi": 100.0,
        }
        
        provider = APIEnvironmentalProvider(latitude=45.5, longitude=9.2, api_client=mock_api)
        
        # Make multiple calls
        for i in range(5):
            current_time = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)
            env_data = provider.get_environmental_data(time_id=i, current_time=current_time)
            
            assert env_data.temperature_ambient == 285.0
            assert env_data.direct_irradiation == 400.0
        
        assert mock_api.get_current_weather.call_count == 5


class TestProviderComparison:
    """Tests comparing behavior of different provider types."""

    @pytest.fixture
    def csv_provider(self):
        """Create a CSV provider with sample data."""
        dates = pd.date_range("2024-01-01", periods=48, freq="h")
        data = {
            "datetime": dates,
            "temperature_ambient": 293.15 * np.ones(48),  # Constant 293.15 K
            "direct_irradiation": 500.0 * np.ones(48),     # Constant 500 W/m²
            "diffuse_irradiation": 100.0 * np.ones(48),    # Constant 100 W/m²
            "temperature_cold_water": 288.15 * np.ones(48),# Constant 288.15 K
        }
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name
            df.to_csv(f, index=False)

        provider = CSVEnvironmentalProvider(temp_path)
        cfg = SimulationConfig(time_step_h=300/3600, time_start_h=0, time_end_h=24)
        provider.initialize(cfg)
        
        yield provider

        if os.path.exists(temp_path):
            os.remove(temp_path)

    @pytest.fixture
    def api_provider(self):
        """Create an API provider with mock client."""
        mock_api = Mock()
        mock_api.get_current_weather.return_value = {
            "temperature": 293.15,
            "dni": 500.0,
            "dhi": 100.0,
        }
        return APIEnvironmentalProvider(latitude=45.5, longitude=9.2, api_client=mock_api)

    def test_both_providers_return_valid_data(self, csv_provider, api_provider):
        """Test that both provider types return valid EnvironmentalData objects."""
        current_time = pd.Timestamp("2024-01-01 12:00:00")
        
        csv_data = csv_provider.get_environmental_data(time_id=0, current_time=current_time)
        api_data = api_provider.get_environmental_data(time_id=0, current_time=current_time)

        assert isinstance(csv_data, EnvironmentalData)
        assert isinstance(api_data, EnvironmentalData)

    def test_both_providers_inherit_interface(self, csv_provider, api_provider):
        """Test that both providers properly inherit from base class."""
        assert isinstance(csv_provider, EnvironmentalDataProvider)
        assert isinstance(api_provider, EnvironmentalDataProvider)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
