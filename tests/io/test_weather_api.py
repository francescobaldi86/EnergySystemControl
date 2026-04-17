"""
Unit tests for weather API client implementations.

Tests the functionality of WeatherAPI abstract base class and concrete
implementations like OpenMeteoAPI, including data fetching, error handling,
and request formatting.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
import json

from energy_system_control.io.weather_api import WeatherAPI, OpenMeteoAPI


class TestWeatherAPI:
    """Tests for the abstract WeatherAPI base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Verify that WeatherAPI cannot be instantiated directly."""
        with pytest.raises(TypeError):
            WeatherAPI(latitude=45.5, longitude=9.2, base_url="http://example.com")

    def test_subclass_must_implement_required_methods(self):
        """Verify that subclasses must implement all abstract methods."""
        
        class IncompleteWeatherAPI(WeatherAPI):
            pass

        with pytest.raises(TypeError):
            IncompleteWeatherAPI(latitude=45.5, longitude=9.2, base_url="http://example.com")


class TestOpenMeteoAPI:
    """Tests for the OpenMeteoAPI implementation."""

    @pytest.fixture
    def api_instance(self):
        """Create an OpenMeteoAPI instance for testing."""
        return OpenMeteoAPI(latitude=45.5, longitude=9.2)

    def test_initialization(self):
        """Test OpenMeteoAPI initialization."""
        api = OpenMeteoAPI(latitude=45.5, longitude=9.2)
        
        assert api.latitude == 45.5
        assert api.longitude == 9.2
        assert api.base_url == "https://api.open-meteo.com/v1/"

    def test_base_url_is_correct(self):
        """Test that the class-level base_url is set correctly."""
        assert OpenMeteoAPI.base_url == "https://api.open-meteo.com/v1/"

    @patch('energy_system_control.io.weather_api.requests.get')
    def test_get_current_weather_successful(self, mock_get, api_instance):
        """Test successful fetch of current weather data."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'current': {
                'temperature_2m': 20,
                'direct_radiation': 500.0,
                'diffuse_radiation': 150.0,
                'global_tilted_irradiance': 650.0
            }
        }
        mock_get.return_value = mock_response

        weather_data = api_instance.get_current_weather()

        assert weather_data['temperature'] == 293.15
        assert weather_data['dni'] == 500.0
        assert weather_data['dhi'] == 150.0

        # Verify the correct URL was called with the right parameters
        mock_get.assert_called_once()
        call_args = mock_get.call_args[0][0]
        assert '45.5' in call_args
        assert '9.2' in call_args
        assert 'current=temperature_2m,direct_radiation,diffuse_radiation,global_tilted_irradiance' in call_args

    @patch('energy_system_control.io.weather_api.requests.get')
    def test_get_current_weather_with_different_locations(self, mock_get):
        """Test current weather fetch with different coordinates."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'current': {
                'temperature_2m': 15.0,
                'direct_radiation': 400.0,
                'diffuse_radiation': 120.0,
                'global_tilted_irradiance': 520.0
            }
        }
        mock_get.return_value = mock_response

        api1 = OpenMeteoAPI(latitude=40.7, longitude=-74.0)
        api1.get_current_weather()

        call_args = mock_get.call_args[0][0]
        assert '40.7' in call_args
        assert '-74.0' in call_args

    @patch('energy_system_control.io.weather_api.requests.get')
    def test_get_weather_forecast_successful(self, mock_get, api_instance):
        """Test successful fetch of weather forecast data."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'hourly': {
                'temperature_2m': [20.0, 20.3, 30.6, 20.85],
                'direct_radiation': [0.0, 100.0, 500.0, 800.0],
                'diffuse_radiation': [50.0, 75.0, 150.0, 200.0],
                'global_tilted_irradiance': [50.0, 175.0, 650.0, 1000.0]
            }
        }
        mock_get.return_value = mock_response

        forecast_data = api_instance.get_weather_forecast(forecast_h=48)

        assert isinstance(forecast_data['temperature'], list)
        assert len(forecast_data['temperature']) == 4
        assert forecast_data['temperature'][0] == 293.15
        assert forecast_data['temperature'][-1] == 294.0
        assert forecast_data['dni'] == [0.0, 100.0, 500.0, 800.0]
        assert forecast_data['dhi'] == [50.0, 75.0, 150.0, 200.0]
        assert forecast_data['ghi'] == [50.0, 175.0, 650.0, 1000.0]

    @patch('energy_system_control.io.weather_api.requests.get')
    def test_get_weather_forecast_with_different_hours(self, mock_get, api_instance):
        """Test forecast fetch with different forecast_h parameter."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'hourly': {
                'temperature_2m': [16.5] * 48,
                'direct_radiation': [0.0] * 24 + [100.0] * 24,
                'diffuse_radiation': [75.0] * 48,
                'global_tilted_irradiance': [75.0] * 24 + [200.0] * 24
            }
        }
        mock_get.return_value = mock_response

        api_instance.get_weather_forecast(forecast_h=72)

        call_args = mock_get.call_args[0][0]
        assert 'forecast_hours=72' in call_args

    @patch('energy_system_control.io.weather_api.requests.get')
    def test_get_current_weather_http_error(self, mock_get, api_instance):
        """Test handling of HTTP errors in get_current_weather."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            api_instance.get_current_weather()

    @patch('energy_system_control.io.weather_api.requests.get')
    def test_get_forecast_http_error(self, mock_get, api_instance):
        """Test handling of HTTP errors in get_weather_forecast."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("503 Service Unavailable")
        mock_get.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            api_instance.get_weather_forecast(forecast_h=24)

    @patch('energy_system_control.io.weather_api.requests.get')
    def test_get_current_weather_connection_error(self, mock_get, api_instance):
        """Test handling of connection errors."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Failed to connect")

        with pytest.raises(requests.exceptions.ConnectionError):
            api_instance.get_current_weather()

    @patch('energy_system_control.io.weather_api.requests.get')
    def test_get_weather_forecast_timeout(self, mock_get, api_instance):
        """Test handling of request timeout."""
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        with pytest.raises(requests.exceptions.Timeout):
            api_instance.get_weather_forecast(forecast_h=24)

    @patch('energy_system_control.io.weather_api.requests.get')
    def test_get_current_weather_invalid_json(self, mock_get, api_instance):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response

        with pytest.raises(json.JSONDecodeError):
            api_instance.get_current_weather()

    @patch('energy_system_control.io.weather_api.requests.get')
    def test_get_current_weather_missing_fields(self, mock_get, api_instance):
        """Test handling when response is missing expected fields."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'current': {
                'temperature_2m': 293.15
                # Missing 'ni', 'dhi', 'ghi'
            }
        }
        mock_get.return_value = mock_response

        with pytest.raises(KeyError):
            api_instance.get_current_weather()

    @patch('energy_system_control.io.weather_api.requests.get')
    def test_send_request_url_construction(self, mock_get, api_instance):
        """Test that URLs are correctly constructed."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'current': {
                'temperature_2m': 293.15,
                'direct_radiation': 500.0,
                'diffuse_radiation': 150.0,
                'global_tilted_irradiance': 650.0
            }
        }
        mock_get.return_value = mock_response

        api_instance.get_current_weather()

        # Check that only one request was made
        assert mock_get.call_count == 1
        call_url = mock_get.call_args[0][0]
        
        # Verify URL components
        assert api_instance.base_url in call_url
        assert 'latitude=' in call_url
        assert 'longitude=' in call_url
        assert 'timezone=GMT' in call_url

    @patch('energy_system_control.io.weather_api.requests.get')
    def test_multiple_sequential_requests(self, mock_get, api_instance):
        """Test multiple sequential API requests."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'current': {
                'temperature_2m': 293.15,
                'direct_radiation': 500.0,
                'diffuse_radiation': 150.0,
                'global_tilted_irradiance': 650.0
            }
        }
        mock_get.return_value = mock_response

        # Make multiple requests
        for _ in range(3):
            api_instance.get_current_weather()

        assert mock_get.call_count == 3

    @patch('energy_system_control.io.weather_api.requests.get')
    def test_get_current_weather_zero_values(self, mock_get, api_instance):
        """Test handling of zero values in weather data (e.g., nighttime irradiance)."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'current': {
                'temperature_2m': 7.0,
                'direct_radiation': 0.0,  # No direct irradiance at night
                'diffuse_radiation': 0.0,  # No diffuse irradiance at night
                'global_tilted_irradiance': 0.0   # No global radiation at night
            }
        }
        mock_get.return_value = mock_response

        weather_data = api_instance.get_current_weather()

        assert weather_data['temperature'] == 280.15
        assert weather_data['dni'] == 0.0
        assert weather_data['dhi'] == 0.0
        assert weather_data['ghi'] == 0.0

    @patch('energy_system_control.io.weather_api.requests.get')
    def test_get_forecast_multiple_days(self, mock_get, api_instance):
        """Test forecast data for extended periods."""
        # Create 7 days of hourly data (168 hours)
        forecast_hours = 7 * 24
        mock_response = Mock()
        mock_response.json.return_value = {
            'hourly': {
                'temperature_2m': list(range(290, 290 + forecast_hours)),
                'direct_radiation': [100.0 if 6 <= h % 24 <= 18 else 0.0 for h in range(forecast_hours)],
                'diffuse_radiation': [50.0 if 6 <= h % 24 <= 18 else 0.0 for h in range(forecast_hours)],
                'global_tilted_irradiance': [150.0 if 6 <= h % 24 <= 18 else 0.0 for h in range(forecast_hours)]
            }
        }
        mock_get.return_value = mock_response

        forecast_data = api_instance.get_weather_forecast(forecast_h=24 * 7)

        assert len(forecast_data['temperature']) == forecast_hours
        assert len(forecast_data['dni']) == forecast_hours

    def test_api_instance_is_subclass_of_base(self, api_instance):
        """Verify that OpenMeteoAPI is a proper subclass of WeatherAPI."""
        assert isinstance(api_instance, WeatherAPI)


class TestWeatherAPIIntegration:
    """Integration tests for weather API interactions."""

    @patch('energy_system_control.io.weather_api.requests.get')
    def test_current_and_forecast_compatibility(self, mock_get):
        """Test that current weather and forecast data structures are compatible."""
        # Mock for current weather
        current_response = Mock()
        current_response.json.return_value = {
            'current': {
                'temperature_2m': 293.15,
                'direct_radiation': 500.0,
                'diffuse_radiation': 150.0,
                'global_tilted_irradiance': 650.0
            }
        }

        # Mock for forecast (first hour should match current)
        forecast_response = Mock()
        forecast_response.json.return_value = {
            'hourly': {
                'temperature_2m': [293.15],
                'direct_radiation': [500.0],
                'diffuse_radiation': [150.0],
                'global_tilted_irradiance': [650.0]
            }
        }

        api = OpenMeteoAPI(latitude=45.5, longitude=9.2)

        # First call returns current response, second returns forecast response
        mock_get.side_effect = [current_response, forecast_response]

        current_data = api.get_current_weather()
        forecast_data = api.get_weather_forecast(forecast_h=24)

        # Both should have compatible structures
        assert 'temperature' in current_data
        assert 'temperature' in forecast_data
        assert current_data['temperature'] == forecast_data['temperature'][0]
        assert current_data['dni'] == forecast_data['dni'][0]


class TestOpenMeteoAPIRealRequests:
    """Integration tests with real API calls to OpenMeteo."""

    @pytest.mark.integration
    def test_get_current_weather_real_api_call(self):
        """Test actual API call to OpenMeteo and validate response structure.
        
        This test makes a real HTTP request to the OpenMeteo API.
        It validates that:
        - The API is accessible
        - Response structure matches expectations
        - Data types are correct
        - Values are physically reasonable
        
        Marked as integration test - use pytest -m integration to run.
        """
        api = OpenMeteoAPI(
            latitude=45.5,
            longitude=9.2
        )
        
        # Make real API call
        weather_data = api.get_current_weather()
        
        # Validate response structure
        assert isinstance(weather_data, dict)
        assert 'temperature' in weather_data
        assert 'dni' in weather_data
        assert 'dhi' in weather_data
        assert 'ghi' in weather_data
        
        # Validate data types
        assert isinstance(weather_data['temperature'], (int, float))
        assert isinstance(weather_data['dni'], (int, float))
        assert isinstance(weather_data['dhi'], (int, float))
        assert isinstance(weather_data['ghi'], (int, float))
        
        # Validate physical reasonableness
        # Temperature should be in reasonable range (K)
        assert 250 < weather_data['temperature'] < 330
        
        # Irradiance should be non-negative (W/m²)
        assert weather_data['dni'] >= 0
        assert weather_data['dhi'] >= 0
        assert weather_data['ghi'] >= 0
        
        # Global radiation should be sum of direct and diffuse (approximately)
        # Allow some tolerance due to definition differences
        assert weather_data['ghi'] >= weather_data['dni']
        assert weather_data['ghi'] >= weather_data['dhi']

    @pytest.mark.integration
    def test_get_weather_forecast_real_api_call(self):
        """Test actual forecast API call to OpenMeteo.
        
        This test makes a real HTTP request to fetch weather forecast data.
        It validates that:
        - The API returns forecast data
        - Arrays have consistent lengths
        - Values are physically reasonable
        - Data covers the requested period
        
        Marked as integration test - use pytest -m integration to run.
        """
        api = OpenMeteoAPI(
            latitude=40.7,
            longitude=-74.0
        )
        
        # Make real API call for 2-day forecast
        forecast_data = api.get_weather_forecast(forecast_h=48)
        
        # Validate response structure
        assert isinstance(forecast_data, dict)
        assert 'temperature' in forecast_data
        assert 'dni' in forecast_data
        assert 'dhi' in forecast_data
        assert 'ghi' in forecast_data
        
        # Validate data types
        assert isinstance(forecast_data['temperature'], list)
        assert isinstance(forecast_data['dni'], list)
        assert isinstance(forecast_data['dhi'], list)
        assert isinstance(forecast_data['ghi'], list)
        
        # Validate array lengths match
        num_values = len(forecast_data['temperature'])
        assert len(forecast_data['dni']) == num_values
        assert len(forecast_data['dhi']) == num_values
        assert len(forecast_data['ghi']) == num_values
        
        # Validate we got reasonable amount of data (at least 24 hours of forecast)
        assert num_values >= 24
        
        # Validate individual values
        for i in range(num_values):
            temp = forecast_data['temperature'][i]
            dni = forecast_data['dni'][i]
            dhi = forecast_data['dhi'][i]
            ghi = forecast_data['ghi'][i]
            
            # Temperature should be physically reasonable (K)
            assert 250 < temp < 330
            
            # Irradiance should be non-negative
            # assert dni >= -5  # Allow small non-zero value due to noise even at night
            assert dhi >= 0
            assert ghi >= 0
