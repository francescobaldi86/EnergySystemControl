from abc import ABC, abstractmethod
from dataclasses import dataclass
from energy_system_control.helpers import C2K
import requests

@dataclass
class WeatherAPI(ABC):
    latitude: float
    longitude: float

    @abstractmethod
    def get_current_weather(self) -> dict:
        """Fetch weather data for the specified location."""
        pass

    @abstractmethod
    def get_weather_forecast(self, forecast_time) -> dict:
        """Fetch weather forecast for a specific time."""
        pass    


class OpenMeteoAPI(WeatherAPI):
    base_url = "https://api.open-meteo.com/v1/"
    def get_current_weather(self) -> dict:
        url = f'{self.base_url}forecast?latitude={self.latitude}&longitude={self.longitude}&current=temperature_2m,direct_radiation,diffuse_radiation,global_tilted_irradiance&timezone=GMT'
        data = self._send_request(url)
        return {
            "temperature": C2K(data['current']['temperature_2m']),
            "dni": data['current']['direct_radiation'],
            "dhi": data['current']['diffuse_radiation'],
            "ghi": data['current']['global_tilted_irradiance']
        }
    
    def get_weather_forecast(self, forecast_h: int) -> dict:
        """Fetch weather forecast for a specific time. This method can be implemented similarly to get_current_weather, but with additional parameters for time filtering."""
        url = f'{self.base_url}forecast?latitude={self.latitude}&longitude={self.longitude}&hourly=temperature_2m,direct_radiation,diffuse_radiation,global_tilted_irradiance&timezone=GMT&forecast_hours={forecast_h}'
        data = self._send_request(url)
        return {
            "temperature": C2K(data['hourly']['temperature_2m']),
            "dni": data['hourly']['direct_radiation'],
            "dhi": data['hourly']['diffuse_radiation'],
            "ghi": data['hourly']['global_tilted_irradiance']
        }

    def _send_request(self, url) -> dict:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()