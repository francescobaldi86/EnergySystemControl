import numpy as np
from dataclasses import dataclass
from energy_system_control.core.base_classes import Sensor

def calculate_prediction_metrics(actual, predicted):
    """
    Calculate prediction quality metrics.
    
    Parameters
    ----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
        
    Returns
    -------
    dict
        Dictionary containing RMSE (Root Mean Squared Error), MAE (Mean Absolute Error),
        and R² score (coefficient of determination).
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

class MockSensor(Sensor):
    def __init__(self, name, measurements):
        super().__init__(name)
        self.measurements = measurements
        self.index = 0

    def measure(self, environment=None, state=None):
        if self.index < len(self.measurements):
            self.current_measurement = self.measurements[self.index]
            self.index += 1
        return self.current_measurement

@dataclass
class MockEnvironment():
    sensors: dict[str, Sensor]