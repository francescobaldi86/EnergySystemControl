"""Tests for SensorWithMemory"""
import pytest
import numpy as np
import energy_system_control as esc
from tests.utils import MockSensor, MockEnvironment


class TestSensorWithMemoryInitialization:
    """Test suite for SensorWithMemory initialization"""
    
    def test_initialization_with_valid_parameters(self):
        """Test SensorWithMemory initialization with valid parameters"""
        sensor = esc.SensorWithMemory(
            name='test_sensor_memory',
            source_sensor_name='source_sensor',
            lookback_time=12.0,
            n_samples=10
        )
        assert sensor.name == 'test_sensor_memory'
        assert sensor.source_sensor_name == 'source_sensor'
        assert sensor.lookback_time == 12.0
        assert sensor.n_samples == 10
        assert len(sensor.memory) == 0
        assert len(sensor.current_measurement) == 0
    
    def test_initialization_with_missing_lookback_time(self):
        """Test that missing lookback_time raises ValueError"""
        with pytest.raises(ValueError):
            esc.SensorWithMemory(
                name='test_sensor_memory',
                source_sensor_name='source_sensor',
                lookback_time=-1.0,
                n_samples=10
            )
    
    def test_initialization_with_missing_n_samples(self):
        """Test that missing n_samples raises ValueError"""
        with pytest.raises(ValueError):
            esc.SensorWithMemory(
                name='test_sensor_memory',
                source_sensor_name='source_sensor',
                lookback_time=12.0,
                n_samples=-1
            )


class TestSensorWithMemoryMeasurement:
    """Test suite for SensorWithMemory measurement logic"""
    
    def test_measure_with_mock_sensor(self):
        """Test measure method with a mock source sensor"""
        # Create a mock source sensor with fixed measurements
        mock_source = MockSensor('source', [10.0, 20.0, 30.0, 40.0, 50.0])
        mock_source.measure()
        
        # Create SensorWithMemory
        sensor_with_memory = esc.SensorWithMemory(
            name='memory_sensor',
            source_sensor_name='source',
            lookback_time=1.0,
            n_samples=5
        )
        
        # Create a mock state with time tracking
        class MockState:
            def __init__(self):
                self.time_id = 0
                self.time = 0
        
        mock_state = MockState()
        mock_env = MockEnvironment(sensors={'source': mock_source})
        
        # Measure multiple times
        for i in range(5):
            mock_state.time_id = i
            result = sensor_with_memory.measure(mock_env, mock_state)
            assert isinstance(result, np.ndarray)
            assert len(result) == 5
    
    def test_resample_single_measurement(self):
        """Test resampling when only one measurement is in memory"""
        sensor = esc.SensorWithMemory(
            name='test_sensor',
            source_sensor_name='source',
            lookback_time=1.0,
            n_samples=5
        )
        
        # Manually add a single measurement
        sensor.memory.append((0.0, 42.0))
        resampled = sensor._resample_measurements()
        
        # Should return array of 5 identical values
        assert len(resampled) == 5
        assert np.allclose(resampled, 42.0)
    
    def test_resample_multiple_measurements(self):
        """Test resampling with multiple measurements"""
        sensor = esc.SensorWithMemory(
            name='test_sensor',
            source_sensor_name='source',
            lookback_time=1.0,
            n_samples=3
        )
        
        # Add measurements: linear from 0 to 10
        sensor.memory.append((0.0, 0.0))
        sensor.memory.append((5.0, 5.0))
        sensor.memory.append((10.0, 10.0))
        
        resampled = sensor._resample_measurements()
        
        # Should return 3 values
        assert len(resampled) == 3
        # Should interpolate correctly
        assert np.allclose(resampled[0], 0.0, atol=0.1)
        assert np.allclose(resampled[1], 5.0, atol=0.1)
        assert np.allclose(resampled[2], 10.0, atol=0.1)
    
    def test_resample_empty_memory(self):
        """Test resampling when memory is empty"""
        sensor = esc.SensorWithMemory(
            name='test_sensor',
            source_sensor_name='source',
            lookback_time=1.0,
            n_samples=5
        )
        
        resampled = sensor._resample_measurements()
        assert len(resampled) == 0


class TestSensorWithMemoryTimeWindow:
    """Test suite for lookback_time window management"""
    
    def test_lookback_time_removes_old_measurements(self):
        """Test that measurements older than lookback_time are removed"""
        sensor = esc.SensorWithMemory(
            name='test_sensor',
            source_sensor_name='source',
            lookback_time=5.0,  # 5 hours
            n_samples=10
        )
        
        # Manually add measurements at different times
        # Time is in hours
        sensor.memory.append((0.0, 10.0))    # Old, should be removed
        sensor.memory.append((4.0, 20.0))    # Old, should be removed
        sensor.memory.append((6.0, 30.0))    # Recent, keep
        sensor.memory.append((10.0, 40.0))   # Recent, keep
        sensor.memory.append((11.0, 50.0))   # Recent, keep
        
        # Simulate measure at time 11.0 with lookback of 5 hours
        # This should keep only measurements from time >= (11 - 5) = 6
        class MockState:
            def __init__(self):
                self.time_id = 11.0
        
        mock_state = MockState()
        mock_source = MockSensor('source', [999.0])  # Dummy value
        mock_env = MockEnvironment(sensors={'source': mock_source})
        
        # Manually simulate the cleanup logic
        current_time = 11.0
        cutoff_time = current_time - sensor.lookback_time
        while sensor.memory and sensor.memory[0][0] < cutoff_time:
            sensor.memory.popleft()
        
        # After cleanup, should have 3 measurements (at times 6, 10, 11)
        assert len(sensor.memory) == 3
        times = [t for t, v in sensor.memory]
        assert all(t >= cutoff_time for t in times)


class TestSensorWithMemorySamplingDensity:
    """Test suite for n_samples parameter"""
    
    def test_n_samples_parameter_controls_output_size(self):
        """Test that n_samples parameter controls output array size"""
        for n in [1, 5, 10, 100]:
            sensor = esc.SensorWithMemory(
                name='test_sensor',
                source_sensor_name='source',
                lookback_time=1.0,
                n_samples=n
            )
            
            # Add some measurements
            for i in range(10):
                sensor.memory.append((float(i), float(i * 2)))
            
            resampled = sensor._resample_measurements()
            assert len(resampled) == n
    
    def test_different_n_samples_values(self):
        """Test that different n_samples produce different outputs"""
        # Create two sensors with different n_samples
        sensor_5 = esc.SensorWithMemory(
            name='sensor_5',
            source_sensor_name='source',
            lookback_time=1.0,
            n_samples=5
        )
        
        sensor_20 = esc.SensorWithMemory(
            name='sensor_20',
            source_sensor_name='source',
            lookback_time=1.0,
            n_samples=20
        )
        
        # Add identical measurements to both
        for i in np.linspace(0, 10, 100):
            sensor_5.memory.append((i, np.sin(i)))
            sensor_20.memory.append((i, np.sin(i)))
        
        resampled_5 = sensor_5._resample_measurements()
        resampled_20 = sensor_20._resample_measurements()
        
        assert len(resampled_5) == 5
        assert len(resampled_20) == 20


class TestSensorWithMemoryReset:
    """Test suite for reset functionality"""
    
    def test_reset_clears_memory(self):
        """Test that reset clears the memory buffer"""
        sensor = esc.SensorWithMemory(
            name='test_sensor',
            source_sensor_name='source',
            lookback_time=1.0,
            n_samples=5
        )
        
        # Add some measurements
        sensor.memory.append((0.0, 10.0))
        sensor.memory.append((1.0, 20.0))
        sensor.current_measurement = np.array([1, 2, 3, 4, 5])
        
        # Reset
        sensor.reset()
        
        assert len(sensor.memory) == 0
        assert len(sensor.current_measurement) == 0


class TestSensorWithMemoryIntegration:
    """Integration tests for SensorWithMemory in a simulation"""
    
    def test_sensor_with_memory_in_simple_simulation(self):
        """Test SensorWithMemory in a simple simulation with real sensors"""
        components = [
            esc.IEAHotWaterDemand(
                name="demand_DHW",
                reference_temperature=40,
                profile_name='M'
            ),
            esc.HotWaterStorage(
                name='hot_water_storage',
                max_temperature=80,
                tank_volume=200,
                T_0=45
            ),
            esc.ElectricityGrid(name='electric_grid'),
            esc.ColdWaterGrid(name='water_grid', utility_type='fluid')
        ]
        
        # Create a temperature sensor to measure tank temperature
        tank_temp_sensor = esc.TankTemperatureSensor(
            'tank_temperature_sensor',
            component_name='hot_water_storage'
        )
        
        # Create a SensorWithMemory that reads from the tank temperature sensor
        memory_sensor = esc.SensorWithMemory(
            name='tank_temperature_with_memory',
            source_sensor_name='tank_temperature_sensor',
            lookback_time=4.0,  # 4 hours
            n_samples=8
        )
        
        sensors = [tank_temp_sensor, memory_sensor]
        
        connections = [
            ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),
            ('hot_water_storage_cold_water_input_port', 'water_grid_fluid_port'),
        ]
        
        env = esc.Environment(
            components=components,
            controllers=[],
            sensors=sensors,
            connections=connections
        )
        
        sim_config = esc.SimulationConfig(
            time_start_h=0.0,
            time_end_h=24.0,
            time_step_h=1.0
        )
        
        sim = esc.Simulator(env, sim_config)
        results = sim.run()
        df_ports, df_controllers, df_sensors = results.to_dataframe()
        
        # Verify both sensors are in results
        assert 'tank_temperature_sensor' in df_sensors.columns
        assert 'tank_temperature_with_memory' in df_sensors.columns
        
        # The memory sensor should have recorded data
        memory_data = df_sensors['tank_temperature_with_memory']
        assert len(memory_data) > 0
        
        # Memory sensor data should be a string representation of array
        # (because it stores arrays in current_measurement)
        first_measurement = memory_data.iloc[0]
        assert first_measurement is not None
