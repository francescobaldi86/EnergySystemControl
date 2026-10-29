# Energy System Control

Energy System Control is a Python library for modelling and simulating small energy systems. It is designed for testing different control strategies against a common physical model, including rule-based, model-predictive, and reinforcement-learning controllers.

The model is assembled from components connected through typed ports. A simulation can include demands, generators, storage units, heat pumps, inverters, grids, sensors, controllers, predictors, and external time-series data.

## What it provides

- Component-based modelling of thermal, electrical, and fluid systems.
- Explicit components such as demands, PV panels, and constant-power producers.
- Controlled components such as heat pumps and resistance heaters.
- Thermal storage tanks and electrical batteries.
- Electrical and fluid grids, buses, and inverters.
- Sensors for temperatures, power, hot-water demand, and battery state of charge.
- Rule-based controllers, MPC controllers, and reinforcement-learning extensions.
- Simulation results as pandas data frames, cumulative energy values, plots, and boundary/comfort indices.
- Input data from custom profiles and weather or PVGIS-based sources.

## Installation

The package requires Python 3.10 or newer. Install the released package with:

```bash
pip install energy-system-control
```

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/francescobaldi86/EnergySystemControl.git
cd EnergySystemControl
pip install -e .
```

The development dependencies can be installed with:

```bash
pip install -e ".[dev]"
```

## First simulation

The following example models a domestic hot-water system with a demand profile, heat pump, hot-water tank, electricity grid, and cold-water grid. The heat pump is switched by a temperature-band controller.

```python
import energy_system_control as esc

components = [
	esc.IEAHotWaterDemand(
		name="dhw_demand",
		reference_temperature=40,
		profile_name="M",
	),
	esc.HeatPumpConstantEfficiency(
		name="heat_pump",
		Qdot_design=1.5,
		COP_design=3.2,
	),
	esc.HotWaterStorage(
		name="hot_water_storage",
		max_temperature=80,
		tank_volume=200,
		T_0=45,
		convection_coefficient_losses=0.0,
	),
	esc.ElectricityGrid(name="electric_grid"),
	esc.ColdWaterGrid(name="water_grid", utility_type="fluid"),
]

sensors = [
	esc.TankTemperatureSensor(
		"tank_temperature",
		"hot_water_storage",
	),
]

controllers = [
	esc.HeaterControllerWithBandwidth(
		"heat_pump_controller",
		"heat_pump",
		"tank_temperature",
		40,
		10,
	),
]

connections = [
	("dhw_demand_fluid_port", "hot_water_storage_hot_water_output_port"),
	("heat_pump_heat_output_port", "hot_water_storage_main_heat_input_port"),
	("heat_pump_electricity_input_port", "electric_grid_electricity_port"),
	("hot_water_storage_cold_water_input_port", "water_grid_fluid_port"),
]

environment = esc.Environment(
	components=components,
	controllers=controllers,
	sensors=sensors,
	connections=connections,
)

configuration = esc.SimulationConfig(
	time_start_h=0.0,
	simulation_end_h=24.0,
	time_step_h=0.5,
)

results = esc.Simulator(environment, configuration).run()
ports, controller_actions, sensors = results.to_dataframe()

print(sensors.head())
print(
	"Heat-pump electricity:",
	results.get_cumulated_electricity("heat_pump_electricity_input_port"),
	"kWh",
)
```

### How the model is assembled

1. **Components** represent physical units and define their ports.
2. **Connections** join ports using their exact generated names. A connection is a tuple containing two port names.
3. **Sensors** expose measurements to controllers and store them in the results.
4. **Controllers** read sensor values and produce actions for controlled components.
5. **Environment** creates the network, validates its connections, and manages registries.
6. **Simulator** advances the system at each configured time step and returns a `SimulationResults` object.

Component and port names are significant. A component named `heat_pump` commonly creates port names such as `heat_pump_heat_output_port`; inspect the component implementation or use the warnings raised during environment construction when connecting a custom model.

## Working with results

`Simulator.run()` returns results that can be converted to three pandas data frames:

```python
df_ports, df_controllers, df_sensors = results.to_dataframe()

# The index is simulation time in hours.
temperature_at_10h = df_sensors.loc[10.0, "tank_temperature"]

# Electricity values are returned in kWh by default.
net_grid_energy = results.get_cumulated_electricity(
	"electric_grid_electricity_port"
)
energy_imported = results.get_cumulated_electricity(
	"electric_grid_electricity_port",
	sign="only negative",
)
energy_exported = results.get_cumulated_electricity(
	"electric_grid_electricity_port",
	sign="only positive",
)
```

For a selected interval, pass its start and end in hours. Results can also be returned in MWh:

```python
weekly_energy = results.get_cumulated_electricity(
	"electric_grid_electricity_port",
	time_interval_h=(0.0, 24.0 * 7),
	unit="MWh",
)
```

The sign convention is determined by the port flows. For the one-sided helpers, `only positive` sums positive samples and `only negative` reports the magnitude of negative samples.

The built-in plotting helpers return Matplotlib objects and can optionally save a figure:

```python
results.plot_temperature_sensors(
	sensors="tank_temperature",
	comfort_temperature=313.15,
	filename="tank_temperature.png",
)
results.plot_electric_power_sensors(
	power_sensors=["grid_power_sensor"],
	filename="grid_power.png",
)
```

## Time and units

- Simulation configuration durations are expressed in hours.
- `time_step_h` is the simulation step in hours; `time_step_s` is derived automatically.
- Simulation result data frames use an index named `time`, expressed in hours.
- Temperatures in the physical model are generally expressed in kelvin. Convert Celsius values before passing them to components or controllers when required.
- Power and flow signals are integrated using the simulation time step.
- Cumulative electricity is returned in kWh by default, or MWh when requested.

## Data-driven components

Several components accept input profiles from files or data providers. For example, an electricity demand can be loaded from a CSV file:

```python
electricity_demand = esc.ElectricityDemand(
	name="demand",
	path="data/electricity_demand.csv",
	var_unit="kWh",
)
```

The file format and time alignment depend on the component. Check the component constructor and the tests under `tests/` for the expected columns and units. PVGIS and weather-backed components may require network access and valid geographic coordinates.

## Controllers and predictors

Controllers are attached to component names and sensor names. The controller order supplied to `Environment` is preserved, so avoid assigning the same component to multiple controllers. Predictors can be passed to `Environment` through the `predictors` argument and are used by controllers that require forecasts.

The top-level package currently provides common building blocks such as `HeaterControllerWithBandwidth`, `ChargeController`, and `HeatPumpRuleBasedController`. MPC and reinforcement-learning controllers are available in their respective controller modules and may require additional configuration or solver dependencies.

## Testing

Run the test suite from the repository root with:

```bash
pytest
```

Some integration tests use external resources, such as weather APIs. A focused example test can be run with:

```bash
pytest tests/full_examples/test_example.py
```

## Project status and contributions

The library is under active development. The public top-level imports in `energy_system_control/__init__.py` are the preferred starting point, while the source tree and tests contain more advanced component and controller examples.

Issues and contributions are welcome through the [GitHub repository](https://github.com/francescobaldi86/EnergySystemControl).

## License

This project is distributed under the license terms described in [LICENSE](LICENSE).