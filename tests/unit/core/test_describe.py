import warnings

import pytest

import energy_system_control as esc


def build_small_env():
    demand = esc.HotWaterDemand.from_iea(name="demand_DHW", reference_temperature=40, profile_name='M')
    hp = esc.HeatPumpConstantEfficiency(name='heat_pump', Qdot_design=1.5, COP_design=3.2)
    tank = esc.HotWaterStorage(name='hot_water_storage', max_temperature=80, tank_volume=200, T_0=45,
                               convection_coefficient_losses=0.0)
    grid = esc.ElectricityGrid(name='electric_grid')
    water = esc.ColdWaterGrid(name='water_grid', utility_type='fluid')
    sensor = esc.TankTemperatureSensor('storage_tank_temperature_sensor', tank)
    controller = esc.HeaterControllerWithBandwidth('heat_pump_controller', hp, sensor, 40, 10)
    connections = [
        (('demand_DHW', 'fluid_port'), ('hot_water_storage', 'hot_water_output_port')),
        (hp.port('heat_output_port'), tank.port('main_heat_input_port')),
        (('heat_pump', 'electricity_input_port'), 'electric_grid_electricity_port'),
        (('hot_water_storage', 'cold_water_input_port'), ('water_grid', 'fluid_port')),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # aux heat port is intentionally unconnected
        env = esc.Environment(components=[demand, hp, tank, grid, water],
                              controllers=[controller], sensors=[sensor], connections=connections)
    return env


def build_composite_env():
    battery = esc.LithiumIonBattery(name='battery', capacity=2.0, SOC_0=0.5)
    grid = esc.ElectricityGrid(name='electric_grid')
    env = esc.Environment(components=[battery, grid], controllers=[], sensors=[],
                          connections=[('battery_electricity_port', 'electric_grid_electricity_port')])
    return env


def test_to_graph_structure():
    env = build_small_env()
    graph = env.to_graph()
    names = {c['name'] for c in graph['components']}
    assert {'demand_DHW', 'heat_pump', 'hot_water_storage', 'electric_grid', 'water_grid'} <= names
    # roles are exposed, not raw generated names
    hp = next(c for c in graph['components'] if c['name'] == 'heat_pump')
    roles = {p['role'] for p in hp['ports']}
    assert roles == {'heat_output_port', 'electricity_input_port'}
    # connections recorded with endpoints
    pairs = {frozenset((c['a'], c['b'])) for c in graph['connections']}
    assert frozenset(('heat_pump_heat_output_port', 'hot_water_storage_main_heat_input_port')) in pairs
    # unconnected aux port is visible
    assert graph['unconnected_ports'] == ['hot_water_storage_aux_heat_input_port']
    assert any('aux_heat_input_port' in issue for issue in graph['issues'])
    # sensors/controllers described
    assert graph['sensors'][0]['measures'] == "component 'hot_water_storage'"
    assert graph['controllers'][0]['controls'] == ['heat_pump']


def test_describe_text():
    env = build_small_env()
    text = env.describe(print_output=False)
    assert 'heat_pump.heat_output_port' in text
    assert 'hot_water_storage.main_heat_input_port' in text
    assert 'Unconnected ports' in text
    assert 'heat_pump_controller' in text


def test_describe_composite_and_internal_connections():
    env = build_composite_env()
    graph = env.to_graph()
    battery = next(c for c in graph['components'] if c['name'] == 'battery')
    assert battery['composite'] is True
    assert set(battery['members']) == {'battery_pack', 'battery_charger'}
    # the external role 'electricity_port' maps to the charger's internal port
    assert battery['exposed_ports']['electricity_port'] == 'battery_electricity_port'
    internal = [c for c in graph['connections'] if c['internal']]
    assert len(internal) == 1
    assert frozenset((internal[0]['a'], internal[0]['b'])) == {'battery_pack_electricity_port',
                                                                'battery_charger_port'}


def test_describe_formats():
    env = build_composite_env()
    text = env.describe(print_output=False)
    assert 'members: battery_pack' in text
    assert 'exposed ports:' in text and 'electricity_port' in text
    mermaid_flat = env.describe(format='mermaid', expand_composites=False, print_output=False)
    assert 'battery_charger --' in mermaid_flat  # flat view keeps edges on internal members
    mermaid = env.describe(format='mermaid', print_output=False)
    assert mermaid.startswith('graph LR')
    assert 'subgraph battery' in mermaid
    dot = env.describe(format='dot', print_output=False)
    assert dot.startswith('digraph')
    assert 'cluster_battery' in dot
    with pytest.raises(ValueError):
        env.describe(format='yaml', print_output=False)


def test_composite_port_role_resolution():
    # a composite has no own ports; roles resolve to the internal components' ports
    battery = esc.LithiumIonBattery(name='battery', capacity=2.0, SOC_0=0.5)
    port = battery.port('electricity_port')
    assert port.name == 'battery_electricity_port'
    grid = esc.ElectricityGrid(name='electric_grid')
    env = esc.Environment(components=[battery, grid], controllers=[], sensors=[],
                          connections=[(('battery', 'electricity_port'), ('electric_grid', 'electricity_port'))])
    assert env.ports['battery_electricity_port'].connected_port is env.ports['electric_grid_electricity_port']
