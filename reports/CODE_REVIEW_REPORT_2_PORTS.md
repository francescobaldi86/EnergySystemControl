# Code Review Report 2 — Port usability refactor (report §3, steps 1–4)

Implements the four port-related usability items from `CODE_REVIEW_REPORT.md` §3,
fully backward compatible with the existing string-based API.

## What was implemented

### 1. Port handles as first-class objects

- Ports are now created **at component construction** (`Component.__init__` calls
  `create_ports()`; `create_ports` is idempotent so `Environment.create_ports()`
  registers the same objects instead of recreating them).
- New `Component.port(role)` accessor and `Component.port_roles` property.
  A *role* is the port name without the `<component name>_` prefix:
  `heat_pump.port('heat_output_port')` → the port named `heat_pump_heat_output_port`.
  Full names and `Port` objects are also accepted. Typos raise a `KeyError`
  listing available ports/roles with "did you mean" suggestions.
- `Environment.resolve_port(spec)` / `resolve_component(spec)` / `resolve_sensor(spec)`
  normalize any spec (name string, object, `(component, role)` tuple, single-port
  component, `Port`) to the actual object, with helpful errors.
- Connections now accept any combination of formats:

  ```python
  connections = [
      ('demand_DHW_fluid_port', 'hot_water_storage_hot_water_output_port'),      # legacy strings
      (hp.port('heat_output_port'), tank.port('main_heat_input_port')),           # port handles
      (('heat_pump', 'electricity_input_port'), ('electric_grid', 'electricity_port')),  # (component, role)
  ]
  env = esc.Environment(components=..., sensors=..., controllers=..., connections=connections)
  env.connect(('heat_pump', 'electricity_input_port'), ('electric_grid', 'electricity_port'))  # post-init too
  ```

### 2. Fail fast with good messages at construction

- `connect_ports` normalizes and validates every connection entry:
  - malformed entry → `ValueError` showing accepted formats;
  - unknown port → `KeyError` with the full list of available ports + close-match suggestions;
  - self-connection → `ValueError`;
  - duplicate connection → `ValueError`;
  - port type mismatch / conflicting connection → `ValueError` from `Port.connect_port` (names included).
- Duplicate component / controller / sensor / predictor names in `Environment(...)`
  now raise `ValueError` immediately (previously silently overwritten in the dicts).
- Controllers: `Controller.__init__` accepts objects or names anywhere a name is
  expected (`controlled_components`, `sensors`, `predictors`, and the keys of the
  minimum-time dicts — via `helpers.name_of`). `load_controlled_components` /
  `load_sensors` / `load_predictors` raise `KeyError` naming the controller, the
  unknown reference, the available names, and suggestions.

### 3. Unconnected ports: warn loudly, don't silently delete

- `check_unconnected_ports` no longer deletes ports silently. It collects **all**
  unconnected ports and emits one `warnings.warn(...)` listing them. Ports are
  **kept** in the environment: they stay visible in the results (as zero-flow
  columns) and can still be connected afterwards with `env.connect(...)`.
- New `Environment(..., strict_unconnected_ports=True)` raises `ValueError`
  instead of warning.
- To keep the simulator correct with kept ports, unconnected ports are reset to
  **zero flow** each step (they carry no flow by definition): the step plan
  records them in `StepPlan.unconnected_ports` and `_step` zero-fills them after
  the reset. This replaces the old "delete the port" mechanism without changing
  balance semantics.

### 4. Sensors and controllers take objects/roles, not strings

- All sensors accept a `Port` object, a full port-name string, a `(component, role)`
  tuple, or a single-port component where they take a port; and a `Component`
  object or name where they take a component. Parameter names are unchanged
  (`port_name=`, `component_name=`, `source_sensor_name=`), so existing keyword
  calls keep working.
- `HotWaterDemandSensor` no longer hardcodes `f'{component_name}_fluid_port'` as
  the only option: it takes `port_role='fluid_port'` and resolves it through
  `component.port(role)` (the legacy pattern remains the fallback).
- Resolution happens once at `Sensor.initialize(ctx)` via
  `environment.resolve_*`; sensors also keep a lazy name-based fallback so simple
  test doubles (e.g. `MockEnvironment` in `tests/utils.py`) keep working.
- Controllers: `HeaterControllerWithBandwidth(name, heat_pump, tank_sensor, 40, 10)`
  now works with the component and sensor **objects**; `ChargeController` accepts a
  `Battery` object for `battery_name`.

## Files changed

| File | Change |
|---|---|
| `helpers.py` | new `name_of(obj)` and `close_matches(name, available)` helpers |
| `components/base.py` | ports created in `__init__` (idempotent `create_ports`); `Component.port(role)`, `Component.port_roles` |
| `core/base_environment.py` | `_dedup_named` (duplicate-name validation); `resolve_component/sensor/port`; `_normalize_connection`; `connect()`; validating `connect_ports`; loud non-deleting `check_unconnected_ports` + `strict_unconnected_ports`; `StepPlan.unconnected_ports` |
| `sensors/sensors.py` | all sensors accept objects/roles/tuples; `_resolve_refs` hook in `Sensor.initialize`; `HotWaterDemandSensor` uses `port_role` instead of a hardcoded name |
| `controllers/base.py` | object-or-name normalization in `Controller.__init__`, `HeaterControllerWithBandwidth`, `ChargeController`; helpful `KeyError`s in `load_*` |
| `sim/simulator.py`, `sim/step_plan.py` | zero-fill unconnected ports each step (`plan.unconnected_ports`) |

## Verification

- New-API vs legacy-string model produce **identical result DataFrames** (same columns,
  same values) on the DHW + heat-pump + storage + grids example; physics results
  unchanged on the profile runs (2.156 / 12.422 kWh).
- Error paths exercised: unknown port (with suggestion), self-connection, duplicate
  connection, type mismatch, malformed entry, strict unconnected mode, unknown
  sensor in a controller, duplicate component names, `component.port()` typo.
- Full `pytest tests` run: failure set **byte-identical** to the pre-change baseline
  at commit `30683db` (88 pre-existing failures, all from tests targeting a newer
  API than this commit — none introduced by this refactor). `test_example.py::test_2`
  passes.
- cProfile after the change: 1440-step run 0.103 s (vs 0.214 s at `4b4f91c`, before
  the step-plan optimization) — the port refactor added no per-step overhead.

## §3.6 `env.describe()` / `env.to_graph()` — component/port/connection graph (implemented)

New module `core/describe.py`. It imports nothing from the package (it duck-types the
environment), so it can be imported from anywhere without cycles:

- `build_graph(env)` walks components, ports, connections, sensors and controllers and
  returns a plain dict: components (name, type, category, composite flag, ports with
  `role / type / layers / connected_to`, composite `members` and `exposed_ports`),
  `connections` (endpoints, layers, `internal` flag), `sensors`, `controllers`,
  `unconnected_ports`, `issues`.
- Renderers: `render_text` (default), `render_mermaid`, `render_dot`;
  `render(graph, format, expand_composites)` raises `ValueError` on unknown formats.

`Environment` API:

```python
graph = env.to_graph()                            # machine-readable dict
print(env.describe())                             # text; printed and returned
print(env.describe(format='mermaid'))             # 'mermaid' or 'dot'
print(env.describe(expand_composites=False))      # flat view, edges on internal members
```

Supporting changes:

- `core/base_environment.py`: `__init__` initializes `composite_members` (composite name
  → internal component names) and `internal_connections` (frozensets of port pairs),
  filled by `load_internal_components()`; new `to_graph()` and `describe(...)` methods.
- `components/base.py`: `CompositeComponent.port(role)` falls back to searching the
  internal components' ports for `role` or `f'{name}_{role}'` — composites usually have
  no ports of their own (e.g. `('battery', 'electricity_port')` resolves to the internal
  charger's `battery_electricity_port`).
- In text output a composite shows its members, its exposed ports
  (`electricity_port -> battery_charger.battery_electricity_port`) and the member ports
  labelled with the exposed role; in mermaid/dot, external edges are drawn from the
  composite node and internal edges are dashed inside the subgraph.

Files: `core/describe.py` (new), `core/base_environment.py`, `components/base.py`,
`tests/unit/core/test_describe.py` (new).

Example (DHW demand + heat pump + tank + grids + battery):

```
Environment: 8 components (1 composite), 12 ports, 5 connections (1 internal)
Unconnected ports (zero flow): heat_pump_electricity_input_port, tank_aux_heat_input_port
...
  battery  [LithiumIonBattery / CompositeComponent]
    members: battery_pack (LithiumIonBatteryPack), battery_charger (BatteryCharger)
    exposed ports: ..., electricity_port -> battery_charger.battery_electricity_port
    electricity_port  ElectricPort(electricity) --> grid.electricity_port
...
Issues:
  - port 'heat_pump_electricity_input_port' is not connected (carries zero flow)
  - port 'tank_aux_heat_input_port' is not connected (carries zero flow)
```

Verification: `tests/unit/core/test_describe.py` — 5/5 pass (graph structure, text output,
composite + internal connections, all three formats + bad-format error, composite role
resolution). Full `pytest tests` failure set identical to the baseline at `e5fd16e`
(same 88 pre-existing failures, none new); `test_example.py::test_2` passes; the step-2
checks (`verify_ports.py`) still pass.

## Deliberate scope limits / follow-ups (report §3 items 5, 6, 8)

- Port naming is still `f'{name}_...'`-based; `BiogasStorage` (`_input`/`_output`)
  and the charger (`f'{name}_port'`) still break the `_port` suffix convention, and
  composite ports are still implicitly numbered (`inverter_AC_output_port_0`).
  Unifying these is a mechanical but breaking change — do it together with a
  deprecation path for the string names.
- `env.describe()` (print the component/port/connection graph) was implemented after
  this report — see the §3.6 section above.
- Structured results accessors (`results.port(...).layer(...)`) are not yet
  implemented; the `port:layer` DataFrame columns remain the API.
