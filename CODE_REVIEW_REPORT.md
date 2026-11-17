# EnergySystemControl — Code Review Report

Scope: full package (`core`, `sim`, `components`, `controllers` incl. MPC/RL/predictors, `sensors`, `io`, `helpers`, `uncertainty`) plus the test suite, with cProfile runs on the `tests/full_examples` systems. Focus areas requested: **speed**, **algebraic-loop handling / model complexity**, **usability** — starting from `Environment` and `Simulator`.

## 0. How the core works today

`Environment` (`core/base_environment.py`) is a builder, not a runtime object: it flattens composite components (`load_internal_components`), classifies components into fixed buckets (`ExplicitComponent` / `ControlledComponent` / `ImplicitComponent` / `StorageUnit` / `Bus` / `Grid` / `CompositeComponent`), creates one `Port` object per entry in each component's `ports_info` dict (string name → type string), and connects ports from a list of **string tuples** (`self.ports[connection[0]]`, `base_environment.py:145-152`). Unconnected ports are **silently deleted** with a `print` (`check_unconnected_ports`, `:122-137`).

`Simulator._step` (`sim/simulator.py:121-160`) runs 8 phases per time step: update environmental data → measure all sensors → reset all port flows → propagate inherited port attributes → controller actions → simulate components in the fixed class order (Explicit → Controlled → **algebraic loop** → Storage → Grid) → check connection balance → save results cell-by-cell into preallocated numpy arrays via the `SignalRegistry` (`col_index` builds a frozen `SignalKey` dataclass and hashes it **per signal per step**).

The algebraic loop (`_solve_algebric_networks`, `sim/simulator.py:214-227`) is a blind fixed-point iteration: repeatedly call `balance(state)` on every remaining Bus/ImplicitComponent, propagate updated ports, remove solved ones, and raise `RuntimeError` only if a full pass updates nothing. It works today because a `Bus.balance` (`components/base.py:76-132`) only needs one unknown per layer, but it is O(n²) per step (list `remove` during iteration), has no iteration cap, and gives no structural diagnosis when it fails.

### Measured baseline (cProfile, editable install)

| System | Steps | Total | Dominant costs per step |
|---|---|---|---|
| 5 components / 8 ports, 24 h @ 1 min | 1440 | 0.18 s (~125 µs/step) | `_save_simulation_data` ≈ 50% of step (of which `col_index` ≈ 28%), component stepping ≈ 25% |
| 13 components / 20 ports (PV+battery+inverter), 24 h @ 1 min | 1440 | 0.40 s (~280 µs/step) | `_save_simulation_data` 46%, `col_index` 25%, `_take_component_step` 13%, algebraic loop ~9% |

Extrapolated: one year at 1-minute resolution (525,600 steps) costs **~1–2 minutes of pure Python bookkeeping for a 5-component system** before any physics — and with an MPC controller solving a MILP every step (`controllers/MPC.py:246-249`, cvxpy/HIGHS with boolean variables, no warm start), a year becomes hours.

## 1. Speed

The cost is not in the physics — it's in per-step Python bookkeeping that scales with `(#ports + #components + #sensors + #controllers) × #steps`, plus per-step pandas/pvlib/HTTP calls in a few components and controllers. Recommendations, roughly in order of ROI:

1. **Compile the step plan once, not per step.** At `Simulator.run()` start, "compile" the environment into flat execution structures:
   - Precompute for each port a list of `(layer, col)` pairs once — replaces `SignalRegistry.col_index` (dataclass hash + eq per call, ~25% of step time) with a plain list index.
   - Replace the per-port `flows: Dict[str, float]` with a **single flat numpy vector** for all port flows, where each (port, layer) has a precomputed integer slot. All the dict `.items()` loops in `_take_component_step`, `Port.propagate_port_values`, `_check_connection_balance`, `_save_simulation_data` become array slicing. This alone should cut step time several-fold.
   - Precompute the list of (port, layer, col, kind) records for `_save_simulation_data` and write each timestep row with one vectorized assignment (`sim_data.ports[time_id, cols] = vals`) instead of scalar `arr[i, j] = x` writes per signal per step.
   - Precompile `_check_connection_balance` into index pairs and vectorize the check (`np.abs(flows[slots_a] + flows[slots_b]) > tol`).
2. **Kill per-step pandas/pvlib work.** `_update_environmental_data` calls `pd.to_timedelta(...)` and possibly `calculate_solar_angles` (pvlib) every step (`simulator.py:162-176`). Solar angles depend only on `time_id` + location: compute the **whole horizon once** (pvlib accepts a `DatetimeIndex` — `helpers.py:344` already supports it) and index by `time_id`. Same for `PVpanelFromIrradiation.step`, which recomputes solar position per step even though the simulator already computed it (`pv_panels.py:168-182`). `APIEnvironmentalProvider` makes a synchronous HTTP call per step (`io/data_provider.py:350-355`) — providers should be able to return a preloaded array indexed by `time_id`.
3. **Make the algebraic loop cheap.** Precompute each Bus/ImplicitComponent's port/layer list once (currently rebuilt with sets/dicts on every `balance()` call, `base.py:83-95`); iterate over a copy of the remaining list instead of mutating during iteration; better, track a "dirty" set so passes only revisit components whose inputs changed (see §2).
4. **Controller-side costs.** MPC: warm-start and shift the previous solution, or solve every k steps and hold/interpolate actions in between; consider an LP relaxation path (OSQP) when binary variables are fixed. Predictors: `ProfileARPredictor` refits (`lstsq`) **every step** (`predictors.py:877-878`) and `MLBasedPredictor` can refit sklearn inside `predict()` — refit on a schedule (every N steps) and cache pandas index operations (`date_range`/`unique().sort_values()` per call in `OfflineForecastPredictor.predict`, `predictors.py:151-235`).
5. **Structural options for big runs.** (a) ExplicitComponents are pure functions of `time_id` and environment data, so their port histories can be **vectorized across the whole time axis** before the loop — a large fraction of ports never enters the loop. (b) Keep the results arrays `float64` (currently `float32`, `simulation_data.py:12-14` — cumulative energy sums over 525k steps lose precision). (c) Offer a numba/Cython-compiled inner loop behind an optional dependency, and/or a "vectorized ensemble" mode that runs N scenarios by stacking state arrays — useful for uncertainty studies (`uncertainty.py` exists but each run is currently serial). (d) Generalize the input-keyed caching pattern already used by `MultiNodeHotWaterTank` (`thermal_storage.py:300-318`) to other components.

## 2. Algebraic loops / model complexity

The current "try components one by one and hope one closes the loop" approach is a Gauss–Seidel iteration with no structure. It works because today's loops are tiny (one bus + one converter), but it will not scale and cannot distinguish "slow convergence" from "no solution". Replace it with a **build-time graph analysis + per-step structured solve**:

1. **Declare causality explicitly.** Each component declares which port-layers it *reads* and *writes* (metadata on `ports_info`, e.g. `{role: (type, direction)}`). Small API change, and it immediately gives validation: unconnected reads, over-determined ports, and self-loops become build-time errors instead of mid-simulation `RuntimeError`.
2. **Build the dependency graph once in `Environment`.** Nodes = components (or port-layer variables); edges from a written variable to its readers. Then:
   - Topological sort (Kahn) → all acyclic components are stepped **once, in order**. Today's fixed class order (Explicit → Controlled → …) is a hand-rolled approximation of this.
   - Tarjan's SCC → only the strongly connected components remain for iteration. In current models those are just the inverter buses, so the per-step cost collapses to a tiny subgraph.
3. **Choose the solver per SCC:**
   - *Linear SCC* (buses, converters with fixed efficiency — the common case): assemble a sparse linear system `A·x = b` over the SCC's port flows and solve once with `scipy.sparse.linalg.spsolve`. Exact, one shot, no convergence tolerance. Most of today's "algebraic loops" are really linear systems and should never iterate.
   - *Nonlinear SCC*: fixed-point (current behavior, but bounded with max-iterations + tolerance + proper tearing via a minimum feedback arc set to pick iteration variables), or Newton–Raphson if components optionally expose `residual(x)` / `jacobian(x)`. Tearing turns "hope a component closes the loop" into an automatic, minimal choice of unknowns.
4. **Formalize the step as an index-1 DAE:** per step, (a) given state(t) and controller actions, solve the algebraic network for all port flows; (b) update storage states. That's what the code implicitly does; making it explicit (algebraic solve → state update) is what lets you swap solvers and parallelize the acyclic part.
5. **Diagnostics:** when the solve fails, report *which variables are unresolved and which components form the cycle* (from the SCC), not just "remaining components: [...]". Add a max-iteration cap with a clear error, and fix the mutation-during-iteration in `_solve_algebric_networks` (`simulator.py:216-220`). Related fragility: `Bus.balance` currently **raises** when it is already fully solved (`base.py:112`) instead of returning solved — that will crash the solver on topologies where all neighbors are explicit.

## 3. Usability

The biggest issue: **ports exist only as generated strings**. `f'{name}_heat_output_port'` is reproduced by hand in connections, sensors (`HotWaterDemandSensor` hardcodes `f'{component_name}_fluid_port'`, `sensors.py:126`), controllers (`ChargeController` builds `f'{battery_name}_charger'` and reaches into `.charger`/`.battery_pack`, `controllers/base.py:209,232-236`), composite internals (`inverter_PV_input_port_0` with implicit 0-based numbering), and tests. A typo in a connection string is a bare `KeyError` at `connect_ports`; a forgotten connection silently **deletes the port** and changes the topology with only a printed warning (e.g. `WARNING: Port inverter_engine_port was removed...` fires even in `test_5`).

Suggestions:

1. **Port handles as first-class objects.** Components declare ports by *role* (`ports_info = {'heat_output': ('heat', 'out'), ...}`) and expose `component.port('heat_output')` returning the `Port`. `Environment.connect(a, b)` accepts Port objects, component+role tuples (`('heat_pump', 'heat_output') → ('hot_water_storage', 'main_heat_input')`), or legacy strings. The global string name becomes a derived detail, not the API.
2. **Fail fast with good messages at construction.** Unknown port in a connection → error listing the available ports/roles of the likely components ("did you mean `heat_pump_heat_output_port`?"). Port type mismatch already raises in `Port.connect_port` — keep it, but include both names and types. Duplicate connections, self-connections, and connecting two ports of the same bus → build-time errors.
3. **Unconnected ports: warn loudly, don't silently delete.** Collect them and report all at once at the end of `__init__`; add a `strict=True` mode that raises. Deleting a port from `component.ports` mid-build is what makes composite topologies quietly change (the inverter engine port case).
4. **Sensors and controllers take objects/roles, not strings.** `TankTemperatureSensor('t', storage)`, `HotWaterDemandSensor('s', demand)` — removes the hardcoded `f'{component_name}_fluid_port'` coupling. Controller `load_sensors`/`load_predictors` (`controllers/base.py:84-88`) should raise "unknown sensor X, available: [...]" instead of bare `KeyError`.
5. **One naming convention.** Everything is `f'{name}_..._port'` except `BiogasStorage` (`f'{name}_input'`/`f'{name}_output'`, `biogas_storage.py:33-34`) and the charger's `f'{name}_port'` (`electric.py:18`) — unify, and make composite port numbering explicit (declare `pv_inputs: int = 1` rather than an implicit 0-based loop).
6. **Composite components:** expose their internal topology (`env.describe()` printing components, ports, connections as a graph) so users can see what `Inverter` actually expands into; build internal connections from port handles too.
7. **Clean up the initialization path.** `Environment.read_timeseries_data` (`base_environment.py:155-158`) is dead code with a wrong signature; `Simulator._read_timeseries_data` is commented out; `Environment.initialize` calls `provider.initialize(state, cfg)` but `EnvironmentalDataProvider.initialize` expects a `ctx` (`io/data_provider.py:158`) — pick one contract. `SimulationConfig` mixes `time_start_h` and `simulation_start_datetime` and is frozen while holding a Timestamp default.
8. **Results API:** keep the `port:layer` DataFrame columns, but add structured accessors (`results.port('heat_pump_electricity_input_port').electricity`) so users don't have to remember the `:` convention.

## 4. Correctness bugs worth fixing in the same pass

- `minimum_time_on`/`minimum_time_off` are **swapped** in `Controller.__init__` (`controllers/base.py:63-64`; subclasses compensate by swapping args, e.g. `rule_based.py:166-167,325-326`).
- `ConstantPowerDemand.step` writes `port.flow` instead of `port.flows` (`demands.py:45`).
- `Demand.initialize` reads `ctx.seed`/`self.seed` which don't exist (`demands.py:33-38`).
- `GaussianUncertainty.apply` ignores `mode` and is always multiplicative (`uncertainty.py:30-31`).
- Battery self-discharge is dimensionally wrong; `BatteryPack.get_maximum_charge/discharge_power` read attributes that only exist on the charger (`electric_storage.py:30,48,51-55`).
- `BiogasStorage.step` crashes on `None` flows (`biogas_storage.py:116,130-133`).
- `Controller.get_obs` uses `self.horizon` that only MPC/RL controllers define (`base.py:78`); `HeaterControllerWithBandwidth` double-applies the on/off time check (`base.py:199` + `get_action`).
- MPC solves a MILP but the advertised solver default includes OSQP (cannot handle booleans); no warm start (`MPC.py:117,246-249`).
- RL wiring bugs: `SARSAController` passes nonexistent kwargs (`RLcontrollers.py:273-274`); `'config_info'` vs `'config info'` key mismatch (`RLcontrollers.py:45` vs `exploration_policies.py:25`); `ComponentCostReward` reads `controller.last_action` which is never set (`reward_functions.py:297-300`).
- `tests/PLAYGROUND/aaa_physical_system.py` imports nonexistent modules and contains what looks like a **live Home Assistant token** — delete/rotate it. `tests/conftest.py`'s `minimal_environment` fixture references a nonexistent API.

## 5. Suggested priority

1. Port handles + build-time validation (usability; unblocks everything else and kills a whole class of silent bugs).
2. Compiled step plan: flat flow vector + precomputed column indices + vectorized save/balance (speed; biggest single win, ~2–3× on small systems, more as port count grows).
3. Graph-based solve: causality metadata → topo sort + SCC + sparse linear solve for linear SCCs (model complexity; removes the ceiling on loop handling).
4. One-shot environmental/solar data and predictor refit scheduling (speed, mostly affects PV/MPC setups).
5. Fix the correctness list above, especially the swapped on/off times and the token in PLAYGROUND.
