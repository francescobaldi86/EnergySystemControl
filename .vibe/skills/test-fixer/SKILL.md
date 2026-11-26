---
name: test-fixer
description: >-
  Load this skill when the user wants to run the pytest test suite of the
  EnergySystemControl package and automatically diagnose and fix failing
  tests. Invoke with as little information as possible — just say "fix the
  tests" or "run and fix tests". Also loads when the user mentions broken
  tests, failing CI, or asks to make the test suite green again.
---

# test-fixer — Run and Fix the EnergySystemControl Test Suite

You are a test-fixing agent for the **EnergySystemControl** Python package.
Your job: run the test suite, diagnose every failure, fix the root cause in
the source code, and re-run until everything passes. The user should not need
to tell you which tests to run or where the code lives — discover it all
yourself.

---

## 1. Project layout (auto-discovered — do not ask the user)

| What | Where |
|---|---|
| Project root | The directory that contains `.vibe/skills/test-fixer/` (i.e. the repo root) |
| Source code | `src/energy_system_control/` |
| Tests | `tests/` |
| Test config | `pyproject.toml` → `[tool.pytest.ini_options]` (testpaths = `["tests"]`, addopts = `-ra -q`) |
| Python env | conda env **`EnergySystemControl-dev`** |
| Reports | `reports/` — test-fixer writes its reports here as `TEST-FIXER-REPORT YYYY-MM-DD HH:MM:SS.md` |

### Test directory structure

```
tests/
├── conftest.py                  # fixtures, --runslow option, slow-marker skip
├── utils.py                     # MockSensor, MockEnvironment, metrics helpers
├── unit/
│   ├── components/              # test_heat_pumps, test_inverters, test_pv_panels, test_storage
│   ├── controllers/             # test_mpc_controller, test_opti_controller, test_rl_controller,
│   │                            #   test_rule_based, test_offline_forecast_predictor
│   ├── core/                    # test_describe, test_nodes
│   ├── predictors/              # test_ar_predictor, test_base_predictors, test_ml_predictor,
│   │                            #   test_profilear_predictor
│   ├── sensors/                 # test_flow_temperature_sensor, test_hot_water_demand_sensor,
│   │                            #   test_power_sensor, test_sensor_with_memory, test_soc_sensor,
│   │                            #   test_tank_temperature_sensor
│   ├── test_io_loaders.py
│   └── test_results.py
├── full_examples/test_example.py
├── io/                          # test_data_provider, test_weather_api (some @integration)
├── DATA/                        # CSV data files used by tests (pvgis_data.csv, dhw_demand_data.csv, …)
├── integration/ perf/ property/  # (currently no .py test files)
├── Dragos_tests/ Martina_tests/ PLAYGROUND/   # personal scratch — usually skip
```

### Source structure (where fixes go)

```
src/energy_system_control/
├── __init__.py                  # public API re-exports
├── components/                  # base, controlled_components/, composite_components/,
│                                #   explicit_components/, grids/, implicit_components/, storage_units/
├── controllers/                 # base.py, rule_based.py, MPC.py, predictors.py, RL/
├── core/                        # base_classes, base_environment, node, port, registry, describe
├── sensors/sensors.py
├── sim/                         # config, simulator, state, results, step_plan, simulation_data
├── io/                          # data_provider, weather_api
├── constants/                   # fluids, thermo
├── helpers.py
└── uncertainty.py
```

---

## 2. How to run the tests

Always run from the **project root**. Use the conda environment's Python so
that all dependencies (cvxpy, pvlib, scikit-learn, …) are available.

### Preferred invocation

```bash
cd "<project-root>"
C:/Users/francesco.baldi/AppData/Local/miniconda3/envs/EnergySystemControl-dev/python.exe -m pytest tests/ -v --tb=short 2>&1
```

On Git Bash (Windows), the path works with forward slashes as above. If that
exact path doesn't exist, fall back to:

```bash
conda run -n EnergySystemControl-dev python -m pytest tests/ -v --tb=short
```

or simply `python -m pytest` if the active env already has the package.

### Useful pytest flags

| Goal | Flag |
|---|---|
| Verbose list of every test | `-v` |
| Short traceback | `--tb=short` |
| Full traceback (when short is unclear) | `--tb=long` |
| Run a single test file | `tests/unit/controllers/test_mpc_controller.py` |
| Run a single test | `tests/unit/controllers/test_mpc_controller.py::test_initialization` |
| Run only fast tests (default — slow are auto-skipped) | (no flag needed) |
| Include slow tests | `--runslow` |
| Stop on first failure | `-x` |
| Show locals in traceback | `-l` |
| Machine-readable report | `--tb=no -q` then parse, or use `-v` and parse output |

### Markers

- `slow` — auto-skipped unless `--runslow` is passed (see `tests/conftest.py`).
- `integration` — tests that hit the network (e.g. `tests/io/test_weather_api.py`). These may fail due to no network; if so, note it and move on — do **not** try to fix network issues.

### Default scope

Run `tests/unit/` and `tests/full_examples/` by default. The `tests/io/` tests
that are marked `@integration` may need network — run them but don't block on
them. **Skip** `tests/Dragos_tests/`, `tests/Martina_tests/`, `tests/PLAYGROUND/`
unless the user explicitly asks for them.

---

## 3. Workflow — follow this loop for every invocation

```
┌──────────────────────────────────────────────────────────────┐
│  1. RUN        Run the test suite (default scope).           │
│  2. PARSE      Extract every failure: test node id, error     │
│                type, traceback, file:line.                    │
│  3. GROUP      Group failures by root cause (same exception   │
│                in same source file = one fix).                │
│  4. DIAGNOSE   For each group, read the failing test, then    │
│                read the source file(s) implicated in the      │
│                traceback. Understand *why* it fails.          │
│  5. FIX        Edit the **source code** in src/ to fix the    │
│                root cause. Do NOT weaken or delete test       │
│                assertions unless the test itself is provably  │
│                wrong (e.g. testing removed API).              │
│  6. RE-RUN     Re-run only the previously failing tests first │
│                (fast feedback), then the full suite.          │
│  7. REPEAT     Go to step 2 until zero failures or you hit a   │
│                hard blocker (see §5).                         │
│  8. REPORT     Write the report to reports/TEST-FIXER-REPORT  │
│                <timestamp>.md and summarise in chat.          │
└──────────────────────────────────────────────────────────────┘
```

**Timestamp:** at the very start of the run (step 1), record the current
date/time as `YYYY-MM-DD HH:MM:SS` (e.g. via `date +"%Y-%m-%d %H:%M:%S"`).
Use this exact string as the `<timestamp>` in the report filename, but on
Windows replace the colons in the time part with dots (colons are invalid
in Windows filenames), e.g. `2026-11-26 18.00.00`.

### Step 3 — Grouping heuristics

- **Import errors** (`ModuleNotFoundError`, `ImportError`, `AttributeError` on import): usually one missing/wrong export in `src/energy_system_control/__init__.py` or a renamed module. Fix the import chain once.
- **`AttributeError` inside a method**: the object doesn't have the expected attribute — likely a refactor renamed it. Find the new name in the source and update all call sites.
- **`TypeError` / wrong number of arguments**: a function signature changed. Update the caller or the callee consistently.
- **`AssertionError` with `math.isclose` / numeric tolerance**: the physics/simulation changed slightly. Investigate whether the source behaviour is correct; if so, the test's expected value or tolerance may need updating — but only after confirming the source is right.
- **`KeyError` on port/component/sensor names**: a name changed during refactoring. Trace the name through `src/` and align it.
- **`ValueError` from cvxpy / solver**: the MPC problem became infeasible. Read the MPC code in `controllers/MPC.py` and the constraints; fix the formulation.
- **Fixture errors** (`fixture 'X' not found`): a fixture was moved or renamed. Check `tests/conftest.py` and the test file's own fixtures.

### Step 4 — Diagnosis rules

1. **Always read the test file first** to understand what behaviour is expected.
2. **Then read the source** implicated in the traceback — start from the deepest frame in `src/`.
3. **Use `grep` to find all usages** of a renamed symbol before editing, so you catch every call site.
4. **Check git history** (`git log --oneline -20`, `git diff HEAD~1 -- src/`) if a recent commit likely broke things.
5. If a test references a class/function that no longer exists, search the source tree for the new equivalent (`grep -r "class.*Name" src/`).

### Step 5 — Fixing rules

- **Fix source, not tests.** The tests encode the expected behaviour; the source should be made to match. Only edit a test if it is objectively wrong (testing a removed feature, typo in expected value, etc.).
- **Make minimal edits.** Change the smallest thing that fixes the root cause. Don't refactor unrelated code.
- **Preserve public API.** If `__init__.py` exports something that tests import, keep it working.
- **One fix per failure group.** If 10 tests fail because `Foo.bar()` was renamed to `Foo.baz()`, fix the rename in one place (or all call sites) — don't patch each test individually.
- **After each fix, re-run the affected tests** before moving to the next group.

---

## 4. Hard blockers — when to stop and report

Stop the loop and report to the user if:

- The same failure persists after **3 different fix attempts** on the same root cause.
- A test requires **network access** and the network is unavailable (integration tests).
- A test requires **external data files** that are missing from `tests/DATA/`.
- The fix would require **redesigning a component** (architectural change) — describe the problem and propose the change, but don't unilaterally rewrite major subsystems.
- You detect that the **test itself is wrong** (not the source). Explain why and ask the user before modifying the test.

---

## 5. Reporting format

When done (or blocked), write the report to a file **and** summarise in chat.

### File output

Write the report to:

```
reports/TEST-FIXER-REPORT <timestamp>.md
```

where `<timestamp>` is the `YYYY-MM-DD HH:MM:SS` string captured at the start
of the run. Create the `reports/` directory if it does not exist. The report
content is the markdown below.

### Chat summary

Also produce the same summary in the chat response:

```markdown
## Test-Fixer Report

**Initial state:** X passed, Y failed, Z skipped (out of N total)
**Final state:**   X' passed, Y' failed, Z' skipped

### Fixes applied

| # | Root cause | Files changed | Tests fixed |
|---|-----------|---------------|-------------|
| 1 | `MPCController.horizon` validation used `<=` instead of `<` | `src/.../MPC.py:42` | test_horizon_validation, … |
| 2 | … | … | … |

### Remaining failures (if any)

- `test_foo` — <reason>, <what was tried>

### Skipped

- Slow tests (use `--runslow` to include)
- Integration tests requiring network

**Report saved to:** `reports/TEST-FIXER-REPORT <timestamp>.md`
```

---

## 6. Quick reference — common commands

```bash
# Full default run
python -m pytest tests/unit tests/full_examples -v --tb=short

# Quick re-run of just the failures from last run
python -m pytest tests/unit tests/full_examples --lf -v --tb=short

# Single file
python -m pytest tests/unit/controllers/test_mpc_controller.py -v --tb=long

# Single test
python -m pytest "tests/unit/controllers/test_mpc_controller.py::test_initialization" -v --tb=long

# With slow tests
python -m pytest tests/unit tests/full_examples -v --tb=short --runslow

# Find a symbol in source
grep -rn "def get_action" src/
grep -rn "class MPCController" src/

# See recent changes that might have broken things
git log --oneline -10
git diff HEAD~1 --stat
```
