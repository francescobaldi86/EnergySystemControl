from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Sequence, Optional, Tuple, Literal
import numpy as np
import pandas as pd
import cvxpy as cp
from energy_system_control.controllers.base import Controller

SolverName = Literal["OSQP", "HIGHS"]

@dataclass(frozen=True, slots=True)
class LinearDiscreteModel:
    # x_{k+1} = A x_k + B u_k + E w_k + g
    A: np.ndarray
    B: np.ndarray
    E: Optional[np.ndarray] = None
    g: Optional[np.ndarray] = None

class MPCController(Controller):
    """
    Generic linear MPC base controller.

    Subclasses must implement:
      - get_model(...)
      - get_current_state(...)
      - get_disturbance_forecast(...)  (or return None)
      - build_stage_constraints(...)
      - build_objective(...)
      - action_from_u0(...)
    """

    def __init__(
        self,
        name: str,
        controlled_components: list[str],
        sensors: Dict[str, str],
        *,
        predictor=None,
        horizon_steps: int,
        solver: SolverName = "OSQP",
        warm_start: bool = True,
    ):
        super().__init__(name, controlled_components, sensors)
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be > 0")
        self.N = int(horizon_steps)
        self.predictor = predictor
        self.solver = solver
        self.warm_start = warm_start

        self._problem: Optional[cp.Problem] = None
        self._x = None
        self._u = None
        self._w = None
        self._params: Dict[str, Any] = {}

    # ----- hooks to implement in concrete MPC controllers -----

    def get_model(self, environment, state) -> LinearDiscreteModel:
        raise NotImplementedError

    def get_current_state(self, environment, state) -> np.ndarray:
        raise NotImplementedError

    def get_disturbance_forecast(
        self,
        environment,
        state,
        now: pd.Timestamp,
        dt_s: float,
    ) -> Optional[np.ndarray]:
        """Return w with shape (nw, N) or None."""
        return None

    def build_stage_constraints(self, environment, state, x, u, w, params) -> list:
        """Add constraints beyond dynamics and initial condition."""
        return []

    def build_objective(self, environment, state, x, u, w, params) -> cp.Expression:
        raise NotImplementedError

    def action_from_u0(self, u0: np.ndarray) -> Dict[str, Any]:
        """Map first control move to simulator action dict."""
        raise NotImplementedError

    # ----- internal: build + solve -----

    def _build_problem(self, model: LinearDiscreteModel, nx: int, nu: int, nw: int) -> None:
        N = self.N

        x = cp.Variable((nx, N + 1))
        u = cp.Variable((nu, N))
        w = cp.Parameter((nw, N)) if nw > 0 else None

        x0 = cp.Parameter(nx)

        # Model matrices as parameters (lets you update them if needed)
        A = cp.Parameter((nx, nx))
        B = cp.Parameter((nx, nu))
        E = cp.Parameter((nx, nw)) if nw > 0 else None
        g = cp.Parameter(nx)

        self._x, self._u, self._w = x, u, w
        self._params = {"x0": x0, "A": A, "B": B, "g": g}
        if nw > 0:
            self._params["E"] = E

        constraints = [x[:, 0] == x0]
        for k in range(N):
            if nw > 0:
                constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k] + E @ w[:, k] + g]
            else:
                constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k] + g]

        # Allow subclasses to add constraints/objective terms
        constraints += self.build_stage_constraints(None, None, x, u, w, self._params)
        obj = self.build_objective(None, None, x, u, w, self._params)

        self._problem = cp.Problem(cp.Minimize(obj), constraints)

    def get_action(self, state) -> Dict[str, Any]:
        # We assume environment is available through sensors/obs, but it's cleaner
        # if simulator passes env. Since it doesn't, we use the sensors obs already stored.
        # Better: store env reference in controller when loaded, but keeping minimal change:
        dt_s = float(state.time_step)
        now = state.time

        model = self.get_model(env, state)
        x0_val = self.get_current_state(env, state).astype(float).ravel()

        nx = x0_val.shape[0]
        nu = model.B.shape[1]
        nw = 0 if model.E is None else model.E.shape[1]

        if self._problem is None:
            self._build_problem(model, nx, nu, nw)

        # Update parameters
        self._params["x0"].value = x0_val
        self._params["A"].value = model.A
        self._params["B"].value = model.B
        self._params["g"].value = np.zeros(nx) if model.g is None else model.g

        if nw > 0:
            self._params["E"].value = model.E
            w_val = self.get_disturbance_forecast(env, state, now=now, dt_s=dt_s)
            if w_val is None:
                raise ValueError("Model expects disturbances (E not None) but no disturbance forecast provided.")
            if w_val.shape != (nw, self.N):
                raise ValueError(f"w forecast has shape {w_val.shape}, expected {(nw, self.N)}")
            self._w.value = w_val

        # Solve
        self._problem.solve(solver=self.solver, warm_start=self.warm_start)
        if self._problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MPC solve failed: {self._problem.status}")

        u0 = np.asarray(self._u.value)[:, 0].ravel()
        actions = self.action_from_u0(u0)
        self.previous_action = actions
        return actions
