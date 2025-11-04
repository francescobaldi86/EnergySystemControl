# tests/unit/test_control_rule_based.py
def test_rule_based_respects_device_limits(time_index):
    from energy_system_control import HeaterController
    ctrl = HeaterController(...)
    # Feed a state that would push outputs beyond limits
    out = ctrl.compute_action(state={...})
    assert abs(out["heater_kw"]) <= ctrl.max_kw
