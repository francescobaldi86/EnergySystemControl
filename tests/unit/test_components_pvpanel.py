# tests/unit/test_components_pvpanel.py
import numpy as np
import pandas as pd

def test_pvpanel_linear_with_irradiance():
    from energy_system_control import PVPanel
    pv = PVPanel(area=10.0, efficiency=0.2)

    ghi = pd.Series([0.0, 200.0, 800.0])  # W/m2
    # expected DC power ~ area * eff * ghi (if that’s your model)
    expected = ghi * pv.area * pv.efficiency

    got = pv.power_from_irradiance(ghi)  # adapt to your method
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-6)
    assert (got >= 0).all()
