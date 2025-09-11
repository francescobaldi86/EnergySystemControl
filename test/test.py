from EnergySystemControl.environments.base_environment import Environment, ElectricalNode
from EnergySystemControl.environments.utilities import HeatSource
from EnergySystemControl.environments.storage_units import SimpleStorage
from EnergySystemControl.environments.demands import ThermalLoss


def test():
    # One node: house thermal mass
    nodes = {"contatore": ElectricalNode("contatore"), }

    components = [
        HeatSource("heater", node="house", Q_W=2000.0),   # 2 kW heater
        ThermalLoss("loss", node="house", T_ambient=5.0, U_W_per_K=100.0)
    ]

    env = Environment(nodes=nodes, components=components, dt_s=60.0)  # dt = 60 s
    env.run(0.0, 3600.0 * 6)  # simulate 6 hours
    df_nodes, df_comps = env.to_dataframe()
    print(df_nodes.head())
    print(df_comps.head())

test()