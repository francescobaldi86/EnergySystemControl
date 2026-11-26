from energy_system_control.components.base import Grid
from energy_system_control.components.grids.grids import GasGrid
from energy_system_control.sim.state import SimulationState


class LPGSupply(GasGrid):
    """
    Fornitura di GPL dalle bombole del rifugio (introdotta per test_9).

    E' un'utenza di bilanciamento: fornisce al motore DualFuelEngine tutto il GPL che richiede e
    permette di contarne i kg dai risultati della porta '<name>_fuel_port'.

    SOSTITUISCE (solo per test_9) i valori fissi impostati da GasGrid.set_inherited_port_values
    in grids.py (LHV = 47000 kJ/kg, density = 0.7 kg/m3), che non corrispondono al GPL:
        - LHV = 46055 kJ/kg, lo stesso usato nella pulizia dei dati delle bilance;
        - density = 2.0 kg/m3, GPL in fase gassosa (propano ~1.9, butano ~2.5 kg/m3).

    Parameters
    ----------
    name : str
        Nome del componente.
    LHV : float, optional
        Potere calorifico inferiore del GPL [kJ/kg]. Defaults to 46055.
    density : float, optional
        Densita' del GPL in fase gassosa [kg/m3]. Defaults to 2.0.
    """

    def __init__(self, name: str, LHV: float = 46055.0, density: float = 2.0):
        super().__init__(name, utility_type='fuel')
        self.LHV = LHV
        self.density = density

    def set_inherited_port_values(self, state: SimulationState):
        port = self.ports[self.port_name]
        port.T = state.environmental_data.temperature_cold_water
        port.LHV = self.LHV
        port.density = self.density
        return {self.port_name: {'T': port.T, 'LHV': self.LHV, 'density': self.density}}


class AtmosphericVent(Grid):
    """
    Sfiato del biogas in atmosfera (introdotto per test_9 al posto di EmergencyFlare).

    Nel rifugio non e' prevista una torcia: il biogas in eccesso puo' solo essere rilasciato incombusto.
    Questo componente non decide nulla: riceve dalla porta '<name>_flare_output' di BiogasStorageWithVent
    esattamente il biogas che non entra nel gasometro pieno (valvola a solo traboccamento) e permette di
    contarlo dai risultati della porta '<name>_biogas_port' (layer 'mass', 'volume', 'chemical_energy').

    Il metano rilasciato si ottiene moltiplicando la massa sfiatata per la frazione massica di metano
    del biogas; la CO2 equivalente moltiplicando il metano per il suo GWP.
    """

    def __init__(self, name: str):
        super().__init__(name, utility_type='biogas')
