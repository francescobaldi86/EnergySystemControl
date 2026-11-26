from typing import List
from energy_system_control.controllers.base import Controller
from energy_system_control.sim.state import SimulationState


class DualFuelEngineRuleBasedController(Controller):
    """
    Controllore unificato del motore bi-fuel (DualFuelEngine) e della batteria (introdotto per test_9).

    Ha la stessa struttura di BiogasEngineRuleBasedController (rule_based.py): controlla il caricatore
    della batteria e il motore, e la batteria riceve il bilancio PV + motore - domanda. Rispetto a quello:

    SOSTITUISCE (solo per test_9) le seguenti parti di BiogasEngineRuleBasedController:
        1. La domanda elettrica: BiogasEngineRuleBasedController somma alla domanda dell'edificio la
           pompa di calore, letta da state.control_actions['heat_pump']. In test_9 la pompa di calore
           non c'e' (la lettura varrebbe 0 senza errori) e le utenze sono tre: rifugio, cucina a
           induzione, caldaia elettrica. Qui la domanda e' la somma dei sensori in demand_sensors.
        2. La logica del motore: BiogasEngineRuleBasedController accende il motore in base a deficit
           elettrico e SOC della batteria, e ne modula il carico con max_battery_charge_power = 0.6 kW
           (valore pensato per la batteria da 3.18 kWh). In test_9 il rifugio e' collegato alla rete, che
           assorbe il surplus e copre i deficit, quindi il motore segue il biogas:
             - si accende quando il SOC del gasometro raggiunge biogas_soc_on_threshold;
             - lavora al fattore di carico engine_load_factor (di norma quello di massimo rendimento, 0.75);
             - si spegne quando il SOC del gasometro scende a biogas_soc_off_threshold, ma solo dopo il
               tempo minimo di accensione (min_time_on_h). Nel frattempo Controller.check_dynamic_limitations
               lo tiene acceso al carico minimo (min_engine_load_factor) e DualFuelEngine completa il
               fabbisogno con il GPL, perche' il gasometro e' sotto la riserva minima.
           Per avere questo comportamento il min_biogas_soc del DualFuelEngine va posto uguale a
           biogas_soc_off_threshold.

    Il tempo minimo di accensione funziona come tale solo con la correzione di Controller.__init__
    (controllers/base.py), dove minimum_time_on e minimum_time_off erano invertiti.

    Parameters
    ----------
    name : str
        Nome del controllore.
    battery_name : str
        Nome della batteria: il controllore comanda il suo caricatore '<battery_name>_charger'.
    engine_name : str
        Nome del DualFuelEngine.
    biogas_soc_sensor : str
        Sensore dello stato di carica del gasometro.
    demand_sensors : list of str
        Sensori di potenza delle utenze elettriche (porte AC dell'inverter).
    pv_power_sensor_est, pv_power_sensor_ovest : str
        Sensori di potenza dei due campi fotovoltaici.
    engine_p_el_design : float
        Potenza elettrica nominale del motore [kW].
    engine_load_factor : float, optional
        Fattore di carico del motore quando e' acceso per bruciare biogas [-]. Defaults to 0.75.
    min_engine_load_factor : float, optional
        Carico minimo tecnico, usato quando il motore e' tenuto acceso dal tempo minimo [-]. Defaults to 0.15.
    biogas_soc_on_threshold, biogas_soc_off_threshold : float, optional
        Soglie di accensione e spegnimento sul SOC del gasometro [-]. Defaults to 0.8 e 0.2.
    min_time_on_h, min_time_off_h : float, optional
        Tempi minimi di accensione e di spegnimento del motore [h]. Defaults to 0.75 e 0.0.
    biogas_soc_tolerance : float, optional
        Tolleranza sulla soglia di spegnimento [-]. Quando il gasometro arriva alla riserva minima il
        DualFuelEngine preleva solo il biogas sopra la riserva, quindi il SOC resta appena sopra la soglia
        (di quanto entra dal digestore in un passo) e non la raggiungerebbe mai. Defaults to 0.01.
    """

    def __init__(self, name: str,
                 battery_name: str,
                 engine_name: str,
                 biogas_soc_sensor: str,
                 demand_sensors: List[str],
                 pv_power_sensor_est: str,
                 pv_power_sensor_ovest: str,
                 engine_p_el_design: float,
                 engine_load_factor: float = 0.75,
                 min_engine_load_factor: float = 0.15,
                 biogas_soc_on_threshold: float = 0.8,
                 biogas_soc_off_threshold: float = 0.2,
                 min_time_on_h: float = 0.75,
                 min_time_off_h: float = 0.0,
                 biogas_soc_tolerance: float = 0.01):

        self.battery_charger_name = f"{battery_name}_charger"
        self.demand_keys = [f'demand_{i}' for i in range(len(demand_sensors))]

        sensors = {key: sensor for key, sensor in zip(self.demand_keys, demand_sensors)}
        sensors.update({'PV_power_est': pv_power_sensor_est,
                        'PV_power_ovest': pv_power_sensor_ovest,
                        'biogas_SOC': biogas_soc_sensor})

        super().__init__(name=name,
                         controlled_components=[self.battery_charger_name, engine_name],
                         sensors=sensors,
                         minimum_time_off_h={engine_name: min_time_off_h},
                         minimum_time_on_h={engine_name: min_time_on_h},
                         minimum_load={engine_name: min_engine_load_factor})

        self.battery_name = battery_name
        self.engine_name = engine_name
        self.engine_p_el_design = engine_p_el_design
        self.engine_load_factor = engine_load_factor
        self.min_engine_load_factor = min_engine_load_factor
        self.biogas_soc_on_threshold = biogas_soc_on_threshold
        self.biogas_soc_off_threshold = biogas_soc_off_threshold
        self.biogas_soc_tolerance = biogas_soc_tolerance
        self.engine_is_running = False

    def _read_power_balance(self):
        """Restituisce (domanda totale, produzione fotovoltaica totale) in kW, entrambe positive."""
        total_demand = sum(abs(self.obs.get(key, 0.0) or 0.0) for key in self.demand_keys)
        total_pv_power = abs(self.obs['PV_power_est']) + abs(self.obs['PV_power_ovest'])
        return total_demand, total_pv_power

    def _compute_action(self, state: SimulationState, **kwargs):
        biogas_soc = self.obs['biogas_SOC']

        # logica di accensione e spegnimento del motore, guidata dal gasometro
        if not self.engine_is_running:
            engine_action = self.engine_load_factor if biogas_soc >= self.biogas_soc_on_threshold else 0.0
        else:
            biogas_exhausted = biogas_soc <= self.biogas_soc_off_threshold + self.biogas_soc_tolerance
            engine_action = 0.0 if biogas_exhausted else self.engine_load_factor

        # azione teorica (la batteria viene calcolata dopo i vincoli temporali sul motore)
        return {
            self.engine_name: engine_action,
            self.battery_charger_name: self.previous_action.get(self.battery_charger_name, 0.0)
        }

    def get_action(self, state: SimulationState, **kwargs):
        # azione teorica
        desired_action = self._compute_action(state, **kwargs)

        # vincoli dinamici (tempi minimi di accensione/spegnimento, carico minimo)
        limited_action = self.check_dynamic_limitations(state, desired_action)

        # azione finale del motore e aggiornamento del suo stato
        final_engine_action = limited_action[self.engine_name]
        self.engine_is_running = (final_engine_action > 0.0)

        # bilancio per la batteria: positivo = carica, negativo = scarica. Il resto va alla rete
        total_demand, total_pv_power = self._read_power_balance()
        engine_power = self.engine_p_el_design * final_engine_action
        limited_action[self.battery_charger_name] = total_pv_power + engine_power - total_demand

        self.previous_action = limited_action
        return limited_action
