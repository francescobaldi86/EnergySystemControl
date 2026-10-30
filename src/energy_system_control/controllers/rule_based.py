from energy_system_control.controllers.base import HeaterControllerWithBandwidth
from energy_system_control.sim.state import SimulationState
from energy_system_control.helpers import *
from energy_system_control.controllers.base import Controller

class HeatPumpRuleBasedController(HeaterControllerWithBandwidth):
    """
    Rule-based heat-pump controller that combines temperature and PV control.

    The controller first applies the bandwidth logic from
    :class:`HeaterControllerWithBandwidth` to keep the storage temperature near
    the comfort temperature. It then activates the heat pump whenever measured
    PV power is at least ``power_PV_activation`` and the storage temperature is
    below ``max_storage_temperature_for_activation``. This PV-based activation
    overrides the inherited bandwidth action, allowing available PV power to be
    used for heating while the storage remains below its safety limit.

    Parameters
    ----------
    name : str
        Name of the controller.
    controlled_component : str
        Name of the heat pump or heater controlled by this controller.
    temperature_sensor : str
        Name of the sensor measuring storage temperature.
    PV_power_sensor : str
        Name of the sensor measuring available PV power.
    temperature_comfort : float
        Target storage temperature in degrees Celsius.
    temperature_bandwidth : float
        Allowed temperature bandwidth above ``temperature_comfort``.
    power_PV_activation : float
        Minimum PV power required for PV-based heat-pump activation.
    max_storage_temperature_for_activation : float, default=60
        Maximum storage temperature for PV-based activation, in degrees
        Celsius.
    """
    def __init__(self, name, 
                 controlled_component: str, 
                 temperature_sensor: str, 
                 PV_power_sensor: str, 
                 temperature_comfort: float, 
                 temperature_bandwidth: float, 
                 power_PV_activation: float, 
                 max_storage_temperature_for_activation: float = 60,
                 minimum_time_off_h: float | None = None,
                 minimum_time_on_h: float | None = None):
                 
        super().__init__(name, 
                         controlled_component, 
                         temperature_sensor, 
                         temperature_comfort, 
                         temperature_bandwidth,
                         minimum_time_off_h=minimum_time_off_h,
                         minimum_time_on_h=minimum_time_on_h)
                         
        self.sensor_names.update({'PV power': PV_power_sensor})
        self.max_storage_temperature_for_activation = C2K(max_storage_temperature_for_activation)
        self.power_PV_activation = power_PV_activation
        self.PV_power_sensor_name = PV_power_sensor

    def _compute_action(self, state = SimulationState):
        # The principle of this controller is: 
        # - It tries to keep the temperature within limits, thus working as a "standard" bandwidth controller
        # - However, it also measures the power 
        power_PV = self.obs['PV power']
        if power_PV >= self.power_PV_activation and self.obs['Storage temperature'] < self.max_storage_temperature_for_activation:
            external_input = 1
        else:
            external_input = 0
        action = super()._compute_action(state, external_input)
        self.previous_action = action
        return action

class EngineRuleBasedController(Controller):
    """
    calcola il deficit di potenza elettrica come la differenza tra la domanda e la produzione di potenza fotovoltaica, entramnbi letti come valori positivi
    se il deficit supera una certa soglia allora il motore si accende. la soglia di è necessaria per evitare di avviare il motore nel caso di bassissimi valori del deficit
    si introduce anche un tempo minimo di accensione 
    """
    def __init__(self, name: str, 
                controlled_component: str, 
                electricity_demand_sensor: str, 
                pv_power_sensor: str,
                battery_soc_sensor: str, 
                net_power_activation: float,
                min_soc_activation: float = 0.2,
                min_time_on_h: float = 1.0): 
        
        super().__init__(name=name,
                         controlled_components=[controlled_component],
                         sensors={'demand': electricity_demand_sensor,
                                  'PV_power': pv_power_sensor,
                                  'battery_SOC': battery_soc_sensor})
        
        self.controlled_component = controlled_component
        self.electricity_demand_sensor = electricity_demand_sensor
        self.min_soc_activation = min_soc_activation
        self.pv_power_sensor = pv_power_sensor
            
        self.net_power_activation = net_power_activation
        self.min_time_on_h = min_time_on_h
            
        self.last_activation_time = -float('inf')
        self.engine_is_running = False
        self.previous_action = 0.0

    def _compute_action(self, state: SimulationState):
        # Valori dai sensori (ora attesi entrambi come positivi in modulo)
        demand = abs(self.obs['demand'])      
        pv_power = abs(self.obs['PV_power'])
        soc = self.obs['battery_SOC']  
        
        # Deficit netto: potenza richiesta non coperta dal fotovoltaico
        net_deficit = demand - pv_power
        
        # 1. Determinazione azione teorica: il motore si accende quando il deficit è alto e quando la batteria è al di sotto della soglia stabilita
        if net_deficit >= self.net_power_activation and soc <= self.min_soc_activation:
            desired_action = 1.0
        else:
            desired_action = 0.0
            
        # 2. Logica tempo minimo di accensione (override)
        if self.engine_is_running:
            time_elapsed = state.time - self.last_activation_time
            if time_elapsed < self.min_time_on_h:
                action = 1.0
            else:
                action = desired_action
                if action == 0.0:
                    self.engine_is_running = False
        else:
            action = desired_action
            if action == 1.0:
                self.engine_is_running = True
                self.last_activation_time = state.time
                
        self.previous_action = action
        return {self.controlled_component: action}

class ChargeControllerWithEngine(Controller):
    """
    Controllore unificato che gestisce sia il motore a combustione interna sia la batteria.
    Calcola prima l'azione del motore e aggiorna il bilancio di potenza netto per la batteria.
    """
    def __init__(self, name: str, 
                 battery_name: str, 
                 engine_name: str, 
                 battery_soc_sensor: str, 
                 electricity_demand_sensor: str, 
                 pv_power_sensor: str,
                 net_power_activation: float,
                 engine_p_el_design: float,
                 min_soc_activation: float = 0.2,
                 min_time_on_h: float = 1.0): 

        # Creiamo il nome del caricatore internamente
        self.battery_charger_name = f"{battery_name}_charger"
        self.engine_name = engine_name
        self.battery_name = battery_name
        
        super().__init__(name=name,
                         controlled_components=[self.battery_charger_name, self.engine_name],
                         sensors={'demand': electricity_demand_sensor,
                                  'PV_power': pv_power_sensor,
                                  'battery_SOC': battery_soc_sensor})
        
        self.net_power_activation = net_power_activation
        self.min_soc_activation = min_soc_activation
        self.min_time_on_h = min_time_on_h
        self.engine_p_el_design = engine_p_el_design
            
        self.last_activation_time = -float('inf')
        self.is_running = False

    def _compute_action(self, state: SimulationState):
        demand = abs(self.obs['demand'])      
        pv_power = abs(self.obs['PV_power'])
        soc = self.obs['battery_SOC']  
        
        net_deficit = demand - pv_power
        
        # 1. Logica di accensione del Motore
        if net_deficit >= self.net_power_activation and soc <= self.min_soc_activation:
            desired_engine_action = 1.0
        else:
            desired_engine_action = 0.0
            
        if self.is_running:
            if (state.time - self.last_activation_time) < self.min_time_on_h:
                engine_action = 1.0
            else:
                engine_action = desired_engine_action
                if engine_action == 0.0:
                    self.is_running = False
        else:
            engine_action = desired_engine_action
            if engine_action == 1.0:
                self.is_running = True
                self.last_activation_time = state.time

        # 2. Logica di carica/scarica della Batteria
        engine_power = self.engine_p_el_design * engine_action
        net_balance = pv_power + engine_power - demand
        
        return {
            self.engine_name: engine_action,
            self.battery_charger_name: net_balance
        }


class BatteryControllerAware(Controller):
    """
    Controllore della batteria che calcola l'azione leggendo la potenza dal fotovoltaico, 
    la domanda elettrica e la potenza generata dal motore a combustione.
    Include un limitatore dinamico custom per evitare inversioni repentine tra carica e scarica.
    """
    def __init__(self, name: str, 
                 controlled_component: str, 
                 soc_sensor: str, 
                 demand_sensor: str, 
                 pv_sensor: str, 
                 engine_sensor: str,
                 min_time_charge_h: float = 0.0,      # Nuovo parametro
                 min_time_discharge_h: float = 0.0):  # Nuovo parametro

        self.controlled_charger = f"{controlled_component}_charger"
        
        super().__init__(name=name,
                         controlled_components=[controlled_component, self.controlled_charger],
                         sensors={'SOC': soc_sensor, 
                                  'demand': demand_sensor, 
                                  'PV_power': pv_sensor, 
                                  'engine_power': engine_sensor})
        
        self.controlled_component = controlled_component
        
        # Conversione dei tempi in secondi
        self.min_time_charge = min_time_charge_h * 3600
        self.min_time_discharge = min_time_discharge_h * 3600
        
        # Variabili di memoria per lo stato continuo
        self.current_mode = 0  # 1 = Carica, -1 = Scarica, 0 = Iniziale
        self.time_in_current_mode = 0.0

    def _compute_action(self, state: SimulationState):
        demand = abs(self.obs['demand'])
        pv_power = abs(self.obs['PV_power'])
        engine_power = abs(self.obs['engine_power']) 
        
        # Bilancio netto per la batteria
        net_balance = pv_power + engine_power - demand
        
        return {self.controlled_charger: net_balance}

    def check_dynamic_limitations(self, state: SimulationState, action: dict):
        proposed_power = action[self.controlled_charger]
        
        # 1. Analisi della volontà teorica (in che modalità vorrebbe entrare il sistema)
        proposed_mode = 0
        if proposed_power > 0:
            proposed_mode = 1
        elif proposed_power < 0:
            proposed_mode = -1

        # 2. Controllo dei vincoli temporali di inversione
        if proposed_mode == 1 and self.current_mode == -1:
            # Vuole invertire da scarica a carica
            if self.time_in_current_mode < self.min_time_discharge:
                proposed_power = 0.0  # Azione non permessa: mettiamo la batteria in pausa
                proposed_mode = -1    # Manteniamo la memoria del ciclo di scarica in corso
                
        elif proposed_mode == -1 and self.current_mode == 1:
            # Vuole invertire da carica a scarica
            if self.time_in_current_mode < self.min_time_charge:
                proposed_power = 0.0  
                proposed_mode = 1     
        
        # 3. Aggiornamento timer (sfruttando state.time_step come in base.py) e modalità
        if proposed_mode != self.current_mode and proposed_mode != 0:
            # Se è un'inversione permessa o la primissima attivazione, salviamo il nuovo stato
            self.current_mode = proposed_mode
            self.time_in_current_mode = state.time_step 
        else:
            # Se siamo nella stessa modalità (o in pausa forzata che eredita il segno precedente), il tempo avanza
            self.time_in_current_mode += state.time_step
            
        action[self.controlled_charger] = proposed_power
        return action

class EngineOnOffController(Controller):
    def __init__(self, name: str, engine_name: str, demand_sensor: str, storage_sensor: str, 
                 biogas_soc_on_threshold: float = 0.80, biogas_soc_off_threshold: float = 0.20):
        super().__init__(name=name,
                         controlled_components=[engine_name],
                         sensors={'demand': demand_sensor,
                                  'biogas_SOC': storage_sensor})
        self.engine_name = engine_name
        self.biogas_soc_on_threshold = biogas_soc_on_threshold
        self.biogas_soc_off_threshold = biogas_soc_off_threshold
        self.engine_is_running = False  # Memoria dello stato del motore

    def _compute_action(self, state):
        demand = abs(self.obs['demand'])
        biogas_soc = self.obs['biogas_SOC']
        
        if not self.engine_is_running:
            if demand > 0 and biogas_soc >= self.biogas_soc_on_threshold:
                self.engine_is_running = True
        else:
            if biogas_soc <= self.biogas_soc_off_threshold:
                self.engine_is_running = False
                
        action = 1.0 if self.engine_is_running else 0.0
        return {self.engine_name: action}

from energy_system_control.controllers.base import Controller
from energy_system_control.sim.state import SimulationState

class BiogasEngineRuleBasedController(Controller):
    """
    Controllore unificato con regolazione continua della potenza del motore a biogas.
    Modula il fattore di carico (alfa) in funzione del deficit elettrico e del SOC della batteria.
    """
    def __init__(self, name: str, 
                 battery_name: str, 
                 engine_name: str, 
                 battery_soc_sensor: str, 
                 biogas_soc_sensor: str,
                 building_demand_sensor: str,
                 hp_demand_sensor: str,
                 pv_power_sensor_est: str,
                 pv_power_sensor_ovest: str,
                 net_power_activation: float,
                 engine_p_el_design: float,
                 min_battery_soc_activation: float = 0.2,
                 target_battery_soc: float = 0.95,          # SOC desiderato per la batteria in ricarica
                 max_battery_charge_power: float = 0.6,    # Potenza massima [kW] destinata alla ricarica
                 min_engine_load_factor: float = 0.15,     # Carico minimo tecnico (es. 15% di P_el_design)
                 biogas_soc_on_threshold: float = 0.8,
                 biogas_soc_off_threshold: float = 0.2,
                 min_time_on_h: float = 1.0,
                 min_time_off_h: float = 0.0): 

        self.battery_charger_name = f"{battery_name}_charger"

        super().__init__(name=name,
                         controlled_components=[self.battery_charger_name, engine_name],
                         sensors={'building_demand': building_demand_sensor, 
                                  'hp_demand': hp_demand_sensor,             
                                  'PV_power_est': pv_power_sensor_est,
                                  'PV_power_ovest': pv_power_sensor_ovest,
                                  'battery_SOC': battery_soc_sensor,
                                  'biogas_SOC': biogas_soc_sensor},
                         minimum_time_off_h={engine_name: min_time_on_h},
                         minimum_time_on_h={engine_name: min_time_off_h},
                         minimum_load = {engine_name: min_engine_load_factor})
        
        self.battery_name = battery_name
        self.engine_name = engine_name
        
        self.net_power_activation = net_power_activation
        self.min_battery_soc_activation = min_battery_soc_activation
        self.target_battery_soc = target_battery_soc
        self.max_battery_charge_power = max_battery_charge_power
        self.min_engine_load_factor = min_engine_load_factor
        self.biogas_soc_on_threshold = biogas_soc_on_threshold
        self.biogas_soc_off_threshold = biogas_soc_off_threshold
        
        self.engine_p_el_design = engine_p_el_design
        self.engine_is_running = False

    def _compute_modulated_load(self, net_deficit: float, battery_soc: float) -> float:
        """
        Calcola il fattore di carico parziale alfa compreso tra [min_engine_load_factor, 1.0]
        """
        # following the instantaneous net deficit
        required_power = max(0.0, net_deficit)
        
        # battery charging: adding power proportional to the SOC error if below target
        if battery_soc < self.target_battery_soc:
            soc_error = (self.target_battery_soc - battery_soc) / self.target_battery_soc
            required_power += self.max_battery_charge_power * soc_error
            
        # load factor: converting required power to a load factor between min_engine_load_factor and 1.0
        load_factor = required_power / self.engine_p_el_design
        return min(max(load_factor, self.min_engine_load_factor), 1.0)

    def _compute_action(self, state: SimulationState, **kwargs):
        # read sensors and calculate net_deficit
        building_demand = abs(self.obs.get('building_demand', 0.0))
        hp_demand = abs(self.obs.get('hp_demand', 0.0))
        total_demand = building_demand + hp_demand 
        
        pv_power_est = abs(self.obs['PV_power_est'])
        pv_power_ovest = abs(self.obs['PV_power_ovest'])
        total_pv_power = pv_power_est + pv_power_ovest

        battery_soc = self.obs['battery_SOC']  
        biogas_soc = self.obs['biogas_SOC']
        
        net_deficit = total_demand - total_pv_power

        # engine on/off logic with power modulation
        if not self.engine_is_running:
            # start-up conditions
            if (net_deficit >= self.net_power_activation and 
                battery_soc <= self.min_battery_soc_activation and 
                biogas_soc >= self.biogas_soc_on_threshold):
                engine_action = self._compute_modulated_load(net_deficit, battery_soc)
            else:
                engine_action = 0.0
        else:
            # shut-down and maintenance conditions
            if biogas_soc <= self.biogas_soc_off_threshold:
                engine_action = 0.0
            elif net_deficit < self.net_power_activation and battery_soc >= self.target_battery_soc:
                engine_action = 0.0
            else:
                engine_action = self._compute_modulated_load(net_deficit, battery_soc)
            
        # preparation of the desired (theoretical) action
        action = {
            self.engine_name: engine_action,
            self.battery_charger_name: self.previous_action.get(self.battery_charger_name, 0.0)
        }
        
        return action

    def get_action(self, state: SimulationState, **kwargs):
        # calculate the desired theoretical action
        desired_action = self._compute_action(state, **kwargs)
        
        # apply dynamic limitations (minimum ON/OFF times)
        limited_action = self.check_dynamic_limitations(state, desired_action)
        
        # extract the final engine action post-limitations
        final_engine_action = limited_action[self.engine_name]
        
        # update engine state based on the actual action
        self.engine_is_running = (final_engine_action > 0.0)

        # calculate the final battery output
        building_demand = abs(self.obs.get('building_demand', 0.0))
        hp_demand = abs(self.obs.get('hp_demand', 0.0))
        total_demand = building_demand + hp_demand 
        
        pv_power_est = abs(self.obs['PV_power_est'])
        pv_power_ovest = abs(self.obs['PV_power_ovest'])
        total_pv_power = pv_power_est + pv_power_ovest

        # update balance for the battery controller using the limited action
        engine_power = self.engine_p_el_design * final_engine_action
        net_balance = total_pv_power + engine_power - total_demand
        
        limited_action[self.battery_charger_name] = net_balance
        
        # save and return the final action
        self.previous_action = limited_action
        return limited_action