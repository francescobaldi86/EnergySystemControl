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
                 minimum_time_off_between_activations_h: float | None = None,
                 minimum_time_on_between_deactivations_h: float | None = None):
        super().__init__(name, 
                         controlled_component, 
                         temperature_sensor, 
                         temperature_comfort, 
                         temperature_bandwidth,
                         minimum_time_off_between_activations_h,
                         minimum_time_on_between_deactivations_h)
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

        '''    
        if not hasattr(self, 'sensor_names'):
            self.sensor_names = {}
                
        self.sensor_names.update({
            'demand': electricity_demand_sensor,
            'PV_power': pv_power_sensor
        })
        '''
            
        self.net_power_activation = net_power_activation
        self.min_time_on_h = min_time_on_h
            
        self.last_activation_time = -float('inf')
        self.is_running = False
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
        if self.is_running:
            time_elapsed = state.time - self.last_activation_time
            if time_elapsed < self.min_time_on_h:
                action = 1.0
            else:
                action = desired_action
                if action == 0.0:
                    self.is_running = False
        else:
            action = desired_action
            if action == 1.0:
                self.is_running = True
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

        self.battery_charger_name = f"{battery_name}_charger"
        
        super().__init__(name=name,
                         controlled_components=[self.battery_charger_name, engine_name],
                         sensors={'demand': electricity_demand_sensor,
                                  'PV_power': pv_power_sensor,
                                  'battery_SOC': battery_soc_sensor})
        
        self.battery_name = battery_name
        self.engine_name = engine_name
        
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
        # Calcoliamo la potenza effettivamente generata dal motore in questo time-step
        engine_power = self.engine_p_el_design * engine_action
        
        # Il nuovo bilancio tiene conto della potenza aggiuntiva del motore.
        # net_balance > 0 significa surplus (batteria si carica), < 0 significa deficit (batteria si scarica)
        net_balance = pv_power + engine_power - demand
        
        return {
            self.engine_name: engine_action,
            self.battery_charger_name: net_balance
        }

class BatteryControllerAware(Controller):
    """
    Controllore della batteria che calcola l'azione leggendo la potenza dal fotovoltaico, 
    la domanda elettrica e la potenza generata dal motore a combustione.
    """
    def __init__(self, name: str, 
                 controlled_component: str, 
                 soc_sensor: str, 
                 demand_sensor: str, 
                 pv_sensor: str, 
                 engine_sensor: str):

        self.controlled_charger = f"{controlled_component}_charger"
        
        super().__init__(name=name,
                         controlled_components=[controlled_component, self.controlled_charger],
                         sensors={'SOC': soc_sensor, 
                                  'demand': demand_sensor, 
                                  'PV_power': pv_sensor, 
                                  'engine_power': engine_sensor})
        
        self.controlled_component = controlled_component

    def _compute_action(self, state: SimulationState):
        demand = abs(self.obs['demand'])
        pv_power = abs(self.obs['PV_power'])
        
        # Usiamo abs() per assicurarci di trattarla come grandezza positiva aggiuntiva, 
        # a prescindere dalla convenzione dei segni del sensore.
        engine_power = abs(self.obs['engine_power']) 
        
        # Bilancio netto per la batteria
        net_balance = pv_power + engine_power - demand
        
        return {self.controlled_charger: net_balance}