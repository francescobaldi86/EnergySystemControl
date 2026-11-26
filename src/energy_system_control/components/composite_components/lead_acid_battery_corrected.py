from energy_system_control.components.composite_components.batteries import LeadAcidBattery


class LeadAcidBatteryCorrected(LeadAcidBattery):
    """
    Batteria al piombo-acido con limite di scarica coerente con capacita' ed erate e con riduzione
    della potenza di carica vicino al pieno (introdotta per test_9, banco BAE Secura PVV Solar).

    SOSTITUISCE (solo per test_9) due parti di LeadAcidBattery, in batteries.py:
        1. CORREZIONE: LeadAcidBattery.__init__ fissa self.charger.max_discharging_power = 0.636 kW,
           qualunque siano capacita' ed erate (0.636 kW e' 0.2C della batteria da 3.18 kWh di test_8).
           Qui il limite e' capacity * erate.
        2. NUOVA IPOTESI DI MODELLO: LeadAcidBattery ammette la carica massima (capacity * crate) fino
           al 100%. Una batteria al piombo reale accetta la corrente piena solo fino a circa l'80-85%,
           poi la corrente cala (fase di assorbimento). Qui, sopra SOC_taper, la potenza di carica
           massima cala linearmente fino a zero a SOC = 1. Con SOC_taper = 1.0 la riduzione e' disattivata.

    La correzione dell'autoscarica e' gia' stata fatta direttamente in BatteryPack.step
    (electric_storage.py), quindi qui non serve.

    Parameters
    ----------
    SOC_taper : float, optional
        Stato di carica oltre il quale la potenza di carica massima cala linearmente [-]. Defaults to 0.8.
    **kwargs
        Gli stessi di LeadAcidBattery (capacity, crate, erate, SOC_0, SOC_min, SOC_derate, ...).
        Per scegliere la scarica massima passare sempre erate esplicitamente: se manca, LeadAcidBattery
        lo pone uguale a crate.
    """

    def __init__(self, SOC_taper: float = 0.8, **kwargs):
        super().__init__(**kwargs)

        # 1. Correzione del limite di scarica (al posto di self.charger.max_discharging_power = 0.636)
        self.charger.max_discharging_power = self.battery_capacity * self.erate

        # 2. Riduzione della potenza di carica sopra SOC_taper
        self.SOC_taper = SOC_taper
        lead_acid_step = self.charger.step    # metodo gia' modificato da LeadAcidBattery

        def tapered_charger_step(state, action=None):
            if action is not None and action > 0 and self.SOC_taper < 1.0:
                soc = self.battery_pack.SOC
                if soc > self.SOC_taper:
                    taper = max(0.0, (1.0 - soc) / (1.0 - self.SOC_taper))
                    action = min(action, self.charger.max_charging_power * taper)
            return lead_acid_step(state, action)

        self.charger.step = tapered_charger_step

    def get_maximum_charge_power(self) -> float:
        """Potenza di carica massima [kW] allo stato di carica attuale (lato AC della batteria)."""
        soc = self.battery_pack.SOC
        if self.SOC_taper < 1.0 and soc > self.SOC_taper:
            return self.charger.max_charging_power * max(0.0, (1.0 - soc) / (1.0 - self.SOC_taper))
        return self.charger.max_charging_power
