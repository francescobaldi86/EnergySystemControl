from energy_system_control.sim.simulation_data import SimulationData
from dataclasses import dataclass
from typing import Any, List
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class SimulationResults:
    data: SimulationData
    time_step: float
    time_vector: np.array
    signal_registry_ports: Any
    signal_registry_controllers: Any
    signal_registry_sensors: Any

    def to_dataframe(self):
        return self.data.to_dataframe(self.time_vector, self.signal_registry_ports, self.signal_registry_controllers, self.signal_registry_sensors)

    def _get_cumulated_result(self, port_name: str, layer_name: str, scaling_factor: float = 1):
        col = self.signal_registry_ports.col_index(port_name, layer_name)
        return self.data.ports[:, col].sum() * self.time_step * scaling_factor

    def _get_cumulated_result_with_sign(self, port_name: str, layer_name: str, sign: str, scaling_factor: float = 1):
        col = self.signal_registry_ports.col_index(port_name, layer_name)
        match sign:
            case 'only positive':
                return self.data.ports[self.data.ports[:, col] >= 0.0, col].sum() * self.time_step * scaling_factor
            case 'only negative':
                return -self.data.ports[self.data.ports[:, col] <= 0.0, col].sum() * self.time_step * scaling_factor

    def get_cumulated_electricity(self, port_name: str, unit: str = "kWh", sign: str = "net"):
        match unit:
            case "kWh":
                scaling_factor = 1 / 3_600
            case "MWh":
                scaling_factor = 1 / 3_600_000
            case _:
                raise ValueError(unit)

        match sign:
            case "net":
                return self._get_cumulated_result(port_name, "electricity", scaling_factor)
            case "only positive" | "only negative":
                return self._get_cumulated_result_with_sign(port_name, "electricity", sign, scaling_factor)


    def get_DHW_temperature_comfort_index(self, port_name, boundary):
        # Measures the temperature-based comfort given a condition
        condition = abs(self.data.ports[port_name].flow['mass']) > 1e-6
        return sum(self.data.ports[port_name].T[condition] >= boundary) / len(self.simulation_data.ports[port_name].T[condition])


    
    def get_boundary_index(self, sensor_name: str, boundary: float, condition: str):
        # Calculates the fraction of time over the simulation a certain value was above or below a certain boundary
        match condition:
            case "gt" | ">" | ">=":
                return sum(self.data.sensors[:, self.signal_registry_sensors.col_index(sensor_name, "")] >= boundary) / len(self.data.sensors[:, self.signal_registry_sensors.col_index(sensor_name, "")])
            case "lt" | "<" | "<=":
                return sum(self.data.sensors[:, self.signal_registry_sensors.col_index(sensor_name, "")] <= boundary) / len(self.data.sensors[:, self.signal_registry_sensors.col_index(sensor_name, "")])


    def plot_sensors(self, sensors: str | List[str] | None= None, labels: str | List[str] | None = None, ylabel: str | None= None, filename: str | None = None, reference_value: float | None = None):
        # Plots the measured values of a sensor over time
        fig, ax = plt.subplots(figsize=(10, 6))
        if isinstance(sensors, str):
            sensors_list = [sensors]
            labels_list = [labels]
        elif isinstance(sensors, list):
            sensors_list = sensors
            labels_list = labels
        for id, sensor in enumerate(sensors_list):
            self._plot_sensor(ax, sensor, labels_list[id])
        if reference_value:
            ax.hlines([reference_value], xmin = ax.get_xlim()[0], xmax = ax.get_xlim()[1], colors = ['red'], linestyles=['solid'])
        ax.set_xlabel('Time [h]')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid()
        # If filename is provided, save it there (for now, no folder)
        if filename:
            fig.savefig(filename)
        return fig, ax

    def _plot_sensor(self, ax, sensor_name: str, label: str | None = None):
        # Plots the measured values of a sensor over time
        col = self.signal_registry_sensors.col_index(sensor_name, "")
        label = label if label else sensor_name
        ax.plot(self.time_vector/3600, self.data.sensors[:,col], label=label)
        
    def plot_temperature_sensors(self, sensors: str | List[str] | None= None, labels: str | List[str] | None = None, ylabel: str | None= None, filename: str | None = None, comfort_temperature: float | None = None):
        fig, ax = self.plot_sensors(sensors, labels, 'Temperature [K]', filename, comfort_temperature)
        return fig, ax

    def plot_electric_power_sensors(
        self,
        power_sensors: str | List[str],
        SOC_sensor: str | None = None,
        labels: str | List[str] | None = None,
        filename: str | None = None
    ):
        fig, ax = self.plot_sensors(power_sensors, labels, 'Power [kW]', None)

        # Secondary axis for SOC
        if SOC_sensor is not None:
            ax2 = ax.twinx()

            col = self.signal_registry_sensors.col_index(SOC_sensor, "")
            soc_values = self.data.sensors[:, col]

            ax2.plot(
                self.time_vector / 3600,
                soc_values,
                color='black',
                linestyle='--',
                label=SOC_sensor if labels is None else f"{SOC_sensor}"
            )

            ax2.set_ylabel('State of Charge [-]')
            ax2.set_ylim(0, 1)  # assuming SOC is normalized

            # Combine legends from both axes
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2)

        ax.set_xlabel('Time [h]')
        ax.grid()

        if filename:
            fig.savefig(filename)

        return fig, ax

###############  Update Nina Nardella 04/05/2026
    
    # ==========================================
    # AGGIUNTA: CALCOLO CICLI SOC (ADATTATO)
    # ==========================================
    def get_SOC_cycles_advanced(self, sensor_name: str, smoothing_factor: float = 0.5, plot: bool = False):
        """Calcolo  dei cicli SOC basato su smoothing + peak detection"""

        from scipy.interpolate import UnivariateSpline
        from scipy.signal import find_peaks

        # ==========================================
        # PRENDO SERIE TEMPORALE SOC
        # ==========================================
        col = self.signal_registry_sensors.col_index(sensor_name, "")
        soc = self.data.sensors[:, col]

        # tempo in minuti (nel codice originale)
        tempo = self.time_vector / 60

        # rimuovo eventuali NaN
        mask = ~np.isnan(soc)
        soc = soc[mask]
        tempo = tempo[mask]

        if len(soc) < 10:
            return {"Numero cicli": 0}

        # ==========================================
        # SMOOTHING 
        # ==========================================
        spline = UnivariateSpline(tempo, soc, s=len(soc) * smoothing_factor)
        tempo_dense = np.linspace(tempo.min(), tempo.max(), len(soc) * 10)
        soc_smooth = spline(tempo_dense)

        # ==========================================
        # TROVA PICCHI E VALLI
        # ==========================================
        picchi_idx, _ = find_peaks(soc_smooth)
        valli_idx, _ = find_peaks(-soc_smooth)

        picchi_tempo = tempo_dense[picchi_idx]

        # funzione per tornare al valore originale
        def valore_originale(t):
            idx = np.argmin(np.abs(tempo - t))
            return soc[idx]

        # ==========================================
        # SEGMENTAZIONE CICLI 
        # ==========================================
        minimi_segmentati = []
        massimi_segmento = np.concatenate(([tempo_dense[0]], picchi_tempo, [tempo_dense[-1]]))

        for i in range(len(massimi_segmento) - 1):
            inizio = massimi_segmento[i]
            fine = massimi_segmento[i + 1]

            min_in_range_idx = [idx for idx in valli_idx if inizio <= tempo_dense[idx] <= fine]

            if min_in_range_idx:
                min_idx = min_in_range_idx[np.argmin(soc_smooth[min_in_range_idx])]
                minimi_segmentati.append(min_idx)

        valli_segmentate_tempo = tempo_dense[minimi_segmentati]

        # ==========================================
        # COSTRUZIONE CICLI
        # ==========================================
        cicli_finali = []

        for i in range(len(valli_segmentate_tempo) - 1):
            t_min1 = valli_segmentate_tempo[i]
            t_min2 = valli_segmentate_tempo[i + 1]

            massimi_in_mezzo = [t for t in picchi_tempo if t_min1 < t < t_min2]

            if not massimi_in_mezzo:
                continue

            val_min1 = valore_originale(t_min1)
            val_min2 = valore_originale(t_min2)

            M = max(val_min1, val_min2)

            cicli_finali.append((t_min1, t_min2, M))

        numero_cicli = len(cicli_finali)

        # ==========================================
        # PLOT 
        # ==========================================
        if plot:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10,6))
            plt.plot(tempo, soc, label="SOC Originale")
            plt.plot(tempo_dense, soc_smooth, label="SOC Smoothed")

            plt.plot(picchi_tempo, soc_smooth[picchi_idx], 'bo', label="Peaks")
            plt.plot(valli_segmentate_tempo, soc_smooth[minimi_segmentati], 'ro', label="Valleys")

            plt.xlabel("Tempo [min]")
            plt.ylabel("SOC [-]")
            plt.title("Analisi cicli SOC")
            plt.legend()
            plt.grid()
            plt.show()

        # ==========================================
        # OUTPUT
        # ==========================================
        return {
            "Numero cicli": numero_cicli,
            "Cicli": cicli_finali
        }