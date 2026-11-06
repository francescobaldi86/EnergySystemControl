"""
validation_plots.py
===================
Diagrammi di validazione per la simulazione test_8 (energy_system_control).

Stile: replica quello di results.py (figsize 10x6, griglia, legenda, asse x "Time [h]"
= ore dall'inizio della simulazione, asse secondario tratteggiato nero per i SOC,
linea di riferimento rossa continua). Dove serviva piu' chiarezza ho usato alcune
varianti (aree impilate per il bilancio di potenza, due pannelli per il termico).

Uso
---
    python validation_plots.py test_8_results.xlsx [cartella_output]

oppure, da codice, partendo dai DataFrame restituiti da results.to_dataframe():

    from validation_plots import make_all
    make_all(df_ports, df_sensors, df_controllers, "figure_validazione")

Convenzioni di segno (porte): + = energia assorbita dal componente, - = erogata.
Nei grafici di potenza le fonti sono positive e i carichi negativi.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --------------------------------------------------------------------------------------
# Parametri del caso test_8 (da test_8.py / librerie): modificare se cambia la simulazione
# --------------------------------------------------------------------------------------
PARAMS = dict(
    P_el_design=30.0,                      # kW, motore
    eta_coeffs=(-0.3167, 0.4752, 0.0064),  # a2, a1, a0 della curva eta(alfa)
    min_load=0.15,                         # BiogasEngineRuleBasedController.min_engine_load_factor
    min_time_on_h=0.75,
    biogas_soc_on=0.8, biogas_soc_off=0.2, flare_soc_activation=0.98,
    battery_capacity=3.18, battery_crate=0.2, battery_SOC_min=0.3, battery_SOC_derate=0.6,
    tank_comfort_C=40.0,
)

# inizio (00:00) delle finestre da 48 h, una per mese simulato
WINDOWS = {
    "luglio": "2026-07-27",
    "agosto": "2026-08-14",
    "settembre": "2026-09-15",
}
ZOOM_ENGINE = ("2026-07-25 13:30", "2026-07-25 14:15")   # zoom sul primo ciclo del motore
PV_DAY = None   # None = giornata piu' soleggiata del periodo (solo giorni completi); oppure "2026-07-26"

DT_S = 60.0   # passo [s]
CP = 4.186    # kJ/kg/K

C_ENG = "biogas_engine_rule_based_controller:backup_engine"
C_BAT = "biogas_engine_rule_based_controller:battery_charger"
C_FLARE = "flare_controller:biogas_flare"
C_HP = "heat_pump_controller:heat_pump"


# --------------------------------------------------------------------------------------
# utilita'
# --------------------------------------------------------------------------------------
def kwh(series):
    return float(series.sum() * DT_S / 3600.0)


def eta_poly(alpha):
    a2, a1, a0 = PARAMS["eta_coeffs"]
    alpha = np.asarray(alpha, dtype=float)
    return a2 * alpha ** 2 + a1 * alpha + a0


def th(index, t0):
    """ore trascorse dall'inizio della simulazione (come time_vector/3600 in results.py)"""
    return np.asarray((index - t0).total_seconds() / 3600.0)


def new_fig(ylabel=None, xlabel="Time [h]", figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid()
    return fig, ax


def window(df, start, hours=48):
    t0 = pd.Timestamp(start)
    return df.loc[t0: t0 + pd.Timedelta(hours=hours) - pd.Timedelta(minutes=1)]


def mark_days(ax, idx, t):
    for i in np.where((idx.hour == 0) & (idx.minute == 0))[0]:
        ax.axvline(t[i], color="gray", ls=":", lw=1)
        ax.text(t[i] + 0.2, 0.985, idx[i].strftime("%d/%m"), fontsize=8, va="top", color="gray",
                transform=ax.get_xaxis_transform())


def set_window_ticks(ax, t):
    ax.set_xlim(t[0], t[-1])
    ax.set_xticks(np.arange(np.ceil(t[0] / 6) * 6, t[-1] + 0.1, 6))


def soc_axis(ax, t, series, labels):
    """asse secondario per i SOC: nero tratteggiato come in plot_electric_power_sensors"""
    ax2 = ax.twinx()
    for s, lab, ls in zip(series, labels, ["--", ":", "-."]):
        ax2.plot(t, s, color="black", linestyle=ls, label=lab, lw=1.3)
    ax2.set_ylabel("State of Charge [-]")
    ax2.set_ylim(0, 1)
    return ax2


def merged_legend(ax, ax2, loc="upper right", ncol=2, outside=False, bbox=None):
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if outside:
        ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=ncol, fontsize=8)
    else:
        ax.legend(h1 + h2, l1 + l2, loc=loc, bbox_to_anchor=bbox, ncol=ncol, fontsize=8)


def power_balance(P):
    """Fonti (>0) e carichi (<0) dal lato elettrico; la somma deve fare 0."""
    g = P["electric_grid_electricity_port:electricity"]
    b = P["battery_electricity_port:electricity"]
    src = pd.DataFrame({
        "PV est": -P["pv_panels_est_electricity_port:electricity"],
        "PV ovest": -P["pv_panels_ovest_electricity_port:electricity"],
        "Motore a biogas": -P["backup_engine_electricity_port:electricity"],
        "Rete (import)": -g.clip(upper=0),
        "Batteria (scarica)": -b.clip(upper=0),
    })
    snk = pd.DataFrame({
        "Domanda rifugio": -P["demand_rifugio_electricity_port:electricity"],
        "Pompa di calore": -P["heat_pump_electricity_input_port:electricity"],
        "Rete (export)": -g.clip(lower=0),
        "Batteria (carica)": -b.clip(lower=0),
        "Perdite inverter": -(P["inverter_inverter_ac_port:electricity"] + P["inverter_inverter_dc_port:electricity"]),
    })
    return src, snk


SRC_COL = ["#f5a623", "#f8d347", "#2e9e5b", "#9aa0a6", "#8e44ad"]
SNK_COL = ["#1f77b4", "#17becf", "#5f6368", "#c39bd3", "#d62728"]


# --------------------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------------------
def fig_engine_efficiency(P, C):
    fig, ax = new_fig("Electric efficiency [-]", "Load factor [-]")
    a = np.linspace(0, 1, 201)
    ax.plot(a, eta_poly(a), color="tab:blue", lw=2, label="eta(alfa) polinomiale (combustion_engine.py)")
    on = C[C_ENG] > 0
    alpha = C.loc[on, C_ENG]
    eta_sim = (-P.loc[on, "backup_engine_electricity_port:electricity"]) / P.loc[on, "backup_engine_biogas_port:chemical_energy"]
    ax.scatter(alpha[::5], eta_sim[::5], s=18, color="red", zorder=5,
               label=f"punti di funzionamento simulati ({int(on.sum())} min)")
    ax.axvline(PARAMS["min_load"], color="red", ls=":", lw=1)
    ax.text(PARAMS["min_load"] + 0.01, 0.01, "carico minimo 0.15", color="red", fontsize=8)
    am = 0.4752 / (2 * 0.3167)
    ax.plot([am], [eta_poly(am)], "k^", ms=8, zorder=6)
    ax.annotate(f"massimo {eta_poly(am):.1%} a alfa={am:.2f}", (am, eta_poly(am)),
                xytext=(0.60, 0.095), arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.set_ylim(0, 0.40)
    #ax.axhspan(0.25, 0.38, color="green", alpha=0.08)
    #ax.text(0.02, 0.36, "intervallo tipico motori a gas 30 kW_el: 25-38 % (da verificare sul datasheet)",
    #        fontsize=8, color="green")
    ax2 = ax.twinx()
    pf = PARAMS["P_el_design"] * a[5:] / eta_poly(a[5:])
    ax2.plot(a[5:], pf, color="black", ls="--", lw=1.3, label="potenza combustibile richiesta [kW chim.]")
    chem = float(-P["biogas_producer_biogas_port:chemical_energy"].mean())
    ax2.axhline(chem, color="red", ls="-", lw=1.5, label=f"biogas disponibile: {chem:.1f} kW chim.")
    ax2.set_ylim(0, 220)
    ax2.set_ylabel("Fuel chemical power [kW]")
    merged_legend(ax, ax2, loc="upper left", ncol=1, bbox=(0.01, 0.82))
    ax.set_title("Motore a combustione: efficienza vs fattore di carico (P_el_design = 30 kW)")
    return fig


def fig_pv_day(P):
    if PV_DAY is None:
        tot = (-(P["pv_panels_est_electricity_port:electricity"] + P["pv_panels_ovest_electricity_port:electricity"]))
        daily = tot.resample("D").sum()
        full = P.resample("D").size() >= 1440
        day_str = str(daily[full].idxmax().date())
    else:
        day_str = PV_DAY
    day = P.loc[day_str]
    e = -day["pv_panels_est_electricity_port:electricity"]
    w = -day["pv_panels_ovest_electricity_port:electricity"]
    x = day.index.hour + day.index.minute / 60.0
    fig, ax = new_fig("Power [kW]", "Time of day [h] (indice di simulazione)")
    ax.plot(x, e, label=f"PV est (azimuth -90): {kwh(e):.1f} kWh", color="tab:orange")
    ax.plot(x, w, label=f"PV ovest (azimuth +90): {kwh(w):.1f} kWh", color="tab:blue")
    ax.plot(x, e + w, color="gray", lw=1, label="somma")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.02, 0.03, "Nota: dati orari PVGIS interpolati al minuto. Alba/tramonto nell'indice ~04-19 h, circa 2 h prima\n"
            "dell'orario locale estivo (CEST): l'indice del PV sembra in UTC (da confermare).",
            transform=ax.transAxes, fontsize=8, style="italic", color="dimgray")
    ax.set_title(f"Produzione fotovoltaica est/ovest - giornata piu' soleggiata ({pd.Timestamp(day_str):%d/%m/%Y})")
    return fig


def fig_pv_mean(P):
    fig, ax = new_fig("Power [kW]", "Time of day [h] (indice di simulazione)")
    for col, lab, c in [("pv_panels_est_electricity_port:electricity", "PV est", "tab:orange"),
                        ("pv_panels_ovest_electricity_port:electricity", "PV ovest", "tab:blue")]:
        hourly = (-P[col]).resample("1h").mean()
        g = hourly.groupby(hourly.index.hour)
        m, lo, hi = g.mean(), g.quantile(0.1), g.quantile(0.9)
        x = m.index + 0.5
        ax.plot(x, m, color=c, lw=2, label=f"{lab} (media, picco alle {m.idxmax():02d}-{m.idxmax() + 1:02d})")
        ax.fill_between(x, lo, hi, color=c, alpha=0.15, label=f"{lab} (10-90 percentile)")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.legend(fontsize=9)
    ax.set_title("Profilo giornaliero medio PV est/ovest sull'intero periodo (25/07-23/09)")
    return fig


def fig_demand_profile(P):
    fig, ax = new_fig("Power [kW]", "Time of day [h] (indice di simulazione)")
    dem = P["demand_rifugio_electricity_port:electricity"]
    names = {7: "luglio (25-31)", 8: "agosto", 9: "settembre (1-23)"}
    for mth, c in zip([7, 8, 9], ["tab:red", "tab:green", "tab:purple"]):
        s = dem[dem.index.month == mth].resample("1h").mean()
        m = s.groupby(s.index.hour).mean()
        ax.plot(m.index + 0.5, m.values, color=c, label=names[mth])
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.legend()
    ax.set_title("Domanda elettrica del rifugio: profilo giornaliero medio per mese")
    return fig


def fig_battery_derating(P, S):
    cap, cr = PARAMS["battery_capacity"], PARAMS["battery_crate"]
    pmax = cap * cr
    smin, sder = PARAMS["battery_SOC_min"], PARAMS["battery_SOC_derate"]
    soc = S["battery_SOC_sensor"]
    pw = P["battery_electricity_port:electricity"]
    fig, ax = new_fig("Battery port power [kW] (+ carica / - scarica)", "State of Charge [-]")
    dis, chg = pw < -1e-6, pw > 1e-6
    
    # Argomenti label="scarica (simulata)" e label="carica (simulata)" rimossi
    ax.scatter(soc[dis][::7], pw[dis][::7], s=6, alpha=0.35, color="tab:blue")
    ax.scatter(soc[chg][::7], pw[chg][::7], s=6, alpha=0.35, color="tab:orange")
    
    s = np.linspace(smin, 1, 300)
    env = -pmax * np.clip((s - smin) / (sder - smin), 0, 1)
    ax.plot(s, env, "k--", lw=1.5, label="limite di scarica con derating (batteries.py)")
    ax.axhline(pmax, color="red", lw=1.5, label=f"+/- P_max = capacita x C-rate = {pmax:.3f} kW")
    ax.axhline(-pmax, color="red", lw=1.5)
    ax.axvline(smin, color="gray", ls=":")
    ax.axvline(sder, color="gray", ls=":")
    ax.text(smin + 0.005, 0.50, "SOC_min", fontsize=8, color="gray")
    ax.text(sder + 0.005, 0.50, "SOC_derate", fontsize=8, color="gray")
    
    # Spostamento della legenda in basso, all'esterno dell'area del grafico
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=8)
    
    ax.set_title("Batteria al piombo: potenza scambiata vs SOC e limite di derating")
    viol = (pw[dis] < np.interp(soc[dis], s, env) - 1e-3).mean() if dis.any() else 0.0
    return fig, viol


def fig_balance_window(P, S, C, name, start):
    w = window(P, start)
    ws = window(S, start)
    t0 = P.index[0]
    t = th(w.index, t0)
    src, snk = power_balance(w)
    resid = (src.sum(axis=1) + snk.sum(axis=1)).abs().max()
    fig, ax = new_fig("Power [kW]")
    ax.stackplot(t, *[src[c].values for c in src.columns], labels=list(src.columns), colors=SRC_COL, alpha=0.9, lw=0)
    ax.stackplot(t, *[snk[c].values for c in snk.columns], labels=list(snk.columns), colors=SNK_COL, alpha=0.9, lw=0)
    ax.axhline(0, color="k", lw=0.6)
    set_window_ticks(ax, t)
    mark_days(ax, w.index, t)
    ax2 = soc_axis(ax, t, [ws["battery_SOC_sensor"].values], ["SOC batteria"])
    merged_legend(ax, ax2, ncol=4, outside=True)
    ax.set_title(f"Bilancio di potenza elettrica - 48 h di {name} (da {pd.Timestamp(start):%d/%m/%Y} 00:00)")
    return fig, resid


def fig_storage_window(S, C, name, start, t0):
    ws, wc = window(S, start), window(C, start)
    t = th(ws.index, t0)
    on = (wc[C_ENG] > 0).values
    n_start = int(((~on[:-1]) & on[1:]).sum() + on[0])
    fig, ax = new_fig("Biogas SOC / flare action [-]")
    ax.fill_between(t, 0, 1, where=on, step="post", color="tab:green", alpha=0.45,
                    label=f"motore acceso ({n_start} accensioni in 48 h)")
    ax.plot(t, ws["biogas_SOC_sensor"].values, color="tab:blue", lw=0.8, label="SOC stoccaggio biogas (passo 1 min)")
    f10 = wc[C_FLARE].resample("10min").mean()
    ax.plot(th(f10.index, t0), f10.values, color="tab:red", lw=1.6, label="azione torcia (media su 10 min)")
    ax.set_ylim(0, 1.05)
    set_window_ticks(ax, t)
    mark_days(ax, ws.index, t)
    ax2 = soc_axis(ax, t, [ws["battery_SOC_sensor"].values], ["SOC batteria"])
    merged_legend(ax, ax2, ncol=2, outside=True)
    ax.set_title(f"Stoccaggio biogas, motore e torcia - 48 h di {name}")
    return fig


def fig_thermal_window(P, S, C, name, start, t0):
    w, ws = window(P, start), window(S, start)
    t = th(w.index, t0)
    T = ws["storage_tank_temperature_sensor"] - 273.15
    q_hp = -w["heat_pump_heat_output_port:heat"]
    m = w["hot_water_storage_hot_water_output_port:mass"].abs()
    dT = (w["hot_water_storage_hot_water_output_port:temperature.1"]
          - w["hot_water_storage_cold_water_input_port:temperature.1"])
    q_dhw = m * CP * dT
    fig, (ax, axb) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True,
                                  gridspec_kw=dict(height_ratios=[1.2, 1]))
    ax.plot(t, T.values, color="tab:blue", label="temperatura serbatoio")
    ax.axhline(PARAMS["tank_comfort_C"], color="red", lw=1.5, label="40 C (temperatura di riferimento DHW)")
    ax.set_ylabel("Temperature [C]")
    ax.grid()
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Accumulo ACS e pompa di calore - 48 h di {name}")
    axb.fill_between(t, q_dhw.values, color="tab:cyan", alpha=0.6, label="prelievo ACS (potenza termica)")
    axb.plot(t, q_hp.values, color="tab:red", lw=1.5, label="pompa di calore (1.5 kW termici)")
    axb.set_ylabel("Thermal power [kW]")
    axb.set_xlabel("Time [h]")
    axb.grid()
    axb.legend(loc="upper right", fontsize=8)
    set_window_ticks(axb, t)
    mark_days(axb, w.index, t)
    fig.tight_layout()
    return fig


def fig_engine_zoom(P, S, C):
    a, b = ZOOM_ENGINE
    ws, wc = S.loc[a:b], C.loc[a:b]
    t0 = P.index[0]
    tm = np.asarray((ws.index - t0).total_seconds() / 60.0)
    fig, ax = new_fig("State of Charge / control action [-]", "Time [min]")
    ax.plot(tm, ws["biogas_SOC_sensor"], color="tab:blue", lw=2, label="SOC stoccaggio biogas")
    ax.plot(tm, wc[C_ENG], color="tab:green", lw=2, label="azione motore (fattore di carico)")
    ax.plot(tm, wc[C_FLARE], color="tab:red", lw=1.5, label="azione torcia")
    for y, txt in [(PARAMS["biogas_soc_on"], "soglia ON motore 0.8"),
                   (PARAMS["biogas_soc_off"], "soglia OFF motore 0.2"),
                   (PARAMS["flare_soc_activation"], "attivazione torcia 0.98")]:
        ax.axhline(y, color="gray", ls=":", lw=1)
        ax.text(tm[-1] - 0.5, y + 0.012, txt, fontsize=8, color="gray", ha="right")
    on = wc[C_ENG] > 0
    if on.any():
        ts = float((on[on].index[0] - t0).total_seconds() / 60.0)
        run_min = int(on.sum())
        ax.axvspan(ts, ts + PARAMS["min_time_on_h"] * 60, color="red", alpha=0.10,
                   label=f"tempo minimo ON impostato: {PARAMS['min_time_on_h'] * 60:.0f} min")
        ax.annotate(f"motore acceso solo {run_min} min", (ts + run_min / 2, 0.16), xytext=(ts + 3, 0.40),
                    arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(tm[0], tm[-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8)
    ax.set_title(f"Zoom 25/07 {a[-5:]}-{b[-5:]}: ciclo del motore e comportamento della torcia")
    return fig


def fig_energy_balance(P):
    src, snk = power_balance(P)
    es = {k: kwh(v) for k, v in src.items()}
    ek = {k: -kwh(v) for k, v in snk.items()}
    fig, ax = new_fig("Energy [kWh]", "")
    for pos, (d, cols) in enumerate([(es, SRC_COL), (ek, SNK_COL)]):
        bottom = 0.0
        for (k, v), c in zip(d.items(), cols):
            if v <= 0:
                continue
            ax.bar(pos, v, bottom=bottom, color=c, width=0.55, label=k)
            if v > 60:
                ax.text(pos, bottom + v / 2, f"{k}: {v:.0f}", ha="center", va="center", fontsize=8)
            bottom += v
        ax.text(pos, bottom + 50, f"totale {bottom:.0f} kWh", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Fonti (immesso nel sistema)", "Carichi e perdite"])
    ax.set_ylim(0, max(sum(es.values()), sum(ek.values())) * 1.12)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.set_title("Bilancio elettrico dell'intero periodo (1441 h): fonti = carichi + perdite")
    return fig


def fig_biogas_allocation(P):
    prod = -kwh(P["biogas_producer_biogas_port:chemical_energy"])
    eng = kwh(P["backup_engine_biogas_port:chemical_energy"])
    fl = kwh(P["biogas_flare_biogas_port:chemical_energy"])
    el = -kwh(P["backup_engine_electricity_port:electricity"])
    fig, ax = new_fig("Energy [kWh chimici]", "")
    ax.bar(0, prod, color="#cfd8dc", width=0.5)
    ax.text(0, prod / 2, f"Biogas prodotto\n{prod:.0f} kWh", ha="center", fontsize=9)
    ax.bar(1, eng, color="tab:green", width=0.5)
    ax.bar(1, fl, bottom=eng, color="tab:red", width=0.5)
    ax.text(1, eng / 2, f"al motore\n{eng:.0f} kWh ({eng / prod:.0%})", ha="center", fontsize=9, color="white")
    ax.text(1, eng + fl / 2, f"bruciato in torcia\n{fl:.0f} kWh ({fl / prod:.0%})", ha="center", fontsize=9, color="white")
    ax.bar(2, el, color="tab:blue", width=0.5)
    ax.bar(2, eng - el, bottom=el, color="#9e9e9e", width=0.5)
    ax.text(2, eng + 700, f"elettricita utile: {el:.0f} kWh\n({el / eng:.1%} del combustibile)", ha="center", fontsize=9)
    ax.text(2, el + (eng - el) / 2, f"calore e perdite\n{eng - el:.0f} kWh", ha="center", fontsize=9)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Produzione", "Destinazione del biogas", "Resa del motore"])
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.set_title(f"Dove va il biogas: solo il {eng / prod:.0%} arriva al motore, il resto e' bruciato in torcia")
    return fig


# --------------------------------------------------------------------------------------
# riepilogo numerico e orchestrazione
# --------------------------------------------------------------------------------------
def fig_dhw_periodicity(P, S):
    """Prelievo ACS su 72 h e autocorrelazione: il profilo si ripete ogni 11 h invece che ogni 24 h."""
    m = P["demand_DHW_fluid_port:mass"].abs() * 60.0          # kg/min
    start = P.index[0].normalize() + pd.Timedelta(days=10)
    sl = slice(start, start + pd.Timedelta(hours=72) - pd.Timedelta(minutes=1))
    t = (m[sl].index - start).total_seconds() / 3600.0
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)

    ax = axes[0]
    ax.fill_between(t, 0, m[sl].values, step="mid", color="tab:blue")
    for k in range(0, 73, 11):
        ax.axvline(k, color="grey", linestyle=":", linewidth=0.8)
    for k in range(0, 73, 24):
        ax.axvline(k, color="red", linestyle="-", linewidth=1)
    ax.set_xlim(0, 72)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Hot water draw [kg/min]")
    ax.set_title(f"Prelievo ACS dal {start:%d/%m}: rosso = ogni 24 h, grigio = ogni 11 h")
    ax.grid()

    ax = axes[1]
    load = -S["building_demand_sensor"][sl]
    ax.plot(t, load.values, color="black")
    for k in range(0, 73, 24):
        ax.axvline(k, color="red", linestyle="-", linewidth=1)
    ax.set_xlim(0, 72)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Power [kW]")
    ax.set_title("Carico elettrico del rifugio nelle stesse 72 h (si ripete ogni 24 h)")
    ax.grid()

    ax = axes[2]
    x = m.resample("15min").sum().values
    x = x - x.mean()
    n = len(x)
    ac = np.correlate(x, x, "full")[n - 1:] / (x.var() * n)
    lags = np.arange(n) * 0.25
    sel = lags <= 72
    ax.plot(lags[sel], ac[sel], color="tab:blue")
    for k in (11, 22, 33, 44, 55, 66):
        ax.axvline(k, color="grey", linestyle=":", linewidth=0.8)
    for k in (24, 48, 72):
        ax.axvline(k, color="red", linestyle="-", linewidth=1)
    ax.set_xlabel("Lag [h]")
    ax.set_ylabel("Autocorrelazione [-]")
    ax.set_title("Autocorrelazione del prelievo ACS: picchi a 11, 22, 33 h, nessuno a 24 h")
    ax.grid()
    return fig


def summary_lines(P, S, C, extra):
    prod = -kwh(P["biogas_producer_biogas_port:chemical_energy"])
    eng = kwh(P["backup_engine_biogas_port:chemical_energy"])
    fl = kwh(P["biogas_flare_biogas_port:chemical_energy"])
    el = -kwh(P["backup_engine_electricity_port:electricity"])
    g = P["electric_grid_electricity_port:electricity"]
    on = (C[C_ENG] > 0).astype(int)
    starts = int((on.diff().fillna(on.iloc[0]) == 1).sum())
    Q_hp = -kwh(P["heat_pump_heat_output_port:heat"])
    Q_el = kwh(P["heat_pump_electricity_input_port:electricity"])
    
    # --- CALCOLI DINAMICI PER IL RIEPILOGO ---
    # Calcolo l'efficienza massima teorica dalla parabola nei PARAMS
    a2, a1, a0 = PARAMS["eta_coeffs"]
    am = -a1 / (2 * a2) # punto di massimo della parabola
    eta_max = a2 * am**2 + a1 * am + a0
    
    # Conversione in minuti del tempo minimo di accensione
    min_time_on_min = PARAMS["min_time_on_h"] * 60
    
    # Ore totali simulate (dal primo all'ultimo indice)
    total_hours = (P.index[-1] - P.index[0]).total_seconds() / 3600.0

    return [
        "RIEPILOGO VERIFICHE - test_8",
        "",
        f"Periodo: {P.index[0]:%d/%m/%Y %H:%M} - {P.index[-1]:%d/%m/%Y %H:%M} ({len(P)} passi da 1 min)",
        "",
        "COERENZA (OK)",
        f"  - residuo bilancio elettrico globale = perdite inverter ({extra['loss']:.1f} kWh)",
        f"  - residuo massimo istantaneo nei grafici di bilancio: {extra['resid']:.1e} kW",
        f"  - COP pompa di calore = {Q_hp / Q_el:.2f} (progetto 3.2)",
        "",
        "ANOMALIE / DA RIVEDERE",
        #f"  - derating batteria NON rispettato (campioni in scarica fuori limite: {extra['viol']:.1%})",
        f"  - accensioni motore: {starts} in {total_hours:.0f} h, durata media {on.sum() / max(starts, 1):.1f} min (impostato {min_time_on_min:.0f} min)",
        f"  - efficienza media motore {el / eng:.1%} (massimo curva {eta_max:.1%})",
        f"  - biogas in torcia: {fl / prod:.1%} del prodotto; al motore {eng / prod:.1%}",
        f"  - export in rete {kwh(g.clip(lower=0)):.0f} kWh; import {-kwh(g.clip(upper=0)):.0f} kWh",
        f"  - DHW sotto 40 C in {extra['dhw_bad']:.1%} dei campioni con prelievo",
    ]

def make_all(P, S, C, outdir="figure_validazione"):
    os.makedirs(outdir, exist_ok=True)
    t0 = P.index[0]
    figs = []

    def add(name, fig):
        fig.savefig(os.path.join(outdir, name + ".png"), dpi=140, bbox_inches="tight")
        figs.append(fig)

    extra = {}
    loss = kwh(P["inverter_inverter_ac_port:electricity"] + P["inverter_inverter_dc_port:electricity"])
    extra["loss"] = loss
    m = P["hot_water_storage_hot_water_output_port:mass"].abs() > 1e-6
    extra["dhw_bad"] = float((P.loc[m, "hot_water_storage_hot_water_output_port:temperature.1"] < 313.15).mean())

    add("01_motore_efficienza_vs_carico", fig_engine_efficiency(P, C))
    add("02_pv_est_ovest_giorno", fig_pv_day(P))
    add("03_pv_est_ovest_profilo_medio", fig_pv_mean(P))
    add("04_domanda_profilo_medio", fig_demand_profile(P))
    fig, viol = fig_battery_derating(P, S)
    extra["viol"] = viol
    add("05_batteria_derating", fig)
    resid_max = 0.0
    n = 6
    for name, start in WINDOWS.items():
        fig, r = fig_balance_window(P, S, C, name, start)
        resid_max = max(resid_max, r)
        add(f"{n:02d}_{name}_48h_bilancio_elettrico", fig)
        add(f"{n + 1:02d}_{name}_48h_biogas_motore_torcia", fig_storage_window(S, C, name, start, t0))
        add(f"{n + 2:02d}_{name}_48h_termico", fig_thermal_window(P, S, C, name, start, t0))
        n += 3
    add(f"{n:02d}_zoom_ciclo_motore", fig_engine_zoom(P, S, C))
    add(f"{n + 1:02d}_bilancio_elettrico_periodo", fig_energy_balance(P))
    add(f"{n + 2:02d}_destinazione_biogas", fig_biogas_allocation(P))
    add(f"{n + 3:02d}_acs_periodicita_11h", fig_dhw_periodicity(P, S))
    extra["resid"] = resid_max

    lines = summary_lines(P, S, C, extra)
    fs = plt.figure(figsize=(10, 6))
    fs.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9.5)
    figs.insert(0, fs)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with PdfPages(os.path.join(script_dir, "diagrammi_validazione_test8.pdf")) as pdf:
        for f in figs:
            pdf.savefig(f, bbox_inches="tight")
            plt.close(f)
    print("\n".join(lines))
    print(f"\nFigure salvate in: {outdir}")


def load_xlsx(path):
    P = pd.read_excel(path, "Ports", index_col=0)
    S = pd.read_excel(path, "Sensors", index_col=0)
    C = pd.read_excel(path, "Controllers", index_col=0)
    return P, S, C


if __name__ == "__main__":
    # Calcola la cartella base in cui si trova questo script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Costruisci i percorsi assoluti per il file Excel e la cartella di output
    default_xlsx = os.path.join(script_dir, "test_8_results.xlsx")
    default_out = os.path.join(script_dir, "diagrammi test_8")
    
    xlsx = sys.argv[1] if len(sys.argv) > 1 else default_xlsx
    out = sys.argv[2] if len(sys.argv) > 2 else default_out
    
    P, S, C = load_xlsx(xlsx)
    make_all(P, S, C, out)