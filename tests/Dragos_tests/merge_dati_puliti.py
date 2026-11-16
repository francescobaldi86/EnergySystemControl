from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference, ScatterChart, Series
from openpyxl.styles import Alignment, Font

# ---------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------
# I file di ingresso vengono cercati nella cartella di questo script E in tutte le sue
# sottocartelle (a qualsiasi profondità); il file di confronto viene creato nella
# cartella dello script, indipendentemente dalla cartella da cui lo lanci.
BASE = Path(__file__).resolve().parent
NOME_FILE_PESI = "cleaned_weight.xlsx"         # output di weight_cleaning.py
NOME_FILE_TEMP = "dataset_acs_finale.xlsx"     # dataset con temperatura già pulita
FILE_USCITA = BASE / "dati_puliti_confronto.xlsx"

# Cartelle da non esplorare durante la ricerca
CARTELLE_ESCLUSE = {"__pycache__", "venv", ".venv", "env", "site-packages", "node_modules", ".git"}

# True  -> tiene solo il periodo in cui esistono sia pesi sia temperatura
#          (dataset_acs_finale parte dal 25/06 12:41, i pesi dal 15/06 01:00)
# False -> tiene l'intero periodo (le colonne mancanti restano vuote)
SOLO_FINESTRA_COMUNE = True

# Aggiunge anche la temperatura esterna (utile come contesto)
INCLUDI_TEMP_ESTERNA = True

# --- Combustibile per la produzione di ACS ---------------------------
# Potenza termica UTILE per l'ACS [kW]: ipotesi di lavoro, VERIFICATA sulle bilance (foglio "Episodi").
# Consumo medio misurato per minuto di fiamma su 144 episodi / 288 cicli = 13,9 g/min
# -> potenza utile equivalente = k * eta * PCI / 60 = 9,1 kW (IC 95% 7,4-10,6): i 10 kW sono compatibili.
# Lo script ricalcola la stima a ogni esecuzione e la riporta nel foglio Episodi.
P_ACS_KW = 10.0
ETA_CALDAIA = 0.85        # rendimento globale dello scaldabagno [-] (documentazione tecnica)
LHV_KJ_KG = 46055.0       # PCI del GPL [kJ/kg] = 11.000 kcal/kg (stesso valore di temperature_cleaning.py)
MINUTI_SENZA_CONSUMO = 1  # primi minuti di ogni ciclo ON senza combustibile (come temperature_cleaning.py)

# --- Consumo totale (bilance), consumo cucina e profili --------------
# Stessa logica di weight_to_biogas_consumption.py: massa netta = (peso - tara) / 1000,
# consumo al minuto = calo di massa netta (le ricariche non contano come consumo).
SOGLIA_RICARICA_G = 8000.0     # salto di peso [g] che identifica una ricarica/sostituzione
TARA_G_DEFAULT = 19987.0       # tara bombola [g], usata solo se nei dati non si rilevano ricariche

# Un calo di peso al minuto superiore a questa soglia NON e' un consumo fisicamente
# plausibile (0.3 kg/min = 18 kg/h = circa 110 kW di combustibile, >10 volte la caldaia):
# il minuto viene segnalato con un flag ed escluso dai PROFILI (media mobile).
# I valori al minuto e i cumulati restano invece quelli grezzi (stessi totali delle bilance).
# Metti None per non segnalare/escludere nulla.
SOGLIA_ANOMALIA_KG_MIN = 0.3

# Finestra (minuti) della media mobile centrata usata per i profili [kg/h] e [kW]
FINESTRA_PROFILO_MIN = 60

# Grafico: una riga ogni N minuti (Excel fatica con >100k punti per serie)
DECIMAZIONE_GRAFICO = 10

# --- Episodi: calibrazione della potenza su tutti i cicli (foglio "Episodi") ----------
# Cicli separati da <= GAP_EPISODIO_MIN minuti formano un unico episodio. Per ogni episodio:
#   finestra di misura [inizio - EP_PRE, fine + EP_POST]  (comprende lo smussamento del peso pulito)
#   fondo cucina = media del consumo nei EP_FONDO minuti prima e dopo la finestra
#   eccesso = consumo nella finestra - fondo x durata finestra   (massa dovuta alla caldaia)
# k [g per minuto di fiamma] = somma eccessi / somma minuti di fiamma (IC con bootstrap).
GAP_EPISODIO_MIN = 20
EP_PRE, EP_POST, EP_FONDO = 6, 8, 10
N_BOOTSTRAP = 2000

# --- Cicli isolati della caldaia (foglio "Eventi", disattivato di default) ------------
# Sostituito dal foglio Episodi, che usa tutti i cicli. Metti True per rigenerare i 98 grafici per ciclo.
CREA_FOGLIO_EVENTI = False
# Un ciclo ON e' "isolato" se nei ISOLAMENTO_MIN minuti prima e dopo NON ci sono altri cicli,
# la termocoppia e' valida, non ci sono anomalie delle bilance ne' ricariche (salto di peso
# > SALTO_PESO_ESCLUSO_G). Per ogni ciclo isolato viene creato un grafico "Combustibile ACS vs
# rilascio dalle bombole" (valori al minuto) nel foglio "Eventi".
ISOLAMENTO_MIN = 15          # minuti (minimo effettivo 10)
SALTO_PESO_ESCLUSO_G = 1000  # [g] un aumento di peso al minuto oltre questa soglia = ricarica
GRAFICO_PRE_MIN = 10         # minuti mostrati nel grafico prima dell'accensione
GRAFICO_POST_MIN = 15        # minuti mostrati dopo lo spegnimento
# Finestre di misura per ciclo (rispetto all'accensione s e allo spegnimento e):
FONDO_PRE = (10, 4)          # fondo (cucina) = media del consumo in [s-10, s-4]
MISURA_PRE, MISURA_POST = 3, 8   # massa misurata = somma di (consumo - fondo) in [s-3, e+8]

# Nomi delle colonne del foglio "Dati"
C_PESO_R = "Bombola destra [g]"
C_PESO_L = "Bombola sinistra [g]"
C_TFUMI = "Temp. fumi caldaia [°C]"
C_ON = "Caldaia ON [-]"
C_FIAMMA = "Fiamma ON [-]"
C_TEXT = "Temp. esterna [°C]"
C_ACS = "Combustibile ACS modello [kg/min]"
C_ACS_ATT = "Combustibile ACS attribuito [kg/min]"
C_ACS_NOSS = "ACS non osservato dalle bilance [kg/min]"
C_ACS_CUM = "Combustibile ACS attribuito cumulato [kg]"
C_TOT = "Consumo totale bilance [kg/min]"
C_TOT_CUM = "Consumo totale bilance cumulato [kg]"
C_FLAG = "Anomalia bilance [flag]"
C_CUC = "Consumo cucina [kg/min]"
C_CUC_CUM = "Consumo cucina cumulato [kg]"
C_POT = "Potenza utile caldaia [kW]"
C_P_POT = "Profilo potenza utile caldaia [kW]"
C_P_ACS = "Profilo consumo ACS [kg/h]"
C_P_CUC = "Profilo consumo cucina [kg/h]"
C_P_TOT = "Profilo consumo totale [kg/h]"


def trova_file(nome):
    """Cerca `nome` nella cartella dello script e in tutte le sottocartelle.
    Se ci sono più copie usa la più recente (e avvisa)."""
    trovati = []
    for p in BASE.rglob("*"):
        rel = p.relative_to(BASE).parts
        if any(part in CARTELLE_ESCLUSE for part in rel[:-1]):
            continue
        if p.is_file() and p.name.lower() == nome.lower() and not p.name.startswith("~$"):
            trovati.append(p)
    if not trovati:
        presenti = "\n  ".join(str(p.relative_to(BASE)) for p in sorted(BASE.rglob("*.xlsx"))
                              if not any(part in CARTELLE_ESCLUSE for part in p.relative_to(BASE).parts[:-1]))
        raise FileNotFoundError(f"Non trovo {nome} in {BASE} né nelle sue sottocartelle.\n"
                                f"File .xlsx presenti:\n  {presenti or '(nessuno)'}")
    trovati.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if len(trovati) > 1:
        print(f"ATTENZIONE: {len(trovati)} copie di {nome}, uso la più recente:")
        for p in trovati:
            print("   ", p.relative_to(BASE), "<-- usato" if p == trovati[0] else "")
    return trovati[0]


def carica_pesi(path):
    df = pd.read_excel(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    # Nomi colonna come prodotti da weight_cleaning.py (o da cleaned_weight.xlsx)
    rinomina = {
        "LPG (right) Cleaned": C_PESO_R,
        "LPG (left) Cleaned": C_PESO_L,
        "LPG (right)": C_PESO_R,
        "LPG (left)": C_PESO_L,
    }
    df = df.rename(columns=rinomina)
    return df[[C_PESO_R, C_PESO_L]]


def carica_temperature(path):
    df = pd.read_excel(path)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.set_index("Time")
    cols = {"Exhaust_Gas_Temp_C": C_TFUMI, "Boiler_ON": C_ON}
    if INCLUDI_TEMP_ESTERNA:
        cols["Outdoor_Temp_C"] = C_TEXT
    df = df[list(cols)].rename(columns=cols)
    df.index.name = "time"
    return df


def consumo_totale_bilance(pesi):
    """Consumo totale di gas (GPL) [kg/min] dalle due bilance (serie COMPLETA, prima dell'unione).
    Replica weight_to_biogas_consumption.py: tara = media dei pesi subito prima di ogni ricarica,
    massa netta = max(0, (peso - tara)/1000), consumo = calo di massa netta (>= 0)."""
    tare = []
    for col in pesi.columns:
        diffs = pesi[col].diff()
        for idx in diffs[diffs > SOGLIA_RICARICA_G].index:
            pos = pesi.index.get_loc(idx)
            tare.append(pesi[col].iloc[pos - 1])
    if tare:
        tara_g = float(np.mean(tare))
    else:
        tara_g = TARA_G_DEFAULT
        print(f"ATTENZIONE: nessuna ricarica rilevata, uso tara di default {tara_g:.0f} g")
    netta_kg = ((pesi - tara_g) / 1000.0).clip(lower=0)
    consumo = (-netta_kg.diff()).clip(lower=0).fillna(0.0).sum(axis=1)
    return consumo, tara_g


def media_mobile(serie, maschera):
    """Media mobile centrata (FINESTRA_PROFILO_MIN minuti) sui soli minuti validi (maschera True).
    Richiede almeno meta' finestra di minuti validi, altrimenti NaN. Risultato per ora (x60)."""
    return (serie.where(maschera)
                 .rolling(FINESTRA_PROFILO_MIN, center=True, min_periods=FINESTRA_PROFILO_MIN // 2)
                 .mean()) * 60.0


def categorie_testo(ch):
    """Forza le categorie come riferimento a TESTO (strRef): con numRef Excel puo' mostrare 1,2,3... o un asse date."""
    from openpyxl.chart.data_source import AxDataSource, StrRef
    for s in ch.series:
        if s.cat is not None and s.cat.numRef is not None:
            s.cat = AxDataSource(strRef=StrRef(f=s.cat.numRef.f))


def senza_sovrapposizioni(ch):
    """Excel sovrappone titolo, titoli degli assi e legenda all'area del grafico se 'overlay' non e' False."""
    categorie_testo(ch)
    if ch.title is not None:
        ch.title.overlay = False
    for ax in (ch.x_axis, ch.y_axis):
        if ax.title is not None:
            ax.title.overlay = False
    if ch.legend is not None:
        ch.legend.overlay = False


def bilancio_cucina(df):
    """Ripartizione del consumo misurato tra ACS e cucina con un bilancio di massa per episodio.

    Fuori dagli episodi la caldaia e' spenta: tutto il consumo misurato e' cucina.
    In ogni finestra di episodio [inizio - EP_PRE, fine + EP_POST] (stessa definizione del foglio Episodi):
        T = consumo misurato nella finestra, A = combustibile ACS del modello
        F = fondo cucina = media del consumo nei EP_FONDO minuti prima e dopo la finestra x durata finestra
        ACS attribuito  = min(A, max(0, T - F))   (eccesso sopra il fondo cucina, al massimo quanto il modello)
        cucina          = T - ACS attribuito  >= 0
        ACS non osservato = A - ACS attribuito    (diagnostica: modello oltre la misura)
    Senza sottrarre F, il consumo della cucina che cade per caso nella finestra verrebbe attribuito all'ACS
    (verifica placebo: con le accensioni spostate di 45 min si attribuiva comunque il 50% del modello).
    Al minuto, ACS attribuito e cucina si ripartiscono il consumo misurato in proporzione
    (quindi ACS + cucina = totale in ogni minuto) e l'ACS non osservato segue il profilo del modello.
    Restituisce (acs_attribuito, acs_non_osservato) in kg/min; NaN dove l'ACS del modello non e' noto."""
    m = lambda x: pd.Timedelta(minutes=x)
    tot = df[C_TOT].fillna(0.0)
    acs = df[C_ACS]
    att = acs.copy()                         # fuori dagli episodi coincide col modello (= 0 a caldaia spenta)
    noss = acs * 0.0
    on = df[C_ON] == 1
    starts = list(df.index[on & ~on.shift(1, fill_value=False)])
    ends = list(df.index[on & ~on.shift(-1, fill_value=False)])
    finestre = []
    for s_, e_ in zip(starts, ends):
        a, b = s_ - m(EP_PRE), e_ + m(EP_POST)
        if finestre and (s_ - finestre[-1][2]) / m(1) <= GAP_EPISODIO_MIN or (finestre and a <= finestre[-1][1]):
            finestre[-1][1], finestre[-1][2] = max(b, finestre[-1][1]), e_
        else:
            finestre.append([a, b, e_])
    for a, b, _ in finestre:
        w = acs.loc[a:b]
        noto = w.notna()
        if not noto.any():
            continue
        idx = w.index[noto]
        T = float(tot.loc[idx].sum())
        A = float(w[noto].sum())
        fondo = np.nanmean([df[C_TOT].loc[a - m(EP_FONDO): a - m(1)].mean(),
                            df[C_TOT].loc[b + m(1): b + m(EP_FONDO)].mean()])
        F = (0.0 if np.isnan(fondo) else fondo) * len(idx)
        preso = min(A, max(0.0, T - F))
        att.loc[idx] = tot.loc[idx] * (preso / T) if T > 0 else 0.0
        noss.loc[idx] = w[noto] * ((A - preso) / A) if A > 0 else 0.0
    return att, noss


def trova_episodi(df):
    """Raggruppa i cicli ON in episodi e calcola, per ciascuno, l'eccesso di consumo dovuto alla caldaia."""
    m = lambda x: pd.Timedelta(minutes=x)
    on = df[C_ON] == 1
    starts = list(df.index[on & ~on.shift(1, fill_value=False)])
    ends = list(df.index[on & ~on.shift(-1, fill_value=False)])
    gruppi = []
    for s_, e_ in zip(starts, ends):
        if gruppi and (s_ - gruppi[-1][1]) / m(1) <= GAP_EPISODIO_MIN:
            gruppi[-1][1] = e_
            gruppi[-1][2] += 1
        else:
            gruppi.append([s_, e_, 1])
    righe, scarti = [], {}
    for s_, e_, nc in gruppi:
        a, b = s_ - m(EP_PRE), e_ + m(EP_POST)
        tutto = df.loc[a - m(EP_FONDO): b + m(EP_FONDO)]
        motivo = None
        if len(tutto) < (b - a) / m(1) + 1 + 2 * EP_FONDO:
            motivo = "bordo dei dati"
        elif tutto[C_ON].isna().any():
            motivo = "blackout termocoppia"
        elif (tutto[C_FLAG] == 1).any():
            motivo = "anomalia bilance"
        elif (tutto[[C_PESO_R, C_PESO_L]].diff() > SALTO_PESO_ESCLUSO_G).any().any():
            motivo = "ricarica"
        if motivo:
            scarti[motivo] = scarti.get(motivo, 0) + nc
            continue
        win = df.loc[a:b]
        fondo = float(np.mean([df.loc[a - m(EP_FONDO): a - m(1), C_TOT].mean(),
                               df.loc[b + m(1): b + m(EP_FONDO), C_TOT].mean()]))
        misurato = float(win[C_TOT].sum())
        eccesso = misurato - fondo * len(win)
        modello = float(win[C_ACS].sum())
        righe.append(dict(inizio=s_, fine=e_, cicli=nc, minuti_on=int(win[C_ON].sum()),
                          minuti_fiamma=float(win[C_FIAMMA].sum()), misurato=misurato,
                          fondo=fondo, eccesso=eccesso, modello=modello,
                          rapporto=eccesso / modello if modello > 0 else np.nan))
    return pd.DataFrame(righe), len(starts), scarti


def calibra(ep, seme=0):
    """k = somma eccessi / somma minuti di fiamma [kg/min], IC 95% bootstrap, potenza utile equivalente [kW]."""
    x = ep["minuti_fiamma"].to_numpy()
    y = ep["eccesso"].to_numpy()
    if len(ep) < 3 or x.sum() <= 0:
        return None
    k = y.sum() / x.sum()
    rng = np.random.default_rng(seme)
    boot = []
    for _ in range(N_BOOTSTRAP):
        j = rng.integers(0, len(ep), len(ep))
        if x[j].sum() > 0:
            boot.append(y[j].sum() / x[j].sum())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    pot = lambda kk: kk * ETA_CALDAIA * LHV_KJ_KG / 60.0      # kg/min -> kW utili
    return dict(n=len(ep), cicli=int(ep["cicli"].sum()), fiamma=float(x.sum()),
                k=k, k_lo=lo, k_hi=hi, P=pot(k), P_lo=pot(lo), P_hi=pot(hi),
                rapporto=float(y.sum() / ep["modello"].sum()) if ep["modello"].sum() > 0 else np.nan)


def crea_foglio_episodi(wb, ep, n_cicli_tot, scarti, kg_min):
    ws = wb.create_sheet("Episodi")
    b = Font(bold=True)
    ws["A1"] = "Calibrazione della potenza ACS sugli episodi (cicli separati da meno di GAP_EPISODIO_MIN minuti)"
    ws["A1"].font = Font(bold=True, size=12)
    ws["A2"] = (f"Episodi validi: {len(ep)} | cicli inclusi: {int(ep['cicli'].sum())} su {n_cicli_tot} | "
                f"cicli esclusi: " + (", ".join(f"{k} {v}" for k, v in scarti.items()) or "nessuno"))

    # --- tabella di calibrazione per sottoinsiemi (verifica di coerenza)
    sotto = [("Tutti gli episodi", ep),
             ("Episodi di 1 ciclo", ep[ep["cicli"] == 1]),
             ("Episodi di più cicli", ep[ep["cicli"] > 1]),
             ("Fiamma <= 3 min", ep[ep["minuti_fiamma"] <= 3]),
             ("Fiamma > 3 min", ep[ep["minuti_fiamma"] > 3])]
    for mese, g in ep.groupby(ep["inizio"].dt.month):
        sotto.append((f"Mese {mese:02d}", g))
    intest = ["Sottoinsieme", "Episodi", "Cicli", "Minuti fiamma", "k [g/min fiamma]", "IC95 basso", "IC95 alto",
              "P utile equiv. [kW]", "IC95 basso [kW]", "IC95 alto [kW]", f"Rapporto mis./modello (P={P_ACS_KW:g} kW)"]
    ws.append([])
    ws.append(intest)
    for c in ws[ws.max_row]:
        c.font = b
        c.alignment = Alignment(wrap_text=True, vertical="top")
    riga_tutti = None
    for nome, g in sotto:
        cal = calibra(g)
        if cal is None:
            ws.append([nome, len(g), int(g["cicli"].sum()) if len(g) else 0] + ["n.d."] * 8)
            continue
        ws.append([nome, cal["n"], cal["cicli"], round(cal["fiamma"]),
                   round(cal["k"] * 1000, 1), round(cal["k_lo"] * 1000, 1), round(cal["k_hi"] * 1000, 1),
                   round(cal["P"], 2), round(cal["P_lo"], 2), round(cal["P_hi"], 2), round(cal["rapporto"], 2)])
        if riga_tutti is None:
            riga_tutti = cal
    r_note = ws.max_row + 2
    ws.cell(row=r_note, column=1, value="Lettura").font = b
    note = [f"k = somma degli eccessi / somma dei minuti di fiamma. Il modello attuale assume {kg_min * 1000:.1f} g/min (P_ACS_KW = {P_ACS_KW:g} kW, "
            f"rendimento {ETA_CALDAIA:g}, PCI {LHV_KJ_KG:.0f} kJ/kg).",
            "P utile equivalente = k x rendimento x PCI / 60: e' la potenza utile che rende il modello coerente con le bilance (calibra il prodotto P/rendimento/PCI, non le singole grandezze).",
            "Se P_ACS_KW cade fuori dall'IC della riga 'Tutti gli episodi', il modello non e' coerente con le bilance: aggiornare P_ACS_KW (anche in temperature_cleaning.py).",
            "Le righe successive verificano la coerenza: k simile tra sottoinsiemi = stima credibile. Gli episodi brevi e il mese di giugno hanno IC ampi."]
    for i, t in enumerate(note, start=1):
        ws.cell(row=r_note + i, column=1, value=t)

    # --- tabella episodi
    r0 = r_note + len(note) + 3
    cols = ["#", "Inizio", "Fine", "Cicli", "Minuti ON", "Minuti fiamma", "Consumo misurato [kg]", "Fondo cucina [kg/min]",
            "Eccesso caldaia [kg]", "Combustibile ACS modello [kg]", "Rapporto eccesso/modello"]
    for j, t in enumerate(cols, start=1):
        c = ws.cell(row=r0, column=j, value=t)
        c.font = b
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for i, e in enumerate(ep.itertuples(index=False), start=1):
        r = r0 + i
        vals = [i, f"{e.inizio:%d/%m/%Y %H:%M}", f"{e.fine:%H:%M}", e.cicli, e.minuti_on, e.minuti_fiamma,
                round(e.misurato, 4), round(e.fondo, 5), round(e.eccesso, 4), round(e.modello, 4),
                None if pd.isna(e.rapporto) else round(e.rapporto, 2)]
        for j, v in enumerate(vals, start=1):
            ws.cell(row=r, column=j, value=v)
    r_fine = r0 + len(ep)

    # --- dispersione eccesso vs minuti di fiamma, con retta calibrata e retta del modello (in g)
    COL = 20   # colonne T:X per i dati del grafico (fuori dall'area coperta dal grafico)
    ws.cell(row=r0, column=COL, value="Minuti fiamma").font = b
    ws.cell(row=r0, column=COL + 1, value="Eccesso misurato [g]").font = b
    for i, e in enumerate(ep.itertuples(index=False), start=1):
        ws.cell(row=r0 + i, column=COL, value=e.minuti_fiamma)
        ws.cell(row=r0 + i, column=COL + 1, value=round(e.eccesso * 1000, 1))
    xmax = float(ep["minuti_fiamma"].max())
    ws.cell(row=r0, column=COL + 2, value="x retta").font = b
    ws.cell(row=r0, column=COL + 3, value="Calibrata [g]").font = b
    ws.cell(row=r0, column=COL + 4, value="Modello [g]").font = b
    for i, xv in enumerate([0.0, xmax], start=1):
        ws.cell(row=r0 + i, column=COL + 2, value=xv)
        ws.cell(row=r0 + i, column=COL + 3, value=round(riga_tutti["k"] * 1000 * xv, 1))
        ws.cell(row=r0 + i, column=COL + 4, value=round(kg_min * 1000 * xv, 1))
    ch = ScatterChart()
    ch.title = "Eccesso di consumo per episodio vs minuti di fiamma"
    ch.x_axis.title = "Minuti di fiamma nell'episodio"
    ch.y_axis.title = "Eccesso misurato dalle bilance [g]"
    ch.style = 13
    ch.width, ch.height = 22, 12
    s1 = Series(Reference(ws, min_col=COL + 1, min_row=r0 + 1, max_row=r_fine),
                Reference(ws, min_col=COL, min_row=r0 + 1, max_row=r_fine), title="Episodi")
    s1.marker.symbol = "circle"
    s1.marker.size = 6
    s1.marker.graphicalProperties.solidFill = "2A78D6"
    s1.marker.graphicalProperties.line.solidFill = "2A78D6"
    s1.graphicalProperties.line.noFill = True
    xs = Reference(ws, min_col=COL + 2, min_row=r0 + 1, max_row=r0 + 2)
    s2 = Series(Reference(ws, min_col=COL + 3, min_row=r0 + 1, max_row=r0 + 2), xs,
                title=f"Calibrata: {riga_tutti['k'] * 1000:.1f} g/min ({riga_tutti['P']:.1f} kW)")
    s2.graphicalProperties.line.solidFill = "EB6834"
    s2.graphicalProperties.line.width = 28575
    s2.marker.symbol = "none"
    s3 = Series(Reference(ws, min_col=COL + 4, min_row=r0 + 1, max_row=r0 + 2), xs,
                title=f"Modello in uso: {kg_min * 1000:.1f} g/min ({P_ACS_KW:g} kW)")
    s3.graphicalProperties.line.solidFill = "6B6B6B"
    s3.graphicalProperties.line.dashStyle = "dash"
    s3.graphicalProperties.line.width = 19050
    s3.marker.symbol = "none"
    for sr in (s1, s2, s3):
        ch.series.append(sr)
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    ch.legend.position = "b"
    senza_sovrapposizioni(ch)
    ws.add_chart(ch, "M3")

    for col, w in zip("ABCDEFGHIJK", (24, 18, 9, 8, 11, 13, 16, 16, 16, 18, 22)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = ws.cell(row=r0 + 1, column=1)
    return riga_tutti


def trova_cicli_isolati(df):
    """Cicli ON isolati con le metriche di confronto tra bilance e combustibile ACS del modello."""
    iso = max(ISOLAMENTO_MIN, GRAFICO_PRE_MIN, GRAFICO_POST_MIN, 10)
    m = lambda x: pd.Timedelta(minutes=x)
    on = df[C_ON] == 1
    starts = df.index[on & ~on.shift(1, fill_value=False)]
    ends = df.index[on & ~on.shift(-1, fill_value=False)]
    righe = []
    for s_, e_ in zip(starts, ends):
        dur = int((e_ - s_).total_seconds() / 60) + 1
        seg = df.loc[s_ - m(iso): e_ + m(iso)]
        if len(seg) < dur + 2 * iso:                      # troppo vicino ai bordi dei dati
            continue
        if seg[C_ON].isna().any():                        # blackout termocoppia
            continue
        if int((seg[C_ON] == 1).sum()) != dur:            # altro ciclo vicino
            continue
        if (seg[C_FLAG] == 1).any():                      # anomalia delle bilance
            continue
        if (seg[[C_PESO_R, C_PESO_L]].diff() > SALTO_PESO_ESCLUSO_G).any().any():   # ricarica
            continue
        fondo = df[C_TOT].loc[s_ - m(FONDO_PRE[0]): s_ - m(FONDO_PRE[1])].mean()
        ecc = df[C_TOT].loc[s_ - m(MISURA_PRE): e_ + m(MISURA_POST)] - fondo
        misurato = float(ecc.sum())
        modello = float(df[C_ACS].loc[s_:e_].sum())
        pos = ecc.clip(lower=0)
        centro = s_ + m((dur - 1) / 2)
        if pos.sum() > 0:
            dt_min = np.asarray((pos.index - centro).total_seconds()) / 60.0
            ritardo = float((dt_min * pos.values).sum() / pos.sum())
        else:
            ritardo = np.nan
        righe.append(dict(start=s_, end=e_, durata=dur, modello=modello, misurato=misurato,
                          rapporto=misurato / modello if modello else np.nan,
                          ritardo=ritardo, fondo=float(fondo)))
    return pd.DataFrame(righe), len(starts), iso


def crea_foglio_eventi(wb, ws_dati, df, cicli, n_cicli_tot, iso):
    """Foglio 'Eventi': tabella dei cicli isolati, profilo medio e un grafico per ciclo."""
    m = lambda x: pd.Timedelta(minutes=x)
    BLU, ARANCIO = "2A78D6", "EB6834"
    ws = wb.create_sheet("Eventi")
    col_dati = {c.value: j for j, c in enumerate(ws_dati[1], start=1)}
    n = len(cicli)

    ws["A1"] = "Cicli isolati della caldaia: combustibile ACS (modello) vs rilascio dalle bombole (bilance)"
    ws["A1"].font = Font(bold=True, size=13)
    ws["A2"] = (f"Isolati = nessun altro ciclo ON nei ±{iso} min, termocoppia valida, nessuna anomalia/ricarica delle bilance: "
                f"{n} cicli su {n_cicli_tot}. Misurato = somma di (consumo - fondo) in [-{MISURA_PRE}, +{MISURA_POST}] min.")
    intest = ["#", "Inizio", "Durata [min]", "ACS modello [kg]", "Bombole misurato [kg]",
              "Rapporto misurato/modello", "Ritardo baricentro [min]", "Fondo cucina [kg/min]", "Grafico"]
    for j, t in enumerate(intest, start=1):
        c = ws.cell(row=4, column=j, value=t)
        c.font = Font(bold=True)
        c.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[4].height = 32

    # --- posizione dei grafici per ciclo: griglia 3 colonne a destra della tabella
    ancore_col = ["N", "W", "AF"]
    riga0, passo = 24, 16
    ancore = [f"{ancore_col[i % 3]}{riga0 + (i // 3) * passo}" for i in range(n)]

    for i, c in enumerate(cicli.itertuples(index=False), start=1):
        r = 4 + i
        ws.cell(row=r, column=1, value=i)
        ws.cell(row=r, column=2, value=c.start.to_pydatetime()).number_format = "dd/mm/yyyy hh:mm"
        ws.cell(row=r, column=3, value=int(c.durata))
        ws.cell(row=r, column=4, value=c.modello).number_format = "0.000"
        ws.cell(row=r, column=5, value=c.misurato).number_format = "0.000"
        ws.cell(row=r, column=6, value=None if pd.isna(c.rapporto) else c.rapporto).number_format = "0.00"
        ws.cell(row=r, column=7, value=None if pd.isna(c.ritardo) else c.ritardo).number_format = "0.0"
        ws.cell(row=r, column=8, value=c.fondo).number_format = "0.0000"
        lk = ws.cell(row=r, column=9, value=f'=HYPERLINK("#Eventi!{ancore[i - 1]}","vai")')
        lk.font = Font(color="2A78D6", underline="single")
    r_fine = 4 + n
    ws.cell(row=r_fine + 2, column=2, value="Somma / rapporto Σ/Σ").font = Font(bold=True)
    ws.cell(row=r_fine + 2, column=3, value=f"=SUM(C5:C{r_fine})")
    ws.cell(row=r_fine + 2, column=4, value=f"=SUM(D5:D{r_fine})").number_format = "0.000"
    ws.cell(row=r_fine + 2, column=5, value=f"=SUM(E5:E{r_fine})").number_format = "0.000"
    ws.cell(row=r_fine + 2, column=6, value=f"=E{r_fine + 2}/D{r_fine + 2}").number_format = "0.00"
    ws.cell(row=r_fine + 3, column=2, value="Mediana / media").font = Font(bold=True)
    ws.cell(row=r_fine + 3, column=6, value=f"=MEDIAN(F5:F{r_fine})").number_format = "0.00"
    ws.cell(row=r_fine + 3, column=7, value=f"=MEDIAN(G5:G{r_fine})").number_format = "0.0"
    ws.cell(row=r_fine + 3, column=8, value=f"=AVERAGE(H5:H{r_fine})").number_format = "0.0000"
    for col, w in zip("ABCDEFGHIJKLM", (5, 24, 11, 16, 20, 25, 22, 20, 9, 18, 20, 24, 3)):
        ws.column_dimensions[col].width = w

    # --- profilo medio dei cicli isolati, allineato all'accensione (lag 0 = primo minuto ON)
    lags = np.arange(-10, 26)
    acs_m = np.nanmean([[df[C_ACS].get(c.start + m(int(l)), np.nan) for l in lags] for c in cicli.itertuples()], axis=0)
    tot_m = np.nanmean([[df[C_TOT].get(c.start + m(int(l)), np.nan) for l in lags] for c in cicli.itertuples()], axis=0)
    for j, t in enumerate(["Minuti da accensione", "ACS medio [kg/min]", "Bombole medio [kg/min]"], start=10):
        c = ws.cell(row=4, column=j, value=t)
        c.font = Font(bold=True)
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for k, (l, a, t) in enumerate(zip(lags, acs_m, tot_m), start=5):
        ws.cell(row=k, column=10, value=int(l))
        ws.cell(row=k, column=11, value=float(a)).number_format = "0.0000"
        ws.cell(row=k, column=12, value=float(t)).number_format = "0.0000"

    def stile(ch, w, h, ymax):
        ch.width, ch.height = w, h
        ch.y_axis.title = "kg/min"
        ch.y_axis.scaling.min = 0
        ch.y_axis.scaling.max = ymax
        ch.y_axis.number_format = "0.000"
        ch.x_axis.delete = False
        ch.y_axis.delete = False
        ch.legend.position = "b"
        for serie, colore in zip(ch.series, (BLU, ARANCIO)):
            serie.smooth = False                         # niente linee smussate (creano falsi valori < 0)
            serie.graphicalProperties.line.solidFill = colore
            serie.graphicalProperties.line.width = 28575  # 2.25 pt
            serie.marker.symbol = "none"

    ca = LineChart()
    ca.title = f"Profilo medio dei {n} cicli isolati (0 = primo minuto ON)"
    ca.add_data(Reference(ws, min_col=11, max_col=12, min_row=4, max_row=4 + len(lags)), titles_from_data=True)
    ca.set_categories(Reference(ws, min_col=10, min_row=5, max_row=4 + len(lags)))
    ca.x_axis.title = "Minuti dall'accensione"
    stile(ca, 24, 9.5, 0.035)
    senza_sovrapposizioni(ca)
    ws.add_chart(ca, "N3")

    # --- un grafico per ciclo (valori dal foglio Dati; etichette dell'ora come TESTO in colonna AR,
    #     altrimenti Excel usa un asse delle date e comprime tutti i minuti sullo stesso giorno)
    COL_LAB, riga_lab = 44, 5
    ws.cell(row=4, column=COL_LAB, value="Etichette ora (per i grafici)").font = Font(bold=True)
    ws.column_dimensions["AR"].width = 26
    for i, c in enumerate(cicli.itertuples(index=False), start=1):
        r0 = df.index.get_loc(c.start - m(GRAFICO_PRE_MIN)) + 2
        r1 = df.index.get_loc(c.end + m(GRAFICO_POST_MIN)) + 2
        fin = df.loc[c.start - m(GRAFICO_PRE_MIN): c.end + m(GRAFICO_POST_MIN)]
        ymax = max(0.035, float(np.ceil(1.1 * max(fin[C_TOT].max(), fin[C_ACS].max()) / 0.005) * 0.005))
        ch = LineChart()
        rap = "" if pd.isna(c.rapporto) else f" · rapp. {c.rapporto:.2f}"
        ch.title = f"#{i} · {c.start:%d/%m %H:%M} · {int(c.durata)} min ON{rap}"
        for nome, col in (("Combustibile ACS (modello)", C_ACS), ("Rilascio bombole (bilance)", C_TOT)):
            ch.series.append(Series(Reference(ws_dati, min_col=col_dati[col], min_row=r0, max_row=r1), title=nome))
        etich = df.loc[c.start - m(GRAFICO_PRE_MIN): c.end + m(GRAFICO_POST_MIN)].index
        for k, t in enumerate(etich):
            ws.cell(row=riga_lab + k, column=COL_LAB, value=f"{t:%H:%M}")
        ch.set_categories(Reference(ws, min_col=COL_LAB, min_row=riga_lab, max_row=riga_lab + len(etich) - 1))
        riga_lab += len(etich) + 1
        ch.x_axis.tickLblSkip = 5
        ch.x_axis.tickMarkSkip = 5
        stile(ch, 15, 7.5, ymax)
        senza_sovrapposizioni(ch)
        ws.add_chart(ch, ancore[i - 1])
    return ws


def main():
    FILE_PESI = trova_file(NOME_FILE_PESI)
    FILE_TEMP = trova_file(NOME_FILE_TEMP)
    print(f"Pesi:         {FILE_PESI.relative_to(BASE)}")
    print(f"Temperature:  {FILE_TEMP.relative_to(BASE)}")
    print("Lettura file...")
    pesi = carica_pesi(FILE_PESI)
    temp = carica_temperature(FILE_TEMP)

    # Consumo totale dalle bilance: calcolato sulla serie completa PRIMA dell'unione,
    # cosi' il primo minuto della finestra non perde il suo "minuto precedente"
    consumo_tot, tara_g = consumo_totale_bilance(pesi)

    # Join sull'indice temporale (passo 1 minuto in entrambi i file)
    how = "inner" if SOLO_FINESTRA_COMUNE else "outer"
    df = pesi.join(temp, how=how).sort_index()
    df.index.name = "time"
    df[C_TOT] = consumo_tot.reindex(df.index)

    # --- Combustibile per l'ACS ---------------------------------------------
    # potenza al focolare = P_ACS / eta, per i minuti con caldaia ON
    # [kW] * 60 [s/min] / [kJ/kg] = kg/min  (NaN dove la termocoppia e' in blackout)
    kg_per_minuto_on = (P_ACS_KW / ETA_CALDAIA) * 60.0 / LHV_KJ_KG
    # Fiamma: Boiler_ON senza i primi MINUTI_SENZA_CONSUMO minuti di ogni ciclo (nessun calo misurabile alle bilance)
    on_b = df[C_ON] == 1
    n_nel_ciclo = on_b.groupby((on_b & ~on_b.shift(1, fill_value=False)).cumsum()).cumsum()
    fiamma = (on_b & (n_nel_ciclo > MINUTI_SENZA_CONSUMO)).astype(float).where(df[C_ON].notna())
    df[C_FIAMMA] = fiamma
    df[C_ACS] = fiamma * kg_per_minuto_on

    # --- Bilancio di massa per episodio: ACS attribuito e cucina (>= 0) -------
    # NaN dove l'ACS non e' noto (blackout termocoppia). Vedi bilancio_cucina().
    df[C_ACS_ATT], df[C_ACS_NOSS] = bilancio_cucina(df)
    df[C_ACS_CUM] = df[C_ACS_ATT].cumsum()
    df[C_CUC] = (df[C_TOT] - df[C_ACS_ATT]).clip(lower=0)   # il clip toglie solo residui numerici ~1e-17
    df[C_TOT_CUM] = df[C_TOT].cumsum()
    df[C_CUC_CUM] = df[C_CUC].cumsum()

    # --- Potenza utile erogata dalla caldaia ---------------------------------
    df[C_POT] = fiamma * P_ACS_KW

    # --- Anomalie delle bilance e profili ------------------------------------
    if SOGLIA_ANOMALIA_KG_MIN is None:
        df[C_FLAG] = 0
    else:
        df[C_FLAG] = (df[C_TOT] > SOGLIA_ANOMALIA_KG_MIN).astype(int)
    valido = df[C_ACS].notna() & (df[C_FLAG] == 0)   # stessa maschera per tutti i profili
    df[C_P_POT] = media_mobile(df[C_POT], valido) / 60.0          # media in kW (non x60)
    df[C_P_ACS] = media_mobile(df[C_ACS_ATT], valido)
    df[C_P_CUC] = media_mobile(df[C_CUC], valido)
    df[C_P_TOT] = media_mobile(df[C_TOT], valido)

    # Ordine colonne: dati misurati | consumi al minuto | cumulati | caldaia | profili
    ordine = [C_PESO_R, C_PESO_L, C_TFUMI, C_ON, C_FIAMMA, C_TEXT,
              C_TOT, C_ACS, C_ACS_ATT, C_ACS_NOSS, C_CUC, C_TOT_CUM, C_ACS_CUM, C_CUC_CUM,
              C_FLAG, C_POT, C_P_POT, C_P_ACS, C_P_CUC, C_P_TOT]
    df = df[[c for c in ordine if c in df.columns]]

    # --- Riepilogo a console --------------------------------------------------
    acs_noto = df[C_ACS].notna()
    tot_finestra = df[C_TOT].sum()
    tot_valido = df.loc[acs_noto, C_TOT].sum()
    acs_tot = df[C_ACS].sum()
    acs_att_tot = df[C_ACS_ATT].sum()
    acs_noss_tot = df[C_ACS_NOSS].sum()
    cuc_tot = df[C_CUC].sum()
    kg_anom = df.loc[df[C_FLAG] == 1, C_TOT].sum()
    kg_anom_valido = df.loc[acs_noto & (df[C_FLAG] == 1), C_TOT].sum()
    n_neg = int((df[C_CUC] < 0).sum())
    print(f"Combustibile ACS: {kg_per_minuto_on:.6f} kg/min con caldaia ON "
          f"(P_focolare = {P_ACS_KW / ETA_CALDAIA:.2f} kW, PCI = {LHV_KJ_KG:.0f} kJ/kg)")
    print(f"Righe: {len(df)}  |  da {df.index.min()} a {df.index.max()}")
    print(f"Tara bombole: {tara_g:.0f} g")
    print(f"Consumo totale bilance (finestra):          {tot_finestra:8.2f} kg")
    print(f"  di cui nei minuti con ACS noto:           {tot_valido:8.2f} kg")
    print(f"Combustibile ACS modello:                   {acs_tot:8.2f} kg")
    print(f"  attribuito (bilancio per episodio):       {acs_att_tot:8.2f} kg")
    print(f"  non osservato dalle bilance:              {acs_noss_tot:8.2f} kg")
    print(f"Consumo cucina (minuti con ACS noto):       {cuc_tot:8.2f} kg")
    if SOGLIA_ANOMALIA_KG_MIN is not None:
        print(f"Minuti anomali bilance (> {SOGLIA_ANOMALIA_KG_MIN} kg/min): {int(df[C_FLAG].sum())}  "
              f"= {kg_anom:.2f} kg nel totale ({kg_anom_valido:.2f} kg nel consumo cucina)")
    print(f"Minuti con consumo cucina negativo:         {n_neg} su {int(df[C_CUC].notna().sum())}")

    # --- Scrittura Excel ------------------------------------------------------
    df.reset_index().to_excel(FILE_USCITA, index=False, sheet_name="Dati")

    print("Formattazione e grafici...")
    wb = load_workbook(FILE_USCITA)
    ws = wb["Dati"]
    ws.freeze_panes = "B2"
    formati = {"time": "dd/mm/yyyy hh:mm",
               C_ACS: "0.000000", C_ACS_CUM: "0.000", C_ACS_ATT: "0.000000", C_ACS_NOSS: "0.000000",
               C_TOT: "0.000000", C_TOT_CUM: "0.000",
               C_CUC: "0.000000", C_CUC_CUM: "0.000",
               C_POT: "0.00", C_P_POT: "0.000",
               C_P_ACS: "0.0000", C_P_CUC: "0.0000", C_P_TOT: "0.0000",
               C_FLAG: "0"}
    for j, cella in enumerate(ws[1], start=1):
        ws.column_dimensions[cella.column_letter].width = 34 if j > 1 else 20
        fmt = formati.get(cella.value)
        if fmt:
            for row in ws.iter_rows(min_row=2, min_col=j, max_col=j):
                row[0].number_format = fmt

    # Foglio con dati decimati per i grafici (solo le colonne che servono)
    cols_graf = [C_PESO_R, C_PESO_L, C_TFUMI, C_P_ACS, C_P_CUC, C_P_TOT, C_P_POT]
    ws_g = wb.create_sheet("Grafico")
    righe = df[cols_graf].iloc[::DECIMAZIONE_GRAFICO].reset_index()
    ws_g.append(list(righe.columns))
    # L'ora va scritta come TESTO: se e' una data Excel usa un asse delle date con passo di 1 giorno
    # e tutti i punti dello stesso giorno finiscono sulla stessa posizione (grafico "a righe verticali").
    for r in righe.itertuples(index=False):
        ws_g.append([f"{r[0]:%d/%m %H:%M}"] + [None if pd.isna(v) else v for v in r[1:]])
    for col in "ABCDEFGH":
        ws_g.column_dimensions[col].width = 20 if col != "A" else 18
    ng = ws_g.max_row
    cats = Reference(ws_g, min_col=1, min_row=2, max_row=ng)

    # Grafico 1: pesi (asse sx) e temperatura fumi (asse dx)
    c1 = LineChart()
    c1.title = "Pesi bombole e temperatura fumi"
    c1.y_axis.title = "Peso (g)"
    c1.x_axis.title = "Tempo"
    c1.width, c1.height = 32, 16
    c1.add_data(Reference(ws_g, min_col=2, max_col=3, min_row=1, max_row=ng), titles_from_data=True)
    c1.set_categories(cats)
    c1.x_axis.tickLblSkip = max(1, ng // 20)
    c1.x_axis.delete = False
    c1.y_axis.delete = False

    c2 = LineChart()
    c2.add_data(Reference(ws_g, min_col=4, max_col=4, min_row=1, max_row=ng), titles_from_data=True)
    c2.y_axis.axId = 200
    c2.y_axis.title = "Temperatura (°C)"
    c2.y_axis.crosses = "max"
    c2.y_axis.delete = False
    senza_sovrapposizioni(c1)
    senza_sovrapposizioni(c2)
    c1 += c2
    ws_g.add_chart(c1, "J2")

    # Grafico 2: profili di consumo ACS e cucina (il totale e' la loro somma, vedi foglio Dati)
    c3 = LineChart()
    c3.title = f"Profili di consumo GPL (media mobile {FINESTRA_PROFILO_MIN} min)"
    c3.y_axis.title = "kg/h"
    c3.x_axis.title = "Tempo"
    c3.width, c3.height = 32, 16
    c3.add_data(Reference(ws_g, min_col=5, max_col=6, min_row=1, max_row=ng), titles_from_data=True)
    c3.set_categories(cats)
    c3.x_axis.tickLblSkip = max(1, ng // 20)
    c3.x_axis.delete = False
    c3.y_axis.delete = False
    senza_sovrapposizioni(c3)
    ws_g.add_chart(c3, "J36")

    # Foglio Episodi: calibrazione della potenza su tutti i cicli
    episodi, n_cicli_tot, scarti = trova_episodi(df)
    cal = None
    if len(episodi):
        cal = crea_foglio_episodi(wb, episodi, n_cicli_tot, scarti, kg_per_minuto_on)
    if cal:
        print(f"Episodi: {cal['n']} ({cal['cicli']} cicli su {n_cicli_tot}) | "
              f"k = {cal['k'] * 1000:.1f} g/min di fiamma [{cal['k_lo'] * 1000:.1f}, {cal['k_hi'] * 1000:.1f}] | "
              f"P utile equivalente = {cal['P']:.2f} kW [{cal['P_lo']:.2f}, {cal['P_hi']:.2f}] | "
              f"rapporto misurato/modello (P = {P_ACS_KW:g} kW) = {cal['rapporto']:.2f}")
        if not (cal["P_lo"] <= P_ACS_KW <= cal["P_hi"]):
            print(f"ATTENZIONE: P_ACS_KW = {P_ACS_KW:g} kW e' fuori dall'IC 95% della calibrazione: valuta di aggiornarlo.")

    # Foglio Eventi (opzionale): cicli isolati, profilo medio e un grafico per ciclo
    cicli = pd.DataFrame()
    if CREA_FOGLIO_EVENTI:
        cicli, _, iso = trova_cicli_isolati(df)
        print(f"Cicli isolati (±{iso} min): {len(cicli)} su {n_cicli_tot}")
        if len(cicli):
            crea_foglio_eventi(wb, ws, df, cicli, n_cicli_tot, iso)

    # Foglio Legenda: definizione delle colonne, parametri e riepilogo
    ws_l = wb.create_sheet("Legenda")
    ws_l.append(["Colonna", "Unità", "Descrizione"])
    legenda = [
        (C_PESO_R + " / " + C_PESO_L, "g", "Peso lordo pulito delle due bombole (cleaned_weight): erogazione alternata, bombola attiva decisa su finestra centrata (FINESTRA_ATTIVA_MIN in weight_cleaning.py)."),
        (C_TFUMI, "°C", "Temperatura fumi pulita (dataset_acs_finale). Vuota nei blackout della termocoppia."),
        (C_ON, "-", "Stato bruciatore ricostruito dalla termocoppia (1 = acceso). Vuoto nei blackout."),
        (C_FIAMMA, "-", f"Minuti a cui si attribuisce combustibile: ON esclusi i primi {MINUTI_SENZA_CONSUMO} min di ogni ciclo."),
        (C_TEXT, "°C", "Temperatura esterna (dataset_acs_finale)."),
        (C_ACS, "kg/min", f"Combustibile per ACS = Fiamma x (P_ACS/eta) x 60 / PCI = Fiamma x {kg_per_minuto_on:.6f}. Fiamma = ON esclusi i primi {MINUTI_SENZA_CONSUMO} min di ogni ciclo (nessun calo misurabile alle bilance). Vuoto nei blackout."),
        (C_ACS_ATT, "kg/min", "ACS attribuito con il bilancio di massa per episodio: in ogni episodio min(ACS modello, consumo misurato - fondo cucina), ripartito al minuto in proporzione al consumo misurato. Fuori dagli episodi = 0."),
        (C_ACS_NOSS, "kg/min", "Diagnostica: parte dell'ACS del modello che le bilance non vedono nell'episodio (modello - attribuito). Non entra nel totale."),
        (C_ACS_CUM, "kg", "Somma progressiva dell'ACS attribuito (solo minuti con dato)."),
        (C_TOT, "kg/min", "Calo di massa netta delle due bombole (ricariche escluse), stessa logica di weight_to_biogas_consumption.py. Disponibile anche nei blackout."),
        (C_TOT_CUM, "kg", "Somma progressiva del consumo totale sull'intera finestra."),
        (C_FLAG, "-", "1 se il calo di peso al minuto supera SOGLIA_ANOMALIA_KG_MIN: non plausibile come consumo (cambio/manovra bombola o rilascio di deriva)."),
        (C_CUC, "kg/min", "Consumo cucina = Consumo totale bilance - ACS attribuito. Sempre >= 0 per costruzione (bilancio di massa per episodio). Vuoto dove l'ACS non è noto."),
        (C_CUC_CUM, "kg", "Somma progressiva del consumo cucina (solo minuti con ACS noto, anomalie incluse)."),
        (C_POT, "kW", f"Potenza termica utile erogata dalla caldaia = Fiamma x {P_ACS_KW:g} kW (rendimento globale {ETA_CALDAIA:g} già incluso)."),
        (C_P_POT, "kW", f"Potenza utile media mobile centrata {FINESTRA_PROFILO_MIN} min (= duty cycle x {P_ACS_KW:g} kW)."),
        (C_P_ACS, "kg/h", f"Consumo ACS attribuito, media mobile centrata {FINESTRA_PROFILO_MIN} min."),
        (C_P_CUC, "kg/h", f"Consumo cucina, media mobile centrata {FINESTRA_PROFILO_MIN} min."),
        (C_P_TOT, "kg/h", f"Consumo totale, media mobile centrata {FINESTRA_PROFILO_MIN} min. Totale = ACS + cucina."),
    ]
    for r in legenda:
        ws_l.append(list(r))
    ws_l.append(["Foglio Episodi", "", "Cicli raggruppati in episodi (separati da <= GAP_EPISODIO_MIN min). Per ogni episodio: eccesso = consumo bilance nella finestra - fondo cucina. "
                                       "k = somma eccessi / somma minuti di fiamma; P utile equivalente = k x rendimento x PCI / 60. Tabella per sottoinsiemi come verifica di coerenza."])
    if CREA_FOGLIO_EVENTI:
        ws_l.append(["Foglio Eventi", "", "Cicli ON isolati (nessun altro ciclo nei ±ISOLAMENTO_MIN minuti, termocoppia valida, senza anomalie/ricariche): tabella, profilo medio e un grafico per ciclo. "
                                      "Rapporto misurato/modello = massa che le bilance rilasciano nel ciclo (al netto del fondo cucina) / combustibile ACS calcolato. "
                                      "Il peso pulito può sottostimare i cicli brevi (filtro sul minimo corrente): leggere il rapporto come limite inferiore."])
    ws_l.append([])
    ws_l.append(["Nota profili", "", "I profili usano la stessa maschera: solo minuti con ACS noto e senza anomalia delle bilance; "
                                      f"servono almeno {FINESTRA_PROFILO_MIN // 2} minuti validi nella finestra, altrimenti vuoto."])
    ws_l.append([])
    ws_l.append(["Parametri", "", ""])
    for nome, val, un in [("P_ACS_KW (potenza utile, verificata sulle bilance)", P_ACS_KW, "kW"), ("ETA_CALDAIA", ETA_CALDAIA, "-"),
                          ("LHV_KJ_KG (PCI GPL)", LHV_KJ_KG, "kJ/kg"),
                          ("MINUTI_SENZA_CONSUMO (inizio ciclo)", MINUTI_SENZA_CONSUMO, "min"), ("Tara bombole", round(tara_g, 1), "g"),
                          ("SOGLIA_ANOMALIA_KG_MIN", SOGLIA_ANOMALIA_KG_MIN if SOGLIA_ANOMALIA_KG_MIN is not None else "nessuna", "kg/min"),
                          ("FINESTRA_PROFILO_MIN", FINESTRA_PROFILO_MIN, "min")]:
        ws_l.append([nome, un, val])
    ws_l.append([])
    ws_l.append(["Riepilogo", "", ""])
    for nome, val in [("Consumo totale bilance (finestra) [kg]", tot_finestra),
                      ("  di cui nei minuti con ACS noto [kg]", tot_valido),
                      ("Combustibile ACS modello [kg]", acs_tot),
                      ("  attribuito con il bilancio per episodio [kg]", acs_att_tot),
                      ("  non osservato dalle bilance [kg]", acs_noss_tot),
                      ("Consumo cucina (minuti con ACS noto) [kg]", cuc_tot),
                      ("Minuti anomali bilance [n]", int(df[C_FLAG].sum())),
                      ("  kg nel totale (finestra) [kg]", kg_anom),
                      ("  kg nel consumo cucina [kg]", kg_anom_valido),
                      ("Minuti con consumo cucina negativo [n]", n_neg),
                      ("Episodi validi [n]", cal["n"] if cal else 0),
                      ("Cicli negli episodi [n]", cal["cicli"] if cal else 0),
                      ("k misurato [g per minuto di fiamma]", float(cal["k"] * 1000) if cal else "n.d."),
                      ("P utile equivalente stimata [kW]", float(cal["P"]) if cal else "n.d."),
                      ("  IC 95% [kW]", f"{cal['P_lo']:.2f} - {cal['P_hi']:.2f}" if cal else "n.d."),
                      ("Rapporto misurato/modello sugli episodi (Σ/Σ)", float(cal["rapporto"]) if cal else "n.d.")]:
        ws_l.append([nome, "", round(val, 3) if isinstance(val, float) else val])
    ws_l.column_dimensions["A"].width = 46
    ws_l.column_dimensions["B"].width = 9
    ws_l.column_dimensions["C"].width = 120

    wb.save(FILE_USCITA)
    print(f"Salvato {FILE_USCITA}")


if __name__ == "__main__":
    main()