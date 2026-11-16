import pandas as pd
import numpy as np
import openpyxl, os
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import ColorScaleRule
import time

__HERE__ = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# PARAMETRI CALDAIA / COMBUSTIBILE
# ---------------------------------------------------------
P_ACS_UTILE = 10.0            # kW  potenza termica UTILE per l'ACS (ipotesi, verificata sulle bilance: foglio Episodi di dati_puliti_confronto.xlsx)
ETA_BOILER = 0.85             # -   rendimento globale dello scaldabagno (documentazione tecnica)
LHV_BIOGAS = 46055.0          # kJ/kg  potere calorifico inferiore del GPL (11.000 kcal/kg); il nome della variabile e' rimasto per compatibilita'
P_FOCOLARE = P_ACS_UTILE / ETA_BOILER   # kW  potenza al focolare (combustibile) = 11.76 kW
KG_BIOGAS_PER_MIN_ON = P_FOCOLARE * 60.0 / LHV_BIOGAS   # kg/min con caldaia ON

# Primi minuti di ogni ciclo ON senza combustibile attribuito. Dall'analisi a eventi sui cicli isolati
# il primo minuto ON (rilevato dalla salita della temperatura fumi) non produce calo misurabile alle
# bilance (~1 g contro ~15 g/min nei minuti successivi). 0 = comportamento precedente.
MINUTI_SENZA_CONSUMO = 1

def main():
    t0 = time.time()
    input_filename = 'temperature.xlsx'
    output_filename = 'temperature_Qh20_profile_v3.xlsx'
    print(f"Avvio elaborazione di '{input_filename}'...")

    # ---------------------------------------------------------
    # 1. DATA CLEANING & STATE MACHINE PIPELINE
    # ---------------------------------------------------------
    df = pd.read_excel(os.path.join(__HERE__, input_filename), sheet_name='Sheet1')
    df.columns = ['time', 'T_out', 'T_exh']
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # Taglio del periodo iniziale offline
    first_valid = df['T_exh'].first_valid_index()
    df = df.loc[first_valid:].copy().reset_index(drop=True)

    # Rimozione glitch elettrici isolati
    valid = df.dropna(subset=['T_exh']).copy()
    valid['prev_T'] = valid['T_exh'].shift(1)
    valid['next_T'] = valid['T_exh'].shift(-1)
    valid['prev_dt'] = (valid['time'] - valid['time'].shift(1)).dt.total_seconds() / 60.0
    valid['next_dt'] = (valid['time'].shift(-1) - valid['time']).dt.total_seconds() / 60.0
    weight = valid['prev_dt'] / (valid['prev_dt'] + valid['next_dt'])
    valid['interp_neighbor'] = valid['prev_T'] + (valid['next_T'] - valid['prev_T']) * weight
    valid['dev'] = (valid['T_exh'] - valid['interp_neighbor']).abs()
    valid['neighbor_diff'] = (valid['next_T'] - valid['prev_T']).abs()
    
    glitch_mask = ((valid['prev_dt'] <= 30) & (valid['next_dt'] <= 30) & 
                   (valid['dev'] > 10.0) & (valid['neighbor_diff'] < 3.5))

    df['T_exh_clean'] = df['T_exh'].copy()
    df.loc[valid[glitch_mask].index, 'T_exh_clean'] = np.nan

    # Interpolazione dei buchi da deadband
    is_na_exh = df['T_exh_clean'].isna()
    gap_sizes_exh = is_na_exh.groupby((~is_na_exh).cumsum()).transform('sum')
    # df['T_exh_clean'] = df['T_exh_clean'].interpolate(method='linear')
    df['T_exh_clean'] = df['T_exh_clean'].ffill()
    df.loc[gap_sizes_exh > 60, 'T_exh_clean'] = np.nan

    # Interpolazione T_out allineata ai blackout della termocoppia
    df['T_out_clean'] = df['T_out'].interpolate(method='linear').bfill().ffill()
    df.loc[df['T_exh_clean'].isna(), 'T_out_clean'] = np.nan

    # Macchina a stati ON/OFF
    df['T_base'] = df['T_exh_clean'].rolling(window=180, min_periods=1).min()
    df['delta_T_base'] = df['T_exh_clean'] - df['T_base']
    df['dT_dt'] = df['T_exh_clean'].diff()

    T_exh_arr = df['T_exh_clean'].to_numpy()
    dT_arr = df['dT_dt'].to_numpy()
    delta_T_arr = df['delta_T_base'].to_numpy()
    n = len(df)
    state = np.zeros(n, dtype=float)

    for t in range(1, n):
        if np.isnan(T_exh_arr[t]) or np.isnan(dT_arr[t]):
            state[t] = np.nan
            continue
        prev_on = (state[t - 1] == 1.0)
        next_dT = dT_arr[t + 1] if (t + 1 < n and not np.isnan(dT_arr[t + 1])) else 0.0
        
        if not prev_on:
            if dT_arr[t] >= 2.0 or (dT_arr[t] >= 1.2 and next_dT >= 1.2):
                state[t] = 1.0
            else:
                state[t] = 0.0
        else:
            if (dT_arr[t] >= 0.4 and delta_T_arr[t] >= 5.0) or (delta_T_arr[t] >= 15.0 and dT_arr[t] > -1.2):
                state[t] = 1.0
            else:
                state[t] = 0.0

    # Filtro contro micro-oscillazioni
    for t in range(1, n - 1):
        if state[t] == 1.0 and state[t - 1] == 0.0 and state[t + 1] == 0.0:
            if dT_arr[t] < 5.0:
                state[t] = 0.0

    df['Boiler_ON'] = state

    # Cycle ID
    is_on_bool = (df['Boiler_ON'] == 1.0)
    cycle_starts = is_on_bool & (~is_on_bool.shift(1, fill_value=False))
    df['Cycle_ID'] = np.where(df['Boiler_ON'] == 1.0, cycle_starts.cumsum(), np.where(df['Boiler_ON'] == 0.0, 0, np.nan))

    # Fiamma_ON: minuti a cui si attribuisce combustibile (Boiler_ON senza i primi MINUTI_SENZA_CONSUMO di ogni ciclo)
    minuto_nel_ciclo = is_on_bool.groupby(cycle_starts.cumsum()).cumsum()      # 1, 2, 3... dentro ogni ciclo
    df['Fiamma_ON'] = np.where(is_on_bool & (minuto_nel_ciclo > MINUTI_SENZA_CONSUMO), 1.0,
                               np.where(df['Boiler_ON'].isna(), np.nan, 0.0))


    # ---------------------------------------------------------
    # 2. MODELLO POTENZA UTILE ACS (rendimento globale costante)
    # ---------------------------------------------------------
    P_nom = P_FOCOLARE            # potenza al focolare [kW] = P_ACS / eta
    m_comb_biogas = P_nom / LHV_BIOGAS   # portata combustibile con caldaia ON [kg/s]

    # Rendimento globale costante: la potenza utile con caldaia ON e' P_ACS_UTILE
    # (le perdite al camino sono gia' comprese nel rendimento globale 0.9)
    df['Qh20_profile_kW'] = np.where(
        df['Fiamma_ON'] == 1.0, P_ACS_UTILE,
        np.where(df['Fiamma_ON'] == 0.0, 0.0, np.nan)
    )
    df['Eh20_step_kWh'] = (df['Qh20_profile_kW'] / 60.0).round(6)

    df['Biogas_Consumato_kg'] = (df['Fiamma_ON'] * KG_BIOGAS_PER_MIN_ON).round(6)   # kg/min = (P_utile/eta) * 60 s/min / PCI
    # ---------------------------------------------------------
    # 3. PREPARAZIONE TABELLE AGGREGATE
    # ---------------------------------------------------------
    df['Date'] = df['time'].dt.date
    df['Hour'] = df['time'].dt.hour

    daily_df = df.groupby('Date').agg(
        Minuti_Validi=('T_exh_clean', lambda x: x.notna().sum()),
        Minuti_Blackout=('T_exh_clean', lambda x: x.isna().sum()),
        T_out_Media_C=('T_out_clean', 'mean'),
        T_out_Min_C=('T_out_clean', 'min'),
        T_out_Max_C=('T_out_clean', 'max'),
        T_exh_Media_C=('T_exh_clean', 'mean'),
        T_exh_Max_C=('T_exh_clean', 'max'),
        Numero_Cicli_ON=('Cycle_ID', lambda x: x[x > 0].nunique()),
        Minuti_Boiler_ON=('Boiler_ON', lambda x: (x == 1.0).sum()),
        Energia_ACS_kWh=('Eh20_step_kWh', 'sum'),
        Biogas_Consumato_kg=('Biogas_Consumato_kg', 'sum')
    ).reset_index()

    daily_df['Ore_Boiler_ON'] = (daily_df['Minuti_Boiler_ON'] / 60.0).round(2)
    daily_df['Stato_Acquisizione'] = np.where(
        daily_df['Minuti_Validi'] == 0, 'BLACKOUT TOTALE',
        np.where(daily_df['Minuti_Blackout'] > 0, 'PARZIALE', 'COMPLETO')
    )
    daily_df = daily_df[['Date', 'Stato_Acquisizione', 'Minuti_Validi', 'T_out_Media_C', 'T_out_Min_C', 'T_out_Max_C', 
                         'T_exh_Media_C', 'T_exh_Max_C', 'Numero_Cicli_ON', 'Minuti_Boiler_ON', 'Ore_Boiler_ON', 
                         'Energia_ACS_kWh', 'Biogas_Consumato_kg']].round(3)

    on_df = df[df['Boiler_ON'] == 1.0].copy()
    cycles_df = on_df.groupby('Cycle_ID').agg(
        Inizio_Ciclo=('time', 'min'),
        Fine_Spinta=('time', 'max'),
        Durata_ON_min=('Boiler_ON', 'count'),
        T_out_Media_C=('T_out_clean', 'mean'),
        T_exh_Inizio_C=('T_exh_clean', 'first'),
        T_exh_Picco_C=('T_exh_clean', 'max'),
        T_exh_Media_C=('T_exh_clean', 'mean'),
        Gradiente_Max_C_min=('dT_dt', 'max'),
        Qh20_Media_kW=('Qh20_profile_kW', 'mean'),
        Energia_Ciclo_kWh=('Eh20_step_kWh', 'sum')
    ).reset_index()
    cycles_df['Cycle_ID'] = cycles_df['Cycle_ID'].astype(int)
    cycles_df = cycles_df.round({'T_out_Media_C': 2, 'T_exh_Inizio_C': 2, 'T_exh_Picco_C': 2, 'T_exh_Media_C': 2, 
                                 'Gradiente_Max_C_min': 2, 'Qh20_Media_kW': 4, 'Energia_Ciclo_kWh': 4})

    valid_df = df[df['T_exh_clean'].notna()].copy()
    hourly_df = valid_df.groupby('Hour').agg(
        Minuti_Osservati=('T_exh_clean', 'count'),
        T_out_Media_C=('T_out_clean', 'mean'),
        T_exh_Media_C=('T_exh_clean', 'mean'),
        Minuti_Totali_ON=('Boiler_ON', lambda x: (x == 1.0).sum()),
        Energia_Totale_kWh=('Eh20_step_kWh', 'sum')
    ).reset_index()
    hourly_df['Fascia_Oraria'] = hourly_df['Hour'].apply(lambda h: f"{h:02d}:00 - {h:02d}:59")
    hourly_df['Duty_Cycle_Pct'] = (100.0 * hourly_df['Minuti_Totali_ON'] / hourly_df['Minuti_Osservati']).round(2)
    hourly_df['Minuti_ON_Medi_Giorno'] = (60.0 * hourly_df['Minuti_Totali_ON'] / hourly_df['Minuti_Osservati']).round(2)
    hourly_df['Energia_Media_Giornaliera_kWh'] = (hourly_df['Energia_Totale_kWh'] / (hourly_df['Minuti_Osservati'] / 60.0)).round(4)
    hourly_df = hourly_df[['Hour', 'Fascia_Oraria', 'Minuti_Osservati', 'T_out_Media_C', 'T_exh_Media_C', 
                           'Minuti_Totali_ON', 'Duty_Cycle_Pct', 'Minuti_ON_Medi_Giorno', 'Energia_Totale_kWh', 
                           'Energia_Media_Giornaliera_kWh']].round(3)

    df_export = df[['time', 'T_out_clean', 'T_exh_clean', 'dT_dt', 'Boiler_ON', 'Fiamma_ON', 'Cycle_ID', 'Qh20_profile_kW', 'Eh20_step_kWh', 'Biogas_Consumato_kg']].copy()
    df_export.columns = ['Time', 'Outdoor_Temp_C', 'Exhaust_Gas_Temp_C', 'dT_exh_dt_C_min', 'Boiler_ON', 'Fiamma_ON', 'Cycle_ID', 'Qh20_profile_kW', 'Eh20_step_kWh', 'Biogas_Consumato_kg']
    for col in ['Outdoor_Temp_C', 'Exhaust_Gas_Temp_C', 'dT_exh_dt_C_min']:
        df_export[col] = df_export[col].round(3)

    # ---------------------------------------------------------
    # 4. EXPORT BULK E FORMATTAZIONE EXCEL CON OPENPYXL
    # ---------------------------------------------------------
    print("Scrittura tabelle base su Excel...")
    with pd.ExcelWriter(os.path.join(__HERE__, output_filename), engine='openpyxl') as writer:
        pd.DataFrame().to_excel(writer, sheet_name='Sintesi_e_Modello', index=False)
        daily_df.to_excel(writer, sheet_name='Riepilogo_Giornaliero', index=False, startrow=3)
        cycles_df.to_excel(writer, sheet_name='Catalogo_Cicli_ON', index=False, startrow=3)
        hourly_df.to_excel(writer, sheet_name='Profilo_Orario_24h', index=False, startrow=3)
        df_export.to_excel(writer, sheet_name='Dataset_Pulito_1min', index=False)

    print("Applicazione stili e generazione grafici...")
    wb = openpyxl.load_workbook(os.path.join(__HERE__, output_filename))

    # Definizione Stili Costanti
    NAVY_HDR = PatternFill(start_color="1F3A52", end_color="1F3A52", fill_type="solid")
    TEAL_SUB = PatternFill(start_color="2E5B70", end_color="2E5B70", fill_type="solid")
    ZEBRA = PatternFill(start_color="F4F7F9", end_color="F4F7F9", fill_type="solid")
    TOTAL = PatternFill(start_color="E3ECF2", end_color="E3ECF2", fill_type="solid")
    
    F_TITLE = Font(name="Calibri", size=15, bold=True, color="1F3A52")
    F_SUBTITLE = Font(name="Calibri", size=10, italic=True, color="555555")
    F_SEC_HDR = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    F_TBL_HDR = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    F_BOLD = Font(name="Calibri", size=10, bold=True, color="1F3A52")
    F_REG = Font(name="Calibri", size=10, color="222222")

    THIN = Side(border_style="thin", color="D0D7DE")
    DBL_BTM = Side(border_style="double", color="1F3A52")
    MED_BTM = Side(border_style="medium", color="1F3A52")

    B_DATA = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    B_TOT = Border(top=THIN, bottom=DBL_BTM, left=THIN, right=THIN)
    B_HDR = Border(left=THIN, right=THIN, top=THIN, bottom=MED_BTM)

    def style_table(ws, title, subtitle, n_rows, n_cols, num_formats, totals=None):
        ws.views.sheetView[0].showGridLines = True
        ws['A1'], ws['A1'].font = title, F_TITLE
        ws['A2'], ws['A2'].font = subtitle, F_SUBTITLE
        ws.freeze_panes = "A5"
        ws.auto_filter.ref = f"A4:{get_column_letter(n_cols)}{4 + n_rows}"

        for c_i in range(1, n_cols + 1):
            c = ws.cell(row=4, column=c_i)
            c.font, c.fill, c.border = F_TBL_HDR, NAVY_HDR, B_HDR
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.row_dimensions[4].height = 26

        for r in range(5, 5 + n_rows):
            is_zebra = (r % 2 == 1)
            for c_i in range(1, n_cols + 1):
                c = ws.cell(row=r, column=c_i)
                c.font, c.border = F_REG, B_DATA
                if is_zebra: c.fill = ZEBRA
                if c_i in num_formats: c.number_format = num_formats[c_i]

        if totals:
            tot_r = 5 + n_rows
            for c_i in range(1, n_cols + 1):
                c = ws.cell(row=tot_r, column=c_i)
                c.font, c.fill, c.border = F_BOLD, TOTAL, B_TOT
                if c_i in totals: c.value = totals[c_i]
                if c_i in num_formats: c.number_format = num_formats[c_i]
            for c_i in range(1, n_cols + 1):
                ws.column_dimensions[get_column_letter(c_i)].width = 18

    # TAB 1: DASHBOARD
    ws_dash = wb['Sintesi_e_Modello']
    ws_dash.views.sheetView[0].showGridLines = True
    ws_dash['A1'], ws_dash['A1'].font = "ANALISI TERMODINAMICA E PROFILO DI PRODUZIONE ACS (CALDAIA A BIOGAS ON/OFF)", F_TITLE
    ws_dash['A2'], ws_dash['A2'].font = "Campagna di monitoraggio: 25 Giugno 2026 – 23 Settembre 2026 | Passo: 1 minuto", F_SUBTITLE

    ws_dash.merge_cells('A4:C4')
    ws_dash['A4'], ws_dash['A4'].font, ws_dash['A4'].fill = "1. RIEPILOGO DATA CLEANING E KPI OPERATIVI", F_SEC_HDR, NAVY_HDR

    minuti_on = int((df['Boiler_ON'] == 1.0).sum())
    n_cicli = len(cycles_df)
    kpis = [
        ("Inizio acquisizione validata", "2026-06-25 12:41", "Tagliati 15.101 min iniziali offline"),
        ("Fine acquisizione", "2026-09-23 14:59", "Finestra totale: 129.739 minuti"),
        ("Minuti validi", 95122, "Copertura 73,3%"),
        ("Minuti in blackout datalogger", 34617, "6 interruzioni prolungate ignorate"),
        ("Outlier impulsivi rimossi", 5, "Spike elettrici senza coda termica"),
        ("Temperatura fumi globale (T_exh)", 19.46, "°C (Media totale)"),
        ("Temperatura fumi in spinta (ON)", round(float(df.loc[df['Boiler_ON'] == 1.0, 'T_exh_clean'].mean()), 2), "°C (media con caldaia ON)"),
        ("Numero totale cicli ON", n_cicli, "Filtrati micro-inneschi <= 1 min"),
        ("Tempo totale bruciatore acceso", minuti_on, f"minuti ({minuti_on / 60.0:.2f} ore)"),
        ("Durata media ciclo ACS", round(minuti_on / max(n_cicli, 1), 2), "minuti/ciclo"),
        ("Energia termica utile prodotta", round(float(df['Eh20_step_kWh'].sum()), 2), "kWh complessivi"),
        ("Consumo totale stimato GPL (ACS)", round(float(df['Biogas_Consumato_kg'].sum()), 2), f"kg GPL (PCI = {LHV_BIOGAS:,.0f} kJ/kg)".replace(",", "."))
    ]
    for idx, (lbl, val, note) in enumerate(kpis, start=5):
        ws_dash.cell(row=idx, column=1, value=lbl).font = F_REG
        c_val = ws_dash.cell(row=idx, column=2, value=val)
        c_val.font, c_val.alignment = F_BOLD, Alignment(horizontal="right")
        if isinstance(val, float): c_val.number_format = "0.0%" if val < 1.0 else "#,##0.00"
        elif isinstance(val, int): c_val.number_format = "#,##0"
        ws_dash.cell(row=idx, column=3, value=note).font = F_SUBTITLE
        for c_i in range(1, 4):
            cell = ws_dash.cell(row=idx, column=c_i)
            cell.border = B_DATA
            if idx % 2 == 1: cell.fill = ZEBRA

    ws_dash.merge_cells('E4:G4')
    ws_dash['E4'], ws_dash['E4'].font, ws_dash['E4'].fill = "2. PARAMETRI TERMODINAMICI APPLICATI", F_SEC_HDR, NAVY_HDR

    for c_i, h_txt in enumerate(["Parametro / Grandezza", "Unità", "Valore"], start=5):
        c = ws_dash.cell(row=5, column=c_i, value=h_txt)
        c.font, c.fill, c.border, c.alignment = F_TBL_HDR, TEAL_SUB, B_HDR, Alignment(horizontal="center")

    model_rows = [
        ("Potenza utile ACS (P_ACS)", "kW", P_ACS_UTILE),
        ("Rendimento globale caldaia", "-", ETA_BOILER),
        ("Potenza al focolare (P_nom = P_ACS / eta)", "kW", round(P_nom, 4)),
        ("Potere Calorifico Inferiore (LHV)", "kJ/kg", int(LHV_BIOGAS)),
        ("Portata combustibile (m_comb)", "kg/s", float(m_comb_biogas)),
        ("Consumo combustibile con caldaia ON", "kg/h", round(KG_BIOGAS_PER_MIN_ON * 60.0, 4)),
        ("Minuti iniziali per ciclo senza consumo", "min", MINUTI_SENZA_CONSUMO),
    ]
    for idx, (p, u, v) in enumerate(model_rows, start=6):
        ws_dash.cell(row=idx, column=5, value=p).font = F_REG
        c_u = ws_dash.cell(row=idx, column=6, value=u)
        c_u.font, c_u.alignment = F_SUBTITLE, Alignment(horizontal="center")
        c_v = ws_dash.cell(row=idx, column=7, value=v)
        c_v.font, c_v.alignment = F_BOLD, Alignment(horizontal="right")
        if isinstance(v, float): c_v.number_format = "0.000000" if v < 0.01 else ("0.0%" if v < 1.0 else "0.0000")
        elif isinstance(v, int): c_v.number_format = "#,##0"
        for c_i in range(5, 8):
            cell = ws_dash.cell(row=idx, column=c_i)
            cell.border = B_DATA
            if idx % 2 == 1: cell.fill = ZEBRA

    ws_dash.column_dimensions['A'].width, ws_dash.column_dimensions['B'].width = 38, 20
    ws_dash.column_dimensions['C'].width, ws_dash.column_dimensions['E'].width = 44, 40
    ws_dash.column_dimensions['F'].width, ws_dash.column_dimensions['G'].width = 12, 18

    # TAB 2: RIEPILOGO GIORNALIERO
    ws_d = wb['Riepilogo_Giornaliero']
    n_d = len(daily_df)
    f_d = {1: "yyyy-mm-dd", 3: "#,##0", 4: "0.00", 5: "0.00", 6: "0.00", 7: "0.00", 8: "0.00", 9: "#,##0", 10: "#,##0", 11: "0.00", 12: "0.000", 13: "0.000"}
    tot_d = {
        1: "TOT / MEDIA", 2: f"=COUNTIF(B5:B{4+n_d},\"COMPLETO\")&\" gg comp\"", 3: f"=SUM(C5:C{4+n_d})",
        4: f"=AVERAGE(D5:D{4+n_d})", 5: f"=MIN(E5:E{4+n_d})", 6: f"=MAX(F5:F{4+n_d})", 7: f"=AVERAGE(G5:G{4+n_d})",
        8: f"=MAX(H5:H{4+n_d})", 9: f"=SUM(I5:I{4+n_d})", 10: f"=SUM(J5:J{4+n_d})", 11: f"=SUM(K5:K{4+n_d})",
        12: f"=SUM(L5:L{4+n_d})", 13: f"=SUM(M5:M{4+n_d})"
    }
    style_table(ws_d, "RIEPILOGO GIORNALIERO PRODUZIONE ACS", "Statistiche per giorno", n_d, 13, f_d, tot_d)
    ws_d.conditional_formatting.add(f"L5:L{4+n_d}", ColorScaleRule(start_type="min", start_color="F4F7F9", mid_type="percentile", mid_value=50, mid_color="A6C8E0", end_type="max", end_color="2E5B70"))

    # TAB 3: CATALOGO CICLI
    ws_c = wb['Catalogo_Cicli_ON']
    n_c = len(cycles_df)
    f_c = {1: "#,##0", 2: "yyyy-mm-dd hh:mm", 3: "yyyy-mm-dd hh:mm", 4: "#,##0", 5: "0.00", 6: "0.00", 7: "0.00", 8: "0.00", 9: "0.00", 10: "0.0000", 11: "0.0000"}
    tot_c = {
        1: "TOT / MEDIA", 4: f"=SUM(D5:D{4+n_c})", 5: f"=AVERAGE(E5:E{4+n_c})", 6: f"=AVERAGE(F5:F{4+n_c})",
        7: f"=MAX(G5:G{4+n_c})", 8: f"=AVERAGE(H5:H{4+n_c})", 9: f"=MAX(I5:I{4+n_c})",
        10: f"=AVERAGE(J5:J{4+n_c})", 11: f"=SUM(K5:K{4+n_c})"
    }
    style_table(ws_c, "CATALOGO CICLI BRUCIATORE", "Dettaglio singoli inneschi", n_c, 11, f_c, tot_c)

    # TAB 4: PROFILO ORARIO
    ws_h = wb['Profilo_Orario_24h']
    n_h = len(hourly_df)
    f_h = {1: "0", 3: "#,##0", 4: "0.00", 5: "0.00", 6: "#,##0", 7: "0.00", 8: "0.00", 9: "0.000", 10: "0.0000"}
    tot_h = {
        2: "TOT / MEDIA", 3: f"=SUM(C5:C{4+n_h})", 4: f"=AVERAGE(D5:D{4+n_h})", 5: f"=AVERAGE(E5:E{4+n_h})",
        6: f"=SUM(F5:F{4+n_h})", 7: f"=AVERAGE(G5:G{4+n_h})", 8: f"=SUM(H5:H{4+n_h})", 9: f"=SUM(I5:I{4+n_h})", 10: f"=SUM(J5:J{4+n_h})"
    }
    style_table(ws_h, "PROFILO ORARIO TIPO (24H)", "Andamento domanda ACS per ora del giorno", n_h, 10, f_h, tot_h)
    ws_h.conditional_formatting.add(f"J5:J{4+n_h}", ColorScaleRule(start_type="min", start_color="F4F7F9", mid_type="percentile", mid_value=50, mid_color="A6C8E0", end_type="max", end_color="2E5B70"))

    # CHARTS (Dashboard)
    ch_d = BarChart(); ch_d.style = 10; ch_d.title = "Energia ACS Giornaliera (kWh)"; ch_d.width = 16; ch_d.height = 9
    ch_d.add_data(Reference(ws_d, min_col=12, min_row=4, max_row=4+n_d), titles_from_data=True)
    ch_d.set_categories(Reference(ws_d, min_col=1, min_row=5, max_row=4+n_d))
    ch_d.legend = None; ws_dash.add_chart(ch_d, "A19")

    ch_h = BarChart(); ch_h.style = 11; ch_h.title = "Profilo Orario Energia Media (kWh)"; ch_h.width = 15; ch_h.height = 9
    ch_h.add_data(Reference(ws_h, min_col=10, min_row=4, max_row=4+n_h), titles_from_data=True)
    ch_h.set_categories(Reference(ws_h, min_col=2, min_row=5, max_row=4+n_h))
    ch_h.legend = None; ws_dash.add_chart(ch_h, "E19")

    # TAB 5: DATASET
    ws_data = wb['Dataset_Pulito_1min']
    ws_data.freeze_panes = "A2"
    ws_data.auto_filter.ref = f"A1:I{len(df_export) + 1}"
    ws_data.row_dimensions[1].height = 24
    for c_i in range(1, 10):
        c = ws_data.cell(row=1, column=c_i)
        c.font, c.fill, c.alignment = F_TBL_HDR, NAVY_HDR, Alignment(horizontal="center", vertical="center")
        ws_data.column_dimensions[get_column_letter(c_i)].width = 22

    print("Salvataggio file in corso...")
    wb.save(os.path.join(__HERE__, output_filename))
    # ---------------------------------------------------------
    # 5. ESPORTAZIONE AGGIUNTIVA DEL SOLO DATASET (CSV ed EXCEL)
    # ---------------------------------------------------------
    print("Esportazione del dataset autonomo in formato CSV e XLSX...")
    csv_output = 'dataset_acs_finale.csv'
    excel_output = 'dataset_acs_finale.xlsx'
    
    df_export.to_csv(os.path.join(__HERE__, csv_output), index=False)
    df_export.to_excel(os.path.join(__HERE__, excel_output), index=False)
    print(f"Completato in {time.time() - t0:.2f} s. File salvato: {output_filename}")

if __name__ == "__main__":
    main()