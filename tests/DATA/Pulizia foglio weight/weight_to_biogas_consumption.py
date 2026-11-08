import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference

FRAZIONE_METANO = 0.60  
RHO_METANO = 0.717      
RHO_CO2 = 1.977         

def calcola_e_salva_consumi(file_ingresso="cleaned_weight.xlsx", file_uscita="biogas_consumptions.xlsx"):
    print("Calcolo consumi biogas in corso...")
    df = pd.read_excel(file_ingresso)
    df.set_index('time', inplace=True)

    tare_weights = []
    # Attenzione ai nomi delle colonne che arrivano dal file pulito
    col_r = 'LPG (right) Cleaned'
    col_l = 'LPG (left) Cleaned'
    
    for col in [col_r, col_l]:
        diffs = df[col].diff()
        refill_indices = diffs[diffs > 8000].index
        for idx in refill_indices:
            pos = df.index.get_loc(idx)
            val_before = df[col].iloc[pos-1]
            tare_weights.append(val_before)

    tara_g = np.mean(tare_weights)

    net_r_kg = (df[col_r] - tara_g) / 1000.0
    net_r_kg[net_r_kg < 0] = 0  
    net_l_kg = (df[col_l] - tara_g) / 1000.0
    net_l_kg[net_l_kg < 0] = 0

    dm_r_kg_min = -net_r_kg.diff()
    dm_l_kg_min = -net_l_kg.diff()
    dm_r_kg_min[dm_r_kg_min < 0] = 0
    dm_l_kg_min[dm_l_kg_min < 0] = 0

    mass_flow_r_kgs = dm_r_kg_min / 60.0
    mass_flow_l_kgs = dm_l_kg_min / 60.0
    total_mass_flow_kgs = mass_flow_r_kgs.fillna(0) + mass_flow_l_kgs.fillna(0)
    
    # Applichiamo una media mobile di 1 ora per i grafici
    total_mass_flow_kgh_smooth = (total_mass_flow_kgs * 3600).rolling(window=60, min_periods=1).mean()

    biogas_density = (FRAZIONE_METANO * RHO_METANO) + ((1.0 - FRAZIONE_METANO) * RHO_CO2)
    total_vol_flow_m3s = total_mass_flow_kgs / biogas_density
    
    # Media mobile anche per la portata volumetrica (L/h)
    total_vol_flow_lh_smooth = (total_vol_flow_m3s * 3600.0 * 1000.0).rolling(window=60, min_periods=1).mean()

    df_consumi = pd.DataFrame({
        'time': df.index,
        'Net Mass Right (kg)': net_r_kg.round(3),
        'Net Mass Left (kg)': net_l_kg.round(3),
        'Mass Flow Smooth (kg/h)': total_mass_flow_kgh_smooth.round(3),
        'Vol Flow Smooth (L/h)': total_vol_flow_lh_smooth.round(1)
    })
    
    # Esportazione in Excel
    with pd.ExcelWriter(file_uscita, engine='openpyxl') as writer:
        df_consumi.to_excel(writer, index=False, sheet_name='Dati')
        # Creiamo un foglio vuoto per posizionare i grafici in modo ordinato
        workbook = writer.book
        workbook.create_sheet('Dashboard')
    
    # Costruzione Grafici in Excel
    print("Aggiunta grafici Excel (Dashboard)...")
    wb = load_workbook(file_uscita)
    ws_dati = wb['Dati']
    ws_dash = wb['Dashboard']
    
    max_row = ws_dati.max_row
    cats = Reference(ws_dati, min_col=1, min_row=2, max_row=max_row)

    # Grafico 1: Massa Netta
    chart1 = LineChart()
    chart1.title = f"Contenuto Netto Biogas (Tara: {tara_g/1000:.2f} kg)"
    chart1.y_axis.title = "Massa (kg)"
    chart1.width = 30
    chart1.height = 12
    data1 = Reference(ws_dati, min_col=2, min_row=1, max_col=3, max_row=max_row)
    chart1.add_data(data1, titles_from_data=True)
    chart1.set_categories(cats)
    ws_dash.add_chart(chart1, "B2")

    # Grafico 2: Portata Massica
    chart2 = LineChart()
    chart2.title = "Portata Massica Totale (Media Mobile 1h)"
    chart2.y_axis.title = "kg/h"
    chart2.width = 30
    chart2.height = 12
    data2 = Reference(ws_dati, min_col=4, min_row=1, max_col=4, max_row=max_row)
    chart2.add_data(data2, titles_from_data=True)
    chart2.set_categories(cats)
    ws_dash.add_chart(chart2, "B25")

    # Grafico 3: Portata Volumetrica
    chart3 = LineChart()
    chart3.title = f"Portata Volumetrica Totale (Densità {biogas_density:.2f} kg/m³)"
    chart3.y_axis.title = "L/h"
    chart3.width = 30
    chart3.height = 12
    data3 = Reference(ws_dati, min_col=5, min_row=1, max_col=5, max_row=max_row)
    chart3.add_data(data3, titles_from_data=True)
    chart3.set_categories(cats)
    ws_dash.add_chart(chart3, "B48")

    wb.save(file_uscita)
    print(f"File {file_uscita} salvato con successo. Grafici disponibili nel foglio 'Dashboard'.")

if __name__ == "__main__":
    calcola_e_salva_consumi()