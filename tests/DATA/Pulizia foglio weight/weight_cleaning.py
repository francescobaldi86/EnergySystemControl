import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference

def clean_cylinder(series, refill_threshold=8000, error_drop_threshold=15000):
    clean_series = series.copy()
    series_ffill = series.ffill()
    
    first_valid = series_ffill.first_valid_index()
    if first_valid is None:
        return clean_series
        
    current_val = series_ffill[first_valid]
    
    for i in range(len(series_ffill)):
        if pd.isna(series_ffill.iloc[i]):
            clean_series.iloc[i] = current_val
            continue
            
        val = series_ffill.iloc[i]
        
        if val - current_val > refill_threshold:
            current_val = val
        elif current_val - val > error_drop_threshold:
            pass 
        else:
            if val < current_val:
                current_val = val
                
        clean_series.iloc[i] = current_val
        
    return clean_series

def final_clean(clean_series, active_mask, this_side):
    final = clean_series.copy()
    curr_val = final.iloc[0]
    
    for i in range(1, len(final)):
        val = final.iloc[i]
        if val > curr_val:
            curr_val = val
        elif active_mask.iloc[i] == this_side:
            curr_val = val
        else:
            pass 
        final.iloc[i] = curr_val
        
    return final

def elabora_e_salva_pesi(file_ingresso="weight.xlsx", file_uscita="cleaned_weight.xlsx"):
    print("Elaborazione dati di peso in corso...")
    df = pd.read_excel(file_ingresso)
    df.set_index('time', inplace=True)
    min_physical_weight = 15000

    df_med = df.rolling(window='11min', min_periods=1).median()

    s_r = df_med['LPG (right)'].copy()
    s_l = df_med['LPG (left)'].copy()
    s_r[s_r < min_physical_weight] = np.nan
    s_l[s_l < min_physical_weight] = np.nan

    s_r_clean = clean_cylinder(s_r)
    s_l_clean = clean_cylinder(s_l)

    drop_r = s_r_clean.diff(periods=720).fillna(0)
    drop_l = s_l_clean.diff(periods=720).fillna(0)
    active = pd.Series('none', index=s_r_clean.index)
    active[drop_r < drop_l] = 'right'
    active[drop_l < drop_r] = 'left'
    active = active.replace('none', np.nan).ffill().fillna('right')

    s_r_final = final_clean(s_r_clean, active, 'right')
    s_l_final = final_clean(s_l_clean, active, 'left')

    df_final = pd.DataFrame({
        'time': df.index,
        'LPG (right) Cleaned': s_r_final.values,
        'LPG (left) Cleaned': s_l_final.values
    })
    
    # Salvataggio dati su Excel
    df_final.to_excel(file_uscita, index=False, sheet_name="Dati_Puliti")
    
    # Creazione grafico in Excel con openpyxl
    print("Aggiunta del grafico al file Excel...")
    wb = load_workbook(file_uscita)
    ws = wb["Dati_Puliti"]
    
    chart = LineChart()
    chart.title = "Andamento Pesi Bombole (Alternanza)"
    chart.style = 13
    chart.y_axis.title = "Peso (g)"
    chart.x_axis.title = "Tempo"
    chart.width = 30
    chart.height = 15

    # max_row e max_col per definire i range dei dati
    data = Reference(ws, min_col=2, min_row=1, max_col=3, max_row=ws.max_row)
    cats = Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
    
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    
    # Posiziona il grafico a destra dei dati
    ws.add_chart(chart, "E2")
    
    wb.save(file_uscita)
    print(f"File {file_uscita} salvato con successo con grafico incluso.")

if __name__ == "__main__":
    elabora_e_salva_pesi()