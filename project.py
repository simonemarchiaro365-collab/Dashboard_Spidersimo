import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
import csv
from pandas.errors import ParserError
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# 1. CONFIGURAZIONE PAGINA
st.set_page_config(page_title="Gestione Aziendale Pro", layout="wide")

# --- DECORAZIONE: CSS Personalizzato Migliorato ---
st.markdown("""
    <style>
    /* Forza il colore del testo nelle metriche a nero/blu scuro */
    [data-testid="stMetricValue"] {
        color: #1E3A8A !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
    }
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    /* Rendi i titoli più moderni */
    .section-title {
        color: #1E3A8A;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        border-left: 6px solid #3B82F6;
        padding-left: 15px;
        margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CARICAMENTO DATI DA FILE CSV
st.sidebar.title("🛠️ Controllo")

CSV_PATH = 'dati_project_statico.csv'
# Set to True to show repaired/bad CSV line examples for debugging
DEBUG_SHOW_REPAIRS = False

# Proviamo a rilevare il separatore e a leggere in modo robusto: tentiamo di riparare righe con numerici che contengono virgole
try:
    with open(CSV_PATH, 'r', encoding='utf-8', errors='replace') as fh:
        header_line = fh.readline()
        try:
            dialect = csv.Sniffer().sniff(header_line)
            sep = dialect.delimiter
        except Exception:
            sep = ','
except FileNotFoundError:
    st.error(f"File {CSV_PATH} non trovato. Carica il file nella cartella del progetto.")
    st.stop()

# Leggiamo e ripariamo le righe malformate: ricostruiamo i campi numerici finali (PrezzoUnitario, Quantita, ImponibileRiga)
from io import StringIO
rows = []
bad_lines = []
repaired_examples = []
with open(CSV_PATH, 'r', encoding='utf-8', errors='replace') as fh:
    header = fh.readline().strip()
    cols = [c.strip() for c in header.split(sep)]
    expected = len(cols)

    for i, raw in enumerate(fh, start=2):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(sep)
        if len(parts) == expected:
            rows.append(parts)
            continue
        # Tentativo di riparazione: ricostruzione dei campi numerici finali
        tail = parts[8:]
        head = parts[:8]
        repaired = None
        try:
            t = tail.copy()
            # Ricostruisci Imponibile: spesso gli ultimi due frammenti rappresentano 'xxx,cc'
            if len(t) >= 2 and t[-1].strip().isdigit() and len(t[-1].strip()) <= 2:
                imponibile = t[-2].strip() + ',' + t[-1].strip()
                t = t[:-2]
            else:
                imponibile = t[-1].strip()
                t = t[:-1]
            # Ricostruisci PrezzoUnitario
            if len(t) >= 2 and t[1].strip().isdigit() and len(t[1].strip()) <= 2:
                prezzo = t[0].strip() + ',' + t[1].strip()
                quant_parts = t[2:]
            else:
                prezzo = t[0].strip() if t else ''
                quant_parts = t[1:]
            quantita = ''.join(p.strip() for p in quant_parts) if quant_parts else ''

            candidate = head + [prezzo, quantita, imponibile]
            if len(candidate) == expected:
                repaired = candidate
        except Exception:
            repaired = None

        if repaired:
            rows.append(repaired)
            if len(repaired_examples) < 5:
                repaired_examples.append((i, line, sep.join(repaired)))
        else:
            bad_lines.append((i, line))

# Costruiamo il DataFrame dalle righe valide
if rows:
    csv_buf = StringIO()
    csv_buf.write(header + '\n')
    # Normalizziamo i campi numerici finali per evitare virgole non quotate che rompono il CSV
    def sanitize_num_field(s):
        s = str(s).strip()
        s = s.replace('.', '')  # rimuove separatori migliaia
        s = s.replace(',', '.')  # usa punto come separatore decimale
        return s

    for r in rows:
        r_copy = r.copy()
        # Indici attesi: 8=PrezzoUnitario, 9=Quantita, 10=ImponibileRiga
        for idx in [8, 9, 10]:
            if idx < len(r_copy):
                try:
                    r_copy[idx] = sanitize_num_field(r_copy[idx])
                except Exception:
                    r_copy[idx] = r_copy[idx]
        csv_buf.write(sep.join(r_copy) + '\n')
    csv_buf.seek(0)
    df_all = pd.read_csv(csv_buf, sep=sep, dtype=str)
else:
    st.error('Impossibile ricostruire righe valide dal CSV. Controlla il file sorgente.')
    st.stop()

# Report su righe riparate e righe saltate (nascosto di default)
if DEBUG_SHOW_REPAIRS:
    if repaired_examples:
        st.warning(f"Esempi di righe riparate (fino a 5):")
        for ln, orig, new in repaired_examples:
            st.write(f"Riga {ln} — orig: {orig} — riparata: {new}")
    if bad_lines:
        st.warning(f"Saltate {len(bad_lines)} righe troppo malformate per il recovery (mostro fino a 5):")
        for ln, txt in bad_lines[:5]:
            st.write(f"Riga {ln}: {txt}")

# Convertiamo i tipi dove possibile: date e numerici (normalizziamo virgole -> punti e rimuoviamo migliaia)
if 'DataFattura' in df_all.columns:
    df_all['DataFattura'] = pd.to_datetime(df_all['DataFattura'].astype(str), errors='coerce')

for col in ['ImponibileRiga', 'Quantita', 'PrezzoUnitario']:
    if col in df_all.columns:
        # Normalizziamo: sostituiamo virgola con punto (se presente) e rimuoviamo caratteri non numerici
        df_all[col] = df_all[col].astype(str).str.replace(',', '.', regex=False)
        df_all[col] = pd.to_numeric(df_all[col].str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')

# Offriamo due 'tabella' semplificate: Fatture (righe) e Clienti (anagrafica)
lista_tabelle = ['Fatture', 'Clienti']
indice_fatture = 0
scelta_tabella = st.sidebar.selectbox("📂 Seleziona Tabella:", lista_tabelle, index=indice_fatture)

st.sidebar.divider()
st.sidebar.subheader("🔍 Ricerca Rapida")
filtro_testo = st.sidebar.text_input("Cerca valore nella tabella...")

# 4. CARICAMENTO DATI
if scelta_tabella == 'Fatture':
    df_dati = df_all.copy()
else:
    # Costruiamo anagrafica clienti
    cliente_cols = [c for c in ['IdCliente','Nome','Cognome','Nazione','Regione'] if c in df_all.columns]
    df_dati = df_all[cliente_cols].drop_duplicates().reset_index(drop=True) 

# Applica il filtro di ricerca se l'utente scrive qualcosa
if filtro_testo:
    mask = df_dati.astype(str).apply(lambda x: x.str.contains(filtro_testo, case=False)).any(axis=1)
    df_mostrato = df_dati[mask]
else:
    df_mostrato = df_dati

# 5. HEADER E KPI
st.markdown(f'<h1 class="section-title">📊 Dashboard: {scelta_tabella}</h1>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Righe Totali", len(df_dati))
with m2:
    st.metric("Risultati Filtro", len(df_mostrato))
with m3:
    # Mostriamo la somma di 'Spedizione' se esiste
    valore = f" {df_dati['Spedizione'].sum():,.2f}" if 'Spedizione' in df_dati.columns else "N/A"
    st.metric("Volume Spedizioni", valore)
with m4:
    st.metric("Sorgente Dati", f"File: {CSV_PATH}", delta="Statico")

st.divider()

# 6. LAYOUT PRINCIPALE: VISUALS
col_sx, col_dx = st.columns([2, 1])

with col_sx:
    st.subheader("📈 Andamento Mensile Fatturato")
    
   # CASO 1: Tabella Fatture (Grafico Temporale + TOP 10 Clienti)
    if scelta_tabella == 'Fatture':
        # --- AGGIORNATO: Trend Mensile basato su ImponibileRiga ---
        query_trend = """
        SELECT 
            FORMAT(F.DataFattura, 'yyyy-MM') as Mese, 
            SUM(FP.ImponibileRiga) as TotaleFatturato
        FROM Fatture F
        JOIN FattureProdotti FP ON F.IdFattura = FP.IdFattura
        GROUP BY FORMAT(F.DataFattura, 'yyyy-MM') 
        ORDER BY Mese
        """
        
        # Calcoliamo il trend mensile dal CSV
        df_trend = df_all.copy()
        if 'DataFattura' in df_trend.columns:
            df_trend['Mese'] = df_trend['DataFattura'].dt.to_period('M').astype(str)
        else:
            df_trend['Mese'] = pd.to_datetime(df_trend.get('DataFattura', pd.Series([])), errors='coerce').dt.to_period('M').astype(str)
        if 'ImponibileRiga' in df_trend.columns:
            df_trend = df_trend.groupby('Mese')['ImponibileRiga'].sum().reset_index(name='TotaleFatturato').sort_values('Mese')
        else:
            df_trend = pd.DataFrame(columns=['Mese','TotaleFatturato'])

        # Creiamo il grafico ad area
        if not df_trend.empty:
            st.area_chart(df_trend.set_index('Mese'), color="#1E3A8A", height=450)
            # Calcolo del totale per la didascalia
            totale_periodo = df_trend['TotaleFatturato'].sum()
            st.info(f"Valore totale imponibile nel periodo selezionato: **€ {totale_periodo:,.2f}**".replace(',', 'X').replace('.', ',').replace('X', '.'))
        else:
            st.info("Nessun dato disponibile per il trend mensile.")
        st.divider()

    # CASO 2: Tabella Clienti (Grafico per Regione/Nazione)
    elif scelta_tabella == 'Clienti':
        # Creiamo due colonne piccole per mostrare due grafici diversi
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("**Clienti per Nazione**")
            # Contiamo i record per ogni nazione
            df_nazione = df_dati['Nazione'].value_counts().reset_index()
            st.bar_chart(df_nazione.set_index('Nazione'), color="#3B82F6")
            
        with c2:
            st.write("**Clienti per Regione**")
            # Gestiamo i valori mancanti (quelli che erano rosa) rinominandoli
            df_regione = df_dati['Regione'].fillna('Non Specificata').value_counts().reset_index()
            st.bar_chart(df_regione.set_index('Regione'), color="#60A5FA")
            
    # CASO 3: Altre tabelle
    else:
        st.info("💡 Seleziona 'Fatture' o 'Clienti' per visualizzare i grafici avanzati.")
        # Mostra un grafico generico delle prime colonne numeriche
        numeric_cols = df_mostrato.select_dtypes(include=['number']).columns[:2]
        if not numeric_cols.empty:
            st.bar_chart(df_mostrato[numeric_cols].head(20))

with col_dx:
    if scelta_tabella == 'Fatture':
        # --- NUOVO: GRAFICO A TORTA REGIONI ---
        st.subheader("Quota Fatturato per Regione")
        query_torta = """
        SELECT C.Regione, SUM(FP.ImponibileRiga) as Fatturato
        FROM FattureProdotti FP
        JOIN Fatture F ON FP.IdFattura = F.IdFattura
        JOIN Clienti C ON F.IdCliente = C.IdCliente
        GROUP BY C.Regione ORDER BY Fatturato DESC
        """
        try:
            # Raggruppamento per Regione dal CSV
            if 'Regione' in df_all.columns and 'ImponibileRiga' in df_all.columns:
                df_torta = df_all.groupby('Regione', dropna=False)['ImponibileRiga'].sum().reset_index(name='Fatturato').sort_values('Fatturato', ascending=False)
                # Creazione grafico a ciambella interattivo
                if not df_torta.empty:
                    fig = px.pie(df_torta, values='Fatturato', names='Regione', 
                                 color_discrete_sequence=px.colors.sequential.RdBu,
                                 hole=0.4)
                    # Ottimizzazione spazio
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=True, height=450)
                    st.plotly_chart(fig, use_container_width=True)
                    # KPI Rapido: Regione Top
                    top_reg = df_torta.iloc[0]['Regione']
                    st.info(f"📍 Regione dominante: **{top_reg}**")
                else:
                    st.info("Nessun dato per il grafico a torta.")
            else:
                st.info("Colonne 'Regione' o 'ImponibileRiga' mancanti nel file CSV.")
        except Exception as e:
            st.error(f"Errore grafico a torta: {e}")

    else:
        st.info("Seleziona 'Fatture' per vedere i grafici.")

st.divider()

# --- SEZIONE: Grafico Temporale Fatture per Cliente ---
if scelta_tabella == 'Fatture':
        # --- 1. RECUPERO TUTTI I MESI DISPONIBILI (PER ASSE FISSO) ---
        # Lista mesi (uniquement dal CSV)
        if 'DataFattura' in df_all.columns:
            df_tutti_mesi = pd.DataFrame({'Mese': df_all['DataFattura'].dt.to_period('M').astype(str).unique()})
            df_tutti_mesi = df_tutti_mesi.sort_values('Mese').reset_index(drop=True)
        else:
            df_tutti_mesi = pd.DataFrame({'Mese': []})

        # --- 2. FILTRO CLIENTE NELLA SIDEBAR ---
        if {'Nome','Cognome'}.issubset(df_all.columns):
            df_lista_clienti = df_all[['Nome','Cognome']].drop_duplicates()
            df_lista_clienti['Cliente'] = df_lista_clienti['Nome'].astype(str) + ' ' + df_lista_clienti['Cognome'].astype(str)
            lista_nomi = df_lista_clienti['Cliente'].tolist()
            cliente_scelto = st.sidebar.selectbox("👤 Seleziona Cliente per andamento:", lista_nomi)
        else:
            lista_nomi = []
            cliente_scelto = None

        st.subheader(f"📈 Andamento Fatturato: {cliente_scelto}")
        
        # --- 3. QUERY DATI CLIENTE ---
        query_singolo = f"""
        SELECT 
            FORMAT(F.DataFattura, 'yyyy-MM') as Mese, 
            SUM(FP.ImponibileRiga) as TotaleImponibile
        FROM Fatture F
        JOIN FattureProdotti FP ON F.IdFattura = FP.IdFattura
        JOIN Clienti C ON F.IdCliente = C.IdCliente
        WHERE (C.Nome + ' ' + C.Cognome) = '{cliente_scelto}'
        GROUP BY FORMAT(F.DataFattura, 'yyyy-MM')
        """
        
        try:
            # Filtra e aggrega per cliente selezionato
            if cliente_scelto and 'Nome' in df_all.columns and 'Cognome' in df_all.columns and 'ImponibileRiga' in df_all.columns:
                nome, cognome = cliente_scelto.split(' ', 1)
                df_singolo = df_all[(df_all['Nome'] == nome) & (df_all['Cognome'] == cognome)].copy()
                if 'DataFattura' in df_singolo.columns:
                    df_singolo['Mese'] = df_singolo['DataFattura'].dt.to_period('M').astype(str)
                else:
                    df_singolo['Mese'] = pd.to_datetime(df_singolo.get('DataFattura', pd.Series([])), errors='coerce').dt.to_period('M').astype(str)
                df_singolo = df_singolo.groupby('Mese')['ImponibileRiga'].sum().reset_index(name='TotaleImponibile')
            else:
                df_singolo = pd.DataFrame(columns=['Mese','TotaleImponibile'])

            df_completo = pd.merge(df_tutti_mesi, df_singolo, on='Mese', how='left').fillna(0)
            # --- 3. CALCOLO LINEA DI TENDENZA ---
            df_completo['Tendenza'] = df_completo['TotaleImponibile'].rolling(window=3, min_periods=1).mean()
            # Prepariamo il DataFrame per il grafico con due colonne
            df_plot = df_completo.set_index('Mese')[['TotaleImponibile', 'Tendenza']]
            # --- 4. GRAFICO CON DUE COLORI ---
            st.line_chart(df_plot, color=["#2563EB", "#FF4B4B"]) 
            st.caption("🔵 Fatturato Reale | 🔴 Linea di Tendenza (Media Mobile 3 mesi)")
            tot_storico = df_singolo['TotaleImponibile'].sum() if not df_singolo.empty else 0
            st.metric("Fatturato Totale (Imponibile)", f"€ {tot_storico:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
            st.divider()
            
        except Exception as e:
            st.error(f"⚠️ Errore nel calcolo: {e}")

# --- MAPPA INTERATTIVA 3D: FATTURATO PER REGIONE ---
if scelta_tabella == 'Fatture':
    st.divider()
    st.subheader("🗺️ Mappa Interattiva 3D - Densità Fatturato per Regione")
    coordinate_regioni = {
        'Abruzzo': {'lat': 42.3521, 'lon': 13.5898},
        'Basilicata': {'lat': 40.5597, 'lon': 15.8000},
        'Calabria': {'lat': 38.9068, 'lon': 16.5971},
        'Campania': {'lat': 40.8335, 'lon': 14.2681},
        'Emilia-Romagna': {'lat': 44.4949, 'lon': 11.3426},
        'Friuli-Venezia Giulia': {'lat': 45.6558, 'lon': 13.6115},
        'Lazio': {'lat': 41.8719, 'lon': 12.5674},
        'Liguria': {'lat': 44.2693, 'lon': 8.6722},
        'Lombardia': {'lat': 45.6642, 'lon': 9.8100},
        'Marche': {'lat': 43.3168, 'lon': 13.3159},
        'Molise': {'lat': 41.4170, 'lon': 14.6589},
        'Piemonte': {'lat': 44.9359, 'lon': 8.3896},
        'Puglia': {'lat': 41.2033, 'lon': 16.8612},
        'Sardegna': {'lat': 39.9300, 'lon': 8.6699},
        'Sicilia': {'lat': 37.8740, 'lon': 14.2375},
        'Toscana': {'lat': 43.1081, 'lon': 11.8807},
        'Trentino-Alto Adige': {'lat': 46.1910, 'lon': 11.5955},
        'Umbria': {'lat': 42.8703, 'lon': 12.6391},
        'Valle d\'Aosta': {'lat': 45.7367, 'lon': 7.3158},
        'Veneto': {'lat': 45.4072, 'lon': 12.2345},
        # Regioni francesi (per clienti francesi)
        'Île-de-France': {'lat': 48.8566, 'lon': 2.3522},
        'Provence-Alpes-Côte d\'Azur': {'lat': 43.9352, 'lon': 6.0679},
        'Auvergne-Rhône-Alpes': {'lat': 45.5517, 'lon': 4.7738},
        'Bourgogne-Franche-Comté': {'lat': 47.3225, 'lon': 5.0419},
        'Bretagne': {'lat': 48.1173, 'lon': -3.3623},
        'Centre-Val de Loire': {'lat': 47.5941, 'lon': 1.3356},
        'Corse': {'lat': 41.9227, 'lon': 8.7679},
        'Grand Est': {'lat': 48.7399, 'lon': 6.1822},
        'Hauts-de-France': {'lat': 50.3024, 'lon': 2.7766},
        'Nouvelle-Aquitaine': {'lat': 46.2276, 'lon': -0.5595},
        'Normandie': {'lat': 49.1829, 'lon': 0.3705},
        'Occitanie': {'lat': 43.6047, 'lon': 1.4442},
        'Pays de la Loire': {'lat': 47.7210, 'lon': -1.5590}
    }
    
    # Query per ottenere il fatturato per regione (incluso Francia se Nazione='Francia')
    query_mappa = """
    SELECT 
        CASE 
            WHEN C.Nazione = 'Francia' AND C.Regione IS NULL THEN 'Île-de-France'
            ELSE C.Regione 
        END as Regione,
        COUNT(DISTINCT C.IdCliente) as NumClienti,
        SUM(FP.ImponibileRiga) as FatturatoTotale
    FROM Fatture F
    JOIN FattureProdotti FP ON F.IdFattura = FP.IdFattura
    JOIN Clienti C ON F.IdCliente = C.IdCliente
    WHERE (C.Regione IS NOT NULL) OR (C.Nazione = 'Francia')
    GROUP BY CASE 
        WHEN C.Nazione = 'Francia' AND C.Regione IS NULL THEN 'Île-de-France'
        ELSE C.Regione 
    END
    """
    
    try:
        # Costruiamo df_mappa aggregando per regione dal CSV
        if 'Regione' in df_all.columns and 'ImponibileRiga' in df_all.columns:
            df_mappa = df_all.copy()
            # Mappiamo regioni mancanti per Francia
            if 'Regione' in df_mappa.columns and 'Nazione' in df_mappa.columns:
                df_mappa['Regione'] = df_mappa['Regione'].fillna('').astype(str)
                df_mappa.loc[(df_mappa['Nazione']=='Francia') & (df_mappa['Regione']==''), 'Regione'] = 'Île-de-France'
            else:
                df_mappa['Regione'] = df_mappa.get('Regione', pd.Series([''] * len(df_mappa)))
            df_mappa = df_mappa.groupby('Regione', dropna=False).agg(NumClienti=('IdCliente','nunique'), FatturatoTotale=('ImponibileRiga','sum')).reset_index()
        else:
            df_mappa = pd.DataFrame(columns=['Regione','NumClienti','FatturatoTotale'])

        if not df_mappa.empty:
            # Aggiungi le coordinate alle regioni
            df_mappa['Lat'] = df_mappa['Regione'].apply(lambda x: coordinate_regioni.get(x, {}).get('lat', 42.0))
            df_mappa['Lon'] = df_mappa['Regione'].apply(lambda x: coordinate_regioni.get(x, {}).get('lon', 12.0))
            # Normalizza il fatturato per il colore e size
            max_fatturato = df_mappa['FatturatoTotale'].max()
            min_fatturato = df_mappa['FatturatoTotale'].min()
            df_mappa['color_intensity'] = (df_mappa['FatturatoTotale'] - min_fatturato) / (max_fatturato - min_fatturato) if max_fatturato > min_fatturato else 0.5
            # Gradient: Blu (basso) -> Giallo -> Rosso (alto)
            df_mappa['r'] = (df_mappa['color_intensity'] * 255).astype(int)
            df_mappa['g'] = ((1 - abs(df_mappa['color_intensity'] - 0.5) * 2) * 255).astype(int)
            df_mappa['b'] = ((1 - df_mappa['color_intensity']) * 255).astype(int)
            # Crea lista di punti per la heatmap
            points = []
            for idx, row in df_mappa.iterrows():
                try:
                    lat = float(row['Lat'])
                    lon = float(row['Lon'])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        points.append({
                            'latitude': lat,
                            'longitude': lon,
                            'weight': row['FatturatoTotale'] / max_fatturato if max_fatturato > 0 else 0.5,
                            'regione': row['Regione'],
                            'fatturato': row['FatturatoTotale'],
                            'clienti': row['NumClienti'],
                            'r': int(row['r']),
                            'g': int(row['g']),
                            'b': int(row['b'])
                        })
                except:
                    continue
            
            if points:
                # Converte a DataFrame per pydeck
                df_points = pd.DataFrame(points)
                
                # Calcola il centro della mappa (Italia)
                center_lat = 41.8719
                center_lon = 12.5674
                
                # Crea la mappa con Folium in stile scuro
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=6,
                    tiles='CartoDB dark_matter'
                )
                
                # Aggiungi i marcatori colorati
                for idx, row in df_points.iterrows():
                    color_rgb = f'rgb({row["r"]}, {row["g"]}, {row["b"]})'
                    color_hex = '#{:02x}{:02x}{:02x}'.format(row["r"], row["g"], row["b"])
                    
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=row['weight'] * 25,
                        popup=f"<b>{row['regione']}</b><br/>Fatturato: €{row['fatturato']:,.0f}<br/>Clienti: {row['clienti']}",
                        tooltip=f"{row['regione']}",
                        color=color_hex,
                        fill=True,
                        fillColor=color_hex,
                        fillOpacity=0.7,
                        weight=2
                    ).add_to(m)
                
                # Renderizza la mappa
                st_folium(m, use_container_width=True, height=600)
                
                st.caption("🔵 Blu = Fatturato Basso | 🟡 Giallo = Medio | 🔴 Rosso = Alto | Dimensione cerchio = Densità fatturato")
            else:
                st.warning("⚠️ Nessun dato valido trovato per le regioni.")
        else:
            st.info("💡 Nessun dato di fatturato disponibile.")
    except Exception as e:
        st.error(f"Errore nella visualizzazione della mappa: {e}")

# --- CLUSTERING REGIONI: SCATTER PLOT INTERATTIVO (DOPO MAPPA) ---
if scelta_tabella == 'Fatture':
    st.divider()
    st.subheader("🎯 Clustering Regioni - Analisi Comportamento Vendite")
    
    # Query per ottenere i dati aggregati per regione
    query_cluster = """
    SELECT 
        C.Regione,
        COUNT(DISTINCT C.IdCliente) as NumClienti,
        SUM(FP.ImponibileRiga) as FatturatoTotale,
        SUM(FP.Quantita) as QuantitaTotale
    FROM Fatture F
    JOIN FattureProdotti FP ON F.IdFattura = FP.IdFattura
    JOIN Clienti C ON F.IdCliente = C.IdCliente
    WHERE C.Regione IS NOT NULL
    GROUP BY C.Regione
    """
    
    try:
        # Per il clustering usiamo il df_mappa costruito sopra
        df_cluster_data = df_mappa.copy()
        if not df_cluster_data.empty:
            # Assicuriamoci di avere QuantitaTotale (se manca, usiamo 0)
            if 'QuantitaTotale' not in df_cluster_data.columns and 'Quantita' in df_all.columns:
                q = df_all.copy()
                if 'Regione' in q.columns and 'Nazione' in q.columns:
                    q['Regione'] = q['Regione'].fillna('').astype(str)
                    q.loc[(q['Nazione']=='Francia') & (q['Regione']=='') , 'Regione'] = 'Île-de-France'
                quant = q.groupby('Regione')['Quantita'].sum().reset_index(name='QuantitaTotale')
                df_cluster_data = df_cluster_data.merge(quant, on='Regione', how='left')
                df_cluster_data['QuantitaTotale'] = df_cluster_data['QuantitaTotale'].fillna(0)

            X = df_cluster_data[['NumClienti', 'FatturatoTotale']].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df_cluster_data['Cluster'] = kmeans.fit_predict(X_scaled)
            cluster_labels = {0: 'Cluster A', 1: 'Cluster B', 2: 'Cluster C'}
            df_cluster_data['Cluster_Label'] = df_cluster_data['Cluster'].map(cluster_labels)

            fig_cluster = px.scatter(
                df_cluster_data,
                x='NumClienti',
                y='FatturatoTotale',
                color='Cluster_Label',
                size='QuantitaTotale' if 'QuantitaTotale' in df_cluster_data.columns else None,
                hover_name='Regione',
                title='Clustering Regioni: Clienti vs Fatturato',
                labels={'NumClienti': 'Numero Clienti', 'FatturatoTotale': 'Fatturato Totale (€)', 'Cluster_Label': 'Cluster'},
                color_discrete_map={'Cluster A': '#3B82F6', 'Cluster B': '#10B981', 'Cluster C': '#F59E0B'}
            )
            fig_cluster.update_traces(textposition='top center', textfont=dict(size=10), mode='markers+text')
            fig_cluster.update_layout(height=500, hovermode='closest', template='plotly_dark', font=dict(size=12), plot_bgcolor='#1a1a1a')
            st.plotly_chart(fig_cluster, use_container_width=True)

            st.write("### 📊 Analisi per Cluster:")
            col1, col2, col3 = st.columns(3)
            for cluster_id in range(3):
                cluster_name = cluster_labels[cluster_id]
                df_cluster_subset = df_cluster_data[df_cluster_data['Cluster'] == cluster_id]
                with [col1, col2, col3][cluster_id]:
                    st.metric(label=f"**{cluster_name}**", value=len(df_cluster_subset), delta="regioni")
                    st.write(f"**Regioni:** {', '.join(df_cluster_subset['Regione'].dropna().astype(str).tolist())}")
                    st.write(f"**Fatturato Medio:** €{df_cluster_subset['FatturatoTotale'].mean():,.0f}")
                    st.write(f"**Clienti Medi:** {df_cluster_subset['NumClienti'].mean():.0f}")
        else:
            st.warning("⚠️ Nessun dato disponibile per il clustering.")
            
    except Exception as e:
        st.error(f"Errore nel clustering: {e}")

# --- CLIENT HEALTH SCORE: ANALISI RISCHIO CHURN (BASATO SU FREQUENZA ATTESA) ---
if scelta_tabella == 'Fatture':
    st.divider()
    st.subheader("💚 Client Health Score - Analisi Rischio Churn")
    
    # Query per calcolare la salute dei clienti con frequenza attesa
    query_health = """
    SELECT 
        C.IdCliente,
        C.Nome + ' ' + C.Cognome as Cliente,
        COUNT(DISTINCT F.IdFattura) as NumFatture,
        MAX(F.DataFattura) as UltimaFattura,
        MIN(F.DataFattura) as PrimaFattura,
        DATEDIFF(DAY, MIN(F.DataFattura), MAX(F.DataFattura)) as GiorniAttivi,
        DATEDIFF(DAY, MAX(F.DataFattura), (SELECT MAX(DataFattura) FROM Fatture)) as GiorniInattivo,
        SUM(FP.ImponibileRiga) as FatturatoTotale,
        AVG(FP.ImponibileRiga) as FatturatoMedio,
        CAST(DATEDIFF(DAY, MIN(F.DataFattura), MAX(F.DataFattura)) AS FLOAT) / NULLIF(COUNT(DISTINCT F.IdFattura) - 1, 0) as GiorniTraFatture
    FROM Fatture F
    JOIN FattureProdotti FP ON F.IdFattura = FP.IdFattura
    JOIN Clienti C ON F.IdCliente = C.IdCliente
    GROUP BY C.IdCliente, C.Nome, C.Cognome
    HAVING COUNT(DISTINCT F.IdFattura) > 1
    """
    
    try:
        # Creiamo il dataframe health partendo dal CSV
        if {'IdCliente','DataFattura','ImponibileRiga'}.issubset(df_all.columns):
            gp = df_all.groupby('IdCliente')
            df_health = gp.agg(
                Cliente=('IdCliente', lambda x: (df_all.loc[x.index[0],'Nome'] + ' ' + df_all.loc[x.index[0],'Cognome']) if 'Nome' in df_all.columns and 'Cognome' in df_all.columns else x.name),
                NumFatture=('IdFattura', 'nunique'),
                UltimaFattura=('DataFattura', 'max'),
                PrimaFattura=('DataFattura', 'min'),
                FatturatoTotale=('ImponibileRiga','sum')
            ).reset_index()
            # Calcoli aggiuntivi
            df_health['GiorniAttivi'] = (df_health['UltimaFattura'] - df_health['PrimaFattura']).dt.days
            max_date = df_all['DataFattura'].max()
            df_health['GiorniInattivo'] = (max_date - df_health['UltimaFattura']).dt.days
            # Giorni tra fatture (media)
            def avg_days_between(idc):
                dats = sorted(df_all[df_all['IdCliente']==idc]['DataFattura'].dropna().unique())
                if len(dats) <= 1:
                    return float('nan')
                diffs = [(dats[i+1]-dats[i]).days for i in range(len(dats)-1)]
                return sum(diffs)/len(diffs)
            df_health['GiorniTraFatture'] = df_health['IdCliente'].apply(lambda x: avg_days_between(x))
            # Fatturato medio
            df_health['FatturatoMedio'] = df_health['FatturatoTotale'] / df_health['NumFatture']
            df_health = df_health[df_health['NumFatture']>0]
        else:
            df_health = pd.DataFrame()

        if not df_health.empty:
            df_health['RitardoRispettoFrequenza'] = df_health['GiorniInattivo'] / df_health['GiorniTraFatture'].replace(0, 1)
            max_fatture = df_health['NumFatture'].max()
            max_fatturato = df_health['FatturatoMedio'].max()
            df_health['Score_Frequenza'] = (df_health['NumFatture'] / max_fatture) * 35 if max_fatture>0 else 0
            df_health['Score_Valore'] = (df_health['FatturatoMedio'] / max_fatturato) * 35 if max_fatturato>0 else 0
            df_health['Score_Ritardo'] = (1 / (df_health['RitardoRispettoFrequenza'].clip(lower=0.1) + 0.5)) * 30
            df_health['Score_Ritardo'] = df_health['Score_Ritardo'].clip(upper=30)
            df_health['HealthScore'] = (df_health['Score_Frequenza'] + df_health['Score_Valore'] + df_health['Score_Ritardo']).round(1)
            def classify_health(score):
                if score >= 70:
                    return '🟢 Healthy'
                elif score >= 45:
                    return '🟡 At Risk'
                else:
                    return '🔴 Critical'
            df_health['Categoria'] = df_health['HealthScore'].apply(classify_health)

            # KPI per categoria
            st.write("### 📊 Distribuzione Salute Clienti:")
            col_h1, col_h2, col_h3 = st.columns(3)
            healthy = len(df_health[df_health['HealthScore'] >= 70])
            at_risk = len(df_health[(df_health['HealthScore'] >= 45) & (df_health['HealthScore'] < 70)])
            critical = len(df_health[df_health['HealthScore'] < 45])
            with col_h1:
                st.metric(label="🟢 Healthy", value=healthy, delta=f"{(healthy/len(df_health)*100):.0f}%")
            with col_h2:
                st.metric(label="🟡 At Risk", value=at_risk, delta=f"{(at_risk/len(df_health)*100):.0f}%")
            with col_h3:
                st.metric(label="🔴 Critical", value=critical, delta=f"{(critical/len(df_health)*100):.0f}%")

            # Grafico e tabella come prima (riutilizza codice sottostante)
            fig_health = px.scatter(df_health, x='GiorniTraFatture', y='FatturatoMedio', size='HealthScore', color='Categoria', hover_name='Cliente', title='Client Health Score: Frequenza Attesa vs Valore Medio', color_discrete_map={'🟢 Healthy': '#10B981','🟡 At Risk': '#F59E0B','🔴 Critical': '#EF4444'})
            fig_health.update_layout(height=500, template='plotly_dark', hovermode='closest')
            st.plotly_chart(fig_health, use_container_width=True)

            st.caption("💡 Asse X = Giorni medi tra fatture | Bolle grandi = Health Score alto | Colore = Rischio churn")

            st.write("### ⚠️ Clienti Critici - Rischio Churn Elevato:")
            df_critical = df_health[df_health['HealthScore'] < 45].sort_values('HealthScore').head(10)
            if not df_critical.empty:
                df_critical_display = df_critical[[ 'Cliente', 'HealthScore', 'Score_Frequenza', 'Score_Valore', 'Score_Ritardo', 'RitardoRispettoFrequenza', 'GiorniInattivo', 'GiorniTraFatture', 'NumFatture', 'FatturatoMedio' ]].copy()
                df_critical_display['HealthScore'] = df_critical_display['HealthScore'].apply(lambda x: f"{x:.1f}/100")
                df_critical_display['Score_Frequenza'] = df_critical_display['Score_Frequenza'].apply(lambda x: f"{x:.1f}/35")
                df_critical_display['Score_Valore'] = df_critical_display['Score_Valore'].apply(lambda x: f"{x:.1f}/35")
                df_critical_display['Score_Ritardo'] = df_critical_display['Score_Ritardo'].apply(lambda x: f"{x:.1f}/30")
                df_critical_display['RitardoRispettoFrequenza'] = df_critical_display['RitardoRispettoFrequenza'].apply(lambda x: f"{x:.1f}x")
                df_critical_display['GiorniTraFatture'] = df_critical_display['GiorniTraFatture'].apply(lambda x: f"{x:.0f} gg")
                df_critical_display['FatturatoMedio'] = df_critical_display['FatturatoMedio'].apply(lambda x: f"€{x:,.0f}")
                df_critical_display.columns = [ 'Cliente', 'Total Score', 'Freq↓', 'Valore↓', 'Ritardo↓', 'Ritardo vs Abitudini', 'Giorni Inattivo', 'Freq Media', 'Fatture', 'Fatturato Med' ]
                def highlight_min_score(row):
                    colors = [''] * len(row)
                    try:
                        freq_val = float(row['Freq↓'].split('/')[0])
                        valore_val = float(row['Valore↓'].split('/')[0])
                        ritardo_val = float(row['Ritardo↓'].split('/')[0])
                        min_val = min(freq_val, valore_val, ritardo_val)
                        if freq_val == min_val:
                            colors[2] = 'background-color: yellow; font-weight: bold'
                        elif valore_val == min_val:
                            colors[3] = 'background-color: yellow; font-weight: bold'
                        else:
                            colors[4] = 'background-color: red; color: white; font-weight: bold'
                    except:
                        pass
                    return colors
                styled_df = df_critical_display.style.apply(highlight_min_score, axis=1)
                st.dataframe(styled_df, use_container_width=True, hide_index=True, height=350)
                st.write("**📌 Come leggere la tabella:**")
                st.write("""
                - **Total Score**: Somma dei 3 fattori (0-100)
                - **Freq↓**: Score frequenza acquisti (basso = compra poco)
                - **Valore↓**: Score valore medio (basso = spende poco per fattura)
                - **Ritardo↓**: Score puntualità (basso = MOLTO ritardato rispetto al suo solito)
                - **Ritardo vs Abitudini**: Es. 1.5x = compra con 50% più ritardo del solito
                **La cella più scura = il punto debole principale del cliente**
                """)
            else:
                st.success("✅ Nessun cliente critico! Tutti i clienti sono in buona salute.")
        else:
            st.warning("⚠️ Nessun dato disponibile per il health score.")
    except Exception as e:
        st.error(f"Errore nel calcolo del health score: {e}")

# --- TOP 5 PRODOTTI (PRIMA DEL TOP 10) ---
if scelta_tabella == 'Fatture':
    st.divider()
    st.subheader("📦 Top 5 Prodotti Più Venduti")
    
    # Query per i Top Prodotti
    query_p = """
    SELECT TOP 5 P.DescrizioneProdotto, SUM(FP.Quantita) as Quantita, SUM(FP.ImponibileRiga) as Totale
    FROM FattureProdotti FP
    JOIN Prodotti P ON FP.IdProdotto = P.IdProdotto
    GROUP BY P.DescrizioneProdotto ORDER BY Totale DESC
    """
    # Calcolo top prodotti dal CSV (usiamo IdProdotto o Descrizione se presente)
    if 'IdProdotto' in df_all.columns:
        df_p = df_all.groupby(df_all['IdProdotto'].astype(str).rename('Prodotto')).agg(Quantita=('Quantita','sum'), Totale=('ImponibileRiga','sum')).reset_index().sort_values('Totale', ascending=False).head(5)
    else:
        df_p = pd.DataFrame(columns=['Prodotto','Quantita','Totale'])

    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]
    for i, (idx, row) in enumerate(df_p.iterrows()):
        valore_f = f"{row['Totale']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        with cols[i]:
            st.metric(label=str(row['Prodotto'])[:20], value=f"{valore_f} €", delta=f"{int(row['Quantita'])} unità")

    # --- AGGIUNTA 2: Top 10 Prodotti a LARGHEZZA PIENA (SOTTO) ---
    st.divider()
    st.write("📦 **Top 10 Prodotti più Venduti**")
    
    query_top_prodotti = """
    SELECT TOP 10 P.DescrizioneProdotto, SUM(FP.ImponibileRiga) as TotaleVenduto
    FROM FattureProdotti FP
    JOIN Prodotti P ON FP.IdProdotto = P.IdProdotto
    GROUP BY P.DescrizioneProdotto
    ORDER BY TotaleVenduto DESC
    """
    
    try:
        if 'IdProdotto' in df_all.columns:
            df_prodotti = df_all.groupby(df_all['IdProdotto'].astype(str).rename('Prodotto'))['ImponibileRiga'].sum().reset_index().sort_values('ImponibileRiga', ascending=False).head(10).set_index('Prodotto')
            st.bar_chart(df_prodotti, color="#60A5FA", height=400)
        else:
            st.info('Nessun dato prodotti nel CSV.')
    except Exception as e:
        st.error("⚠️ Errore durante il calcolo dei prodotti: {0}".format(e))

# --- SEZIONE PRODOTTI STAGNANTI ---
st.divider()
st.markdown('<h2 class="section-title">🚫 Prodotti Stagnanti (Non venduti da 90+ giorni)</h2>', unsafe_allow_html=True)

query_stagnanti = """
SELECT 
    P.DescrizioneProdotto,
    MAX(F.DataFattura) as UltimaVendita,
    DATEDIFF(DAY, MAX(F.DataFattura), CAST('2020-12-31' AS DATE)) as GiorniInattivo,
    SUM(FP.ImponibileRiga) as FatturatoTotale,
    SUM(FP.Quantita) as QuantitaTotale,
    COUNT(DISTINCT F.IdFattura) as NumeroOrdini
FROM FattureProdotti FP
JOIN Prodotti P ON FP.IdProdotto = P.IdProdotto
JOIN Fatture F ON FP.IdFattura = F.IdFattura
GROUP BY P.DescrizioneProdotto, P.IdProdotto
HAVING DATEDIFF(DAY, MAX(F.DataFattura), CAST('2020-12-31' AS DATE)) >= 90
ORDER BY GiorniInattivo DESC
"""

# Analisi prodotti stagnanti: ultimi 90 giorni rispetto alla massima data disponibile
max_date = df_all['DataFattura'].max() if 'DataFattura' in df_all.columns else pd.Timestamp.today()
if 'IdProdotto' in df_all.columns:
    last_sale = df_all.groupby('IdProdotto').agg(UltimaVendita=('DataFattura','max'), FatturatoTotale=('ImponibileRiga','sum'), QuantitaTotale=('Quantita','sum'), NumeroOrdini=('IdFattura','nunique')).reset_index()
    last_sale['GiorniInattivo'] = (max_date - last_sale['UltimaVendita']).dt.days
    df_stagnanti = last_sale[last_sale['GiorniInattivo'] >= 90].sort_values('GiorniInattivo', ascending=False)
else:
    df_stagnanti = pd.DataFrame()

if not df_stagnanti.empty:
    df_stagnanti_display = df_stagnanti.copy()
    df_stagnanti_display['UltimaVendita'] = pd.to_datetime(df_stagnanti_display['UltimaVendita']).dt.strftime('%d/%m/%Y')
    df_stagnanti_display['FatturatoTotale'] = df_stagnanti_display['FatturatoTotale'].apply(lambda x: f"€{x:,.2f}")
    df_stagnanti_display.columns = [ 'Prodotto', 'Ultimo Acquisto', 'Giorni Fermi', 'Fatturato Tot.', 'Unità', 'Ordini' ]
    
    # Grafico: Prodotti per giorni di inattività
    st.subheader("📉 Timeline Inattività Prodotti")
    fig = px.bar(
        df_stagnanti,
        y='IdProdotto',
        x='GiorniInattivo',
        orientation='h',
        color='GiorniInattivo',
        color_continuous_scale='Reds',
        labels={'GiorniInattivo': 'Giorni Senza Vendite', 'IdProdotto': 'Prodotto'},
        height=400
    )
    fig.update_layout(showlegend=False, xaxis_title='Giorni Senza Vendite')
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabella dettagli
    st.subheader("📋 Dettagli Prodotti Stagnanti")
    st.dataframe(
        df_stagnanti_display,
        use_container_width=True,
        hide_index=True,
        height=350
    )
    
    st.info(f"⚠️ **{len(df_stagnanti)} prodotti non venduti da almeno 90 giorni**. Considera di: rimuoverli dal catalogo, fare promozioni, o bundle con best-seller.")
else:
    st.success("✅ Nessun prodotto stagnante! Tutti i prodotti sono stati venduti negli ultimi 3 mesi.")

# 7. VISUALIZZAZIONE DATI INTEGRALE
st.divider()
st.subheader("📑 Dati Estratti")

# Visualizzazione semplice senza stili (rimuove rosa, verde e ogni altro colore)
st.dataframe(
    df_mostrato, 
    use_container_width=True, 
    height=450,
    hide_index=True # Opzionale: nasconde la colonna degli indici a sinistra
)

# Download
csv = df_mostrato.to_csv(index=False).encode('utf-8')
st.download_button("📥 Esporta Risultati in CSV", data=csv, file_name=f"{scelta_tabella}_export.csv", mime='text/csv')
