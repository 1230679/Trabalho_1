import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Gestão Energética & VE", layout="wide")

@st.cache_data
def load_data():
    try:
        # Leitura direta dos ficheiros Excel
        ip_data = pd.read_excel('IP_data.xlsx', na_values='N/D')
        ptd_data = pd.read_excel('PTD_data.xlsx', na_values='N/D')
        
        # --- LIMPEZA ESSENCIAL (Para evitar erros de Nulos) ---
        ip_data['Potência Instalada Total (W)'] = pd.to_numeric(ip_data['Potência Instalada Total (W)'], errors='coerce').fillna(0)
        
        def limpar_util(v):
            v = str(v).replace('%', '').strip()
            if '-' in v: return float(v.split('-')[-1]) / 100
            if '+' in v: return 1.0
            try: return float(v) / 100
            except: return np.nan
        
        ptd_data['Util_Dec'] = ptd_data['Nível de Utilização [%]'].apply(limpar_util)
        ptd_data = ptd_data.dropna(subset=['Util_Dec'])

        # --- PROCESSAMENTO DAS COORDENADAS (Para o Mapa) ---
        # Assume que o formato é "Lat, Lon"
        coords = ptd_data['Coordenadas Geográficas'].str.split(',', expand=True)
        ptd_data['lat'] = pd.to_numeric(coords[0], errors='coerce')
        ptd_data['lon'] = pd.to_numeric(coords[1], errors='coerce')

        # --- AGRUPAMENTO ---
        ip_concelho = ip_data.groupby(['CodDistritoConcelho', 'Concelho', 'Distrito']).agg(
            P_IP_Total_kW=('Potência Instalada Total (W)', lambda x: x.sum()/1000),
            P_IP_Inef_kW=('Potência Instalada Total (W)', lambda x: x[ip_data.loc[x.index, 'Tipo de Lâmpada'].isin(['Sódio', 'Mercúrio'])].sum()/1000)
        ).reset_index()

        ptd_concelho = ptd_data.groupby(['CodDistritoConcelho']).agg(
            Cap_PTD_kVA=('Potência instalada [kVA]', 'sum'),
            Util_Media=('Util_Dec', 'mean'),
            N_PTDs=('Código de Instalação', 'count'),
            lat=('lat', 'mean'), # Posição média para o mapa
            lon=('lon', 'mean')
        ).reset_index()

        return pd.merge(ip_concelho, ptd_concelho, on='CodDistritoConcelho')
    
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

df_base = load_data()

if df_base is not None:
    # --- SIDEBAR: CENÁRIOS ---
    st.sidebar.header("⚙️ Simulação de VE")
    sim_led = st.sidebar.slider("Eficiência LED (%)", 0, 100, 65) / 100
    sim_ve_perc = st.sidebar.slider("Penetração VE (%)", 0, 100, 60) / 100
    sim_ve_pwr = st.sidebar.select_slider("Potência Carregador (kW)", options=[3.7, 7.4, 11, 22], value=22)

    # --- CÁLCULOS ---
    df = df_base.copy()
    df['Delta_P_LED'] = df['P_IP_Inef_kW'] * sim_led
    df['P_VE'] = df['N_PTDs'] * sim_ve_pwr * sim_ve_perc
    df['P_Folga'] = (df['Cap_PTD_kVA'] * 0.92) * (1 - df['Util_Media'])
    df['D'] = df['P_Folga'] + df['Delta_P_LED'] - df['P_VE']
    df['Viavel'] = df['D'] > 0

    # --- DASHBOARD LAYOUT ---
    st.title("💡 Dashboard de Eficiência e Mobilidade Elétrica")
    
    # Métricas
    c1, c2, c3 = st.columns(3)
    c1.metric("Potência Libertada LED", f"{df['Delta_P_LED'].sum():.1f} kW")
    c2.metric("Carga VE Total", f"{df['P_VE'].sum():.1f} kW")
    c3.metric("Viabilidade Global", f"{(df['Viavel'].mean()*100):.1f}%")

    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏰ Perfis Horários de Consumo IP")
        horas = list(range(18, 25)) + list(range(1, 7))
        base_h = [0.8, 0.9, 1.0, 1.0, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=horas, y=base_h, name="Antes (Sódio/Mercúrio)", line=dict(color='orange')))
        fig_h.add_trace(go.Scatter(x=horas, y=[h*(1-sim_led) for h in base_h], name="Depois (LED)", fill='tonexty'))
        st.plotly_chart(fig_h, use_container_width=True)

    with col2:
        st.subheader("🗺️ Localização e Viabilidade")
        # O Streamlit usa lat/lon para desenhar o mapa
        st.map(df[['lat', 'lon']])

    with st.expander("Ver Tabela de Capacidade Instalada vs Disponível"):
        st.dataframe(df[['Concelho', 'Cap_PTD_kVA', 'P_Folga', 'D', 'Viavel']])