import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Simulador ANADI - VE & LED", layout="wide")

# --- FUNÇÕES DE CARREGAMENTO E LIMPEZA (IGUAL AO TEU NOTEBOOK) ---
@st.cache_data
def load_data():
    try:
        # 1. Leitura Limpa (Excel não usa encoding nem on_bad_lines)
        ip_data = pd.read_excel('IP_data.xlsx', na_values='N/D')
        ptd_data = pd.read_excel('PTD_data.xlsx', na_values='N/D')
        
        ip_data['Potência Instalada Total (W)'] = pd.to_numeric(ip_data['Potência Instalada Total (W)'], errors='coerce').fillna(0)
        
        # 3. Processar Utilização (PTD)
        def limpar_util(v):
            v = str(v).replace('%', '').strip()
            if '-' in v: return float(v.split('-')[-1]) / 100
            if '+' in v: return 1.0
            try: return float(v) / 100
            except: return np.nan
            
        ptd_data['Util_Dec'] = ptd_data['Nível de Utilização [%]'].apply(limpar_util)
        ptd_data = ptd_data.dropna(subset=['Util_Dec'])

        # 4. Agrupamento (Garante que os nomes das colunas batem certo com o Excel)
        ip_df = ip_data.groupby(['CodDistritoConcelho', 'Concelho', 'Distrito']).agg(
            P_IP_Total=('Potência Instalada Total (W)', lambda x: x.sum()/1000),
            P_IP_Inef=('Potência Instalada Total (W)', lambda x: x[ip_data.loc[x.index, 'Tipo de Lâmpada'].isin(['Sódio', 'Mercúrio'])].sum()/1000)
        ).reset_index()

        ptd_df = ptd_data.groupby('CodDistritoConcelho').agg(
            Cap_PTD=('Potência instalada [kVA]', 'sum'),
            Util_Media=('Util_Dec', 'mean'),
            N_PTDs=('Código de Instalação', 'count')
        ).reset_index()

        # 5. Merge Final
        return pd.merge(ip_df, ptd_df, on='CodDistritoConcelho')

    except Exception as e:
        st.error(f"Erro ao carregar ficheiros Excel: {e}")
        return None

# Chamar a função
df_base = load_data()

df_base = load_data()

# --- SIDEBAR: CENÁRIOS INTERATIVOS ---
st.sidebar.header("⚙️ Configuração de Cenários")
st.sidebar.write("Ajuste os parâmetros para ver o impacto na rede.")

# Sliders baseados no enunciado
sim_led = st.sidebar.slider("Eficiência LED (%)", 0, 100, 65) / 100
sim_ve_perc = st.sidebar.slider("Penetração de Carregadores VE (%)", 0, 100, 60) / 100
sim_ve_pwr = st.sidebar.select_slider("Potência do Carregador (kW)", options=[3.7, 7.4, 11, 22], value=22)

# --- CÁLCULOS DINÂMICOS ---
df = df_base.copy()
df['Delta_P_LED'] = df['P_IP_Inef'] * sim_led
df['P_Folga'] = (df['Cap_PTD'] * 0.92) * (1 - df['Util_Media'])
df['P_VE'] = df['N_PTDs'] * sim_ve_pwr * sim_ve_perc
df['D'] = df['P_Folga'] + df['Delta_P_LED'] - df['P_VE']
df['Viavel'] = df['D'] > 0

# --- LAYOUT DO DASHBOARD ---
st.title("⚡ Análise de Viabilidade: Mobilidade Elétrica & Iluminação Pública")
st.markdown(f"**Cenário Atual:** Retrofit LED de {sim_led*100:.0f}% e instalação de carregadores de {sim_ve_pwr}kW em {sim_ve_perc*100:.0f}% dos PTDs.")

# Métricas principais
c1, c2, c3, c4 = st.columns(4)
c1.metric("Potência Libertada (Total)", f"{df['Delta_P_LED'].sum():.1f} kW")
c2.metric("Capacidade Livre Real", f"{df['P_Folga'].sum():.1f} kW")
c3.metric("Carga VE Estimada", f"{df['P_VE'].sum():.1f} kW")
viáveis = df['Viavel'].sum()
c4.metric("Concelhos Viáveis", f"{viáveis}/{len(df)}", delta=f"{viáveis - len(df)}")

st.divider()

# --- VISUALIZAÇÃO ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📍 Viabilidade por Concelho (Top 15 Críticos)")
    # Gráfico de barras dos menores saldos D
    fig_bar = px.bar(df.sort_values('D').head(15), x='D', y='Concelho', 
                     color='D', color_continuous_scale='RdYlGn',
                     orientation='h', title="Saldos de Viabilidade (D)")
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader("📊 Relação Folga vs Carga VE")
    # Scatter plot para ver a relação
    fig_scatter = px.scatter(df, x='P_Folga', y='P_VE', color='Viavel',
                             hover_name='Concelho', size='Cap_PTD',
                             title="Folga de Rede vs. Necessidade VE")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- TABELA DE DADOS ---
with st.expander("Ver dados detalhados por concelho"):
    st.dataframe(df.style.highlight_between(left=-100000, right=0, subset=['D'], color='#ff4b4b'))