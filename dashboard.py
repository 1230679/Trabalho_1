import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import re
from scipy import stats as scipy_stats
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Gestão Energética & Veículos Elétricos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1 { font-weight: 600; font-size: 1.6rem; letter-spacing: -0.02em; color: #0f1923;
         border-bottom: 3px solid #1a6cf0; padding-bottom: 0.5rem; margin-bottom: 1.5rem; }
    h2, h3 { font-weight: 500; letter-spacing: -0.01em; color: #0f1923; }
    [data-testid="metric-container"] { background: #f8f9fb; border: 1px solid #e2e6ea;
        border-left: 4px solid #1a6cf0; border-radius: 4px; padding: 1rem 1.2rem; }
    [data-testid="metric-container"] label { font-size: 0.72rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.08em; color: #6c757d; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; color: #0f1923; }
    [data-testid="stSidebar"] { background: #0f1923; }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
    [data-testid="stSidebar"] h2 { color: #ffffff !important; font-size: 0.85rem;
        text-transform: uppercase; letter-spacing: 0.1em;
        border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }
    hr { border: none; border-top: 1px solid #e2e6ea; margin: 1.5rem 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 2px solid #e2e6ea; }
    .stTabs [data-baseweb="tab"] { font-size: 0.82rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.06em; padding: 0.6rem 1.4rem;
        color: #6c757d; border-bottom: 2px solid transparent; margin-bottom: -2px; }
    .stTabs [aria-selected="true"] { color: #1a6cf0 !important;
        border-bottom: 2px solid #1a6cf0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CORES E TEMA
# ─────────────────────────────────────────────
COR_AZUL     = '#1a6cf0'
COR_VERDE    = '#00a86b'
COR_VERMELHO = '#e74c3c'
COR_CINZA    = '#34495e'
COR_AMARELO  = '#f1c40f'
COR_LARANJA  = '#FF6B35'

LAYOUT_BASE = dict(
    font_family   = 'IBM Plex Sans, sans-serif',
    paper_bgcolor = 'rgba(0,0,0,0)',
    plot_bgcolor  = 'rgba(0,0,0,0)',
    margin        = dict(l=10, r=10, t=40, b=10),
)

# Mapeamento CodDistrito → Nome (igual ao Jupyter)
MAPA_DISTRITOS = {
    1: 'Aveiro', 2: 'Beja', 3: 'Braga', 4: 'Bragança',
    5: 'Castelo Branco', 6: 'Coimbra', 7: 'Évora', 8: 'Faro',
    9: 'Guarda', 10: 'Leiria', 11: 'Lisboa', 12: 'Portalegre',
    13: 'Porto', 14: 'Santarém', 15: 'Setúbal', 16: 'Viana do Castelo',
    17: 'Vila Real', 18: 'Viseu', 20: 'Ilha da Madeira', 30: 'Açores'
}


# ─────────────────────────────────────────────
# ETL — idêntico ao Jupyter
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent
    try:
        df_ip = pd.read_excel(BASE_DIR / 'IP_data.xlsx',  na_values=['N/D', 'ND', '-'])
        df_pt = pd.read_excel(BASE_DIR / 'PTD_data.xlsx', na_values=['N/D', 'ND', '-'])

        # ── ETL IP (célula 3 do Jupyter) ─────────────────────────────────────
        df_ip['Is_Ineficiente'] = df_ip['Tipo de Lâmpada'].apply(
            lambda x: 1 if x in ['Sódio', 'Mercúrio'] else 0)
        df_ip['Potencia_kW'] = (
            pd.to_numeric(df_ip['Potência Instalada Total (W)'], errors='coerce').fillna(0) / 1000)

        df_ip_grouped = df_ip.groupby(['CodDistrito', 'Concelho', 'CodDistritoConcelho']).agg(
            P_IP_Total=('Potencia_kW', 'sum'),
            P_IP_Inef =('Potencia_kW', lambda x: x[df_ip.loc[x.index, 'Is_Ineficiente'] == 1].sum()),
        ).reset_index()

        # ── ETL PTD (célula 4 do Jupyter) ────────────────────────────────────
        def convert_utilizacao(valor):
            if isinstance(valor, str) and '%' in valor:
                nums = re.findall(r'\d+', valor)
                if nums:
                    return float(nums[-1]) / 100
            return np.nan

        df_pt['Util_Decimal'] = df_pt['Nível de Utilização [%]'].apply(convert_utilizacao)
        df_pt_clean = df_pt.dropna(subset=['Util_Decimal']).copy()

        # Coordenadas para o mapa (extração separada, não afeta o agrupamento)
        if 'Coordenadas Geográficas' in df_pt_clean.columns:
            coords = df_pt_clean['Coordenadas Geográficas'].str.split(',', expand=True)
            df_pt_clean['lat'] = pd.to_numeric(coords[0], errors='coerce')
            df_pt_clean['lon'] = pd.to_numeric(coords[1], errors='coerce')
        else:
            df_pt_clean['lat'], df_pt_clean['lon'] = np.nan, np.nan

        df_pt_grouped = df_pt_clean.groupby(['CodDistritoConcelho']).agg(
            Cap_PTD    = ('Potência instalada [kVA]', 'sum'),
            Util_Media = ('Util_Decimal', 'mean'),
            N_PTDs     = ('Concelho', 'count'),
            lat        = ('lat', 'mean'),
            lon        = ('lon', 'mean')
        ).reset_index()

        # ── Merge e variáveis derivadas (célula 5 do Jupyter) ────────────────
        df_final = pd.merge(df_ip_grouped, df_pt_grouped, on='CodDistritoConcelho', how='inner')

        df_final['Delta_P_LED']       = df_final['P_IP_Inef'] * 0.65
        df_final['P_Folga']           = (df_final['Cap_PTD'] * 0.92) * (1 - df_final['Util_Media'])
        df_final['P_VE']              = df_final['N_PTDs'] * 22 * 0.60
        df_final['D']                 = df_final['P_Folga'] + df_final['Delta_P_LED'] - df_final['P_VE']
        df_final['Rate_Ineficiencia'] = df_final['P_IP_Inef'] / df_final['P_IP_Total'].replace(0, np.nan)
        df_final['Nome_Distrito']     = df_final['CodDistrito'].map(MAPA_DISTRITOS)

        return df_final, df_pt_clean, df_ip

    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None, None


# ─────────────────────────────────────────────
# CENÁRIOS INTERATIVOS (sidebar)
# ─────────────────────────────────────────────
@st.cache_data
def compute_scenario(df_base, sim_led, sim_ve_perc, sim_ve_pwr):
    df = df_base.copy()
    df['Delta_P_LED'] = df['P_IP_Inef'] * sim_led
    df['P_IP_Antes']  = df['P_IP_Total']
    df['P_IP_Depois'] = df['P_IP_Total'] - df['Delta_P_LED']
    df['P_VE']        = df['N_PTDs'] * sim_ve_pwr * sim_ve_perc * 0.60
    df['Cap_PTD_kW']  = df['Cap_PTD'] * 0.92
    df['P_Ocupada']   = df['Cap_PTD_kW'] * df['Util_Media']
    df['P_Folga']     = df['Cap_PTD_kW'] * (1 - df['Util_Media'])
    df['D']           = df['P_Folga'] + df['Delta_P_LED'] - df['P_VE']
    df['Viavel']      = df['D'] > 0
    return df


# ─────────────────────────────────────────────
# INTERFACE
# ─────────────────────────────────────────────
df_base, df_pt_raw, df_ip_raw = load_data()

if df_base is not None:

    st.title("Dashboard de Eficiência Energética e Mobilidade Elétrica")

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    st.sidebar.markdown("## Parametrização de Cenários")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Iluminação Pública**")
    sim_led = st.sidebar.slider("Eficiência de substituição LED (%)", 0, 100, 65, 5) / 100
    st.sidebar.markdown("**Veículos Elétricos**")
    sim_ve_perc = st.sidebar.slider("Postos de carregamento por PTD (%)", 0, 100, 60, 5) / 100
    sim_ve_pwr  = st.sidebar.select_slider(
        "Potência do carregador (kW)", options=[3.7, 7.4, 11, 22], value=22)

    df = compute_scenario(df_base, sim_led, sim_ve_perc, sim_ve_pwr)

    # ── MÉTRICAS GLOBAIS ─────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Potência Libertada (LED)",   f"{df['Delta_P_LED'].sum():,.0f} kW")
    c2.metric("Folga Total na Rede",        f"{df['P_Folga'].sum():,.0f} kW")
    c3.metric("Carga Projetada (VE)",       f"{df['P_VE'].sum():,.0f} kW")
    c4.metric("Saldo Final de Viabilidade", f"{df['D'].sum():,.0f} kW")
    viab_pct = df['Viavel'].sum() / len(df) * 100
    c5.metric("Municípios Viáveis", f"{df['Viavel'].sum()} / {len(df)}", f"{viab_pct:.1f}%")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Perfis de Consumo",
        "Capacidade PTD & VE",
        "Análise Exploratória",
        "Mapa",
        "Dados"
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Perfis horários (antes vs. depois)
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Perfis Horários de Consumo — Iluminação Pública (Antes vs. Depois)")
        st.info(
            "**Nota metodológica:** Os ficheiros de dados originais não contêm séries temporais "
            "horárias. O perfil apresentado é uma **aproximação baseada no comportamento típico da "
            "iluminação pública** — ativa entre o pôr do sol (~18h) e o nascer do sol (~6h). "
            "A potência total antes e depois da modernização LED é calculada a partir dos dados reais."
        )

        horas  = list(range(24))
        perfil = [1 if (h >= 18 or h <= 6) else 0 for h in horas]
        p_antes  = df['P_IP_Antes'].sum()
        p_depois = df['P_IP_Depois'].sum()

        fig_perfil = go.Figure()
        fig_perfil.add_trace(go.Scatter(
            x=horas, y=[p_antes  * p for p in perfil], mode='lines',
            name='Antes da modernização', fill='tozeroy',
            line=dict(color=COR_LARANJA, width=2), fillcolor='rgba(255,107,53,0.15)'))
        fig_perfil.add_trace(go.Scatter(
            x=horas, y=[p_depois * p for p in perfil], mode='lines',
            name='Depois (LED)', fill='tozeroy',
            line=dict(color=COR_VERDE, width=2), fillcolor='rgba(0,168,107,0.15)'))
        fig_perfil.update_layout(
            **LAYOUT_BASE,
            xaxis=dict(title="Hora do dia", tickmode='linear', dtick=2, gridcolor='#e9ecef'),
            yaxis=dict(title="Carga IP agregada (kW)", gridcolor='#e9ecef'),
            legend=dict(orientation='h', y=1.05))
        st.plotly_chart(fig_perfil, use_container_width=True)

        ca, cb, cc = st.columns(3)
        ca.metric("Potência antes",  f"{p_antes:,.0f} kW")
        cb.metric("Potência depois", f"{p_depois:,.0f} kW")
        poup = ((p_antes - p_depois) / p_antes * 100) if p_antes > 0 else 0
        cc.metric("Redução", f"{p_antes - p_depois:,.0f} kW", f"-{poup:.1f}%")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Capacidade PTD & Cenários VE
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Capacidade Instalada e Disponível nos PTD")

        col1, col2 = st.columns(2)
        with col1:
            top10 = df.nlargest(10, 'Cap_PTD_kW').copy()
            fig_cap = go.Figure()
            fig_cap.add_trace(go.Bar(x=top10['Concelho'], y=top10['P_Ocupada'],
                                     name='Carga Actual', marker_color=COR_CINZA))
            fig_cap.add_trace(go.Bar(x=top10['Concelho'], y=top10['P_Folga'],
                                     name='Capacidade Disponível', marker_color=COR_VERDE))
            fig_cap.update_layout(**LAYOUT_BASE, barmode='stack',
                title='Top 10 Municípios — Capacidade PTD (kW)',
                xaxis=dict(title='Município', gridcolor='#e9ecef'),
                yaxis=dict(title='Potência (kW)', gridcolor='#e9ecef'),
                legend=dict(orientation='h', y=1.05))
            st.plotly_chart(fig_cap, use_container_width=True)

        with col2:
            fig_hist = px.histogram(df, x='Util_Media', nbins=20,
                title='Distribuição do Nível de Utilização Médio dos PTD',
                labels={'Util_Media': 'Nível de utilização'},
                color_discrete_sequence=[COR_AZUL])
            fig_hist.update_layout(**LAYOUT_BASE,
                xaxis=dict(tickformat='.0%', gridcolor='#e9ecef'),
                yaxis=dict(gridcolor='#e9ecef'))
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")
        st.subheader("Estimativa de Potência Libertada pelas Medidas de Eficiência")
        top_led = df.nlargest(10, 'Delta_P_LED').sort_values('Delta_P_LED', ascending=True)
        fig_led = px.bar(top_led, x='Delta_P_LED', y='Concelho', orientation='h',
            title='Top 10 Municípios — Potencial de Poupança LED (kW)',
            labels={'Delta_P_LED': 'Potência libertada (kW)', 'Concelho': ''},
            color_discrete_sequence=[COR_AMARELO])
        fig_led.update_layout(**LAYOUT_BASE,
            xaxis=dict(gridcolor='#e9ecef'), yaxis=dict(gridcolor='#e9ecef'))
        st.plotly_chart(fig_led, use_container_width=True)

        st.markdown("---")
        st.subheader("Cenários de Integração de Carregadores VE e Impacto na Carga do PTD")
        st.caption(f"Carregador: **{sim_ve_pwr} kW** | Cobertura: **{sim_ve_perc*100:.0f}%** | "
                   f"Fator de simultaneidade: 60%")

        col3, col4 = st.columns(2)
        with col3:
            top10_ve = df.nlargest(10, 'Cap_PTD_kW').copy()
            fig_ve = go.Figure()
            fig_ve.add_trace(go.Bar(x=top10_ve['Concelho'], y=top10_ve['P_Ocupada'],
                                    name='Carga Actual', marker_color=COR_CINZA))
            fig_ve.add_trace(go.Bar(x=top10_ve['Concelho'], y=top10_ve['P_VE'],
                                    name='Impacto VE', marker_color=COR_VERMELHO))
            fig_ve.add_trace(go.Bar(
                x=top10_ve['Concelho'],
                y=(top10_ve['P_Folga'] + top10_ve['Delta_P_LED'] - top10_ve['P_VE']).clip(lower=0),
                name='Margem Restante', marker_color=COR_VERDE))
            fig_ve.update_layout(**LAYOUT_BASE, barmode='stack',
                title='Impacto da Carga VE no PTD (Top 10)',
                xaxis=dict(title='Município', gridcolor='#e9ecef'),
                yaxis=dict(title='Potência (kW)', gridcolor='#e9ecef'),
                legend=dict(orientation='h', y=1.05))
            st.plotly_chart(fig_ve, use_container_width=True)

        with col4:
            viab_counts = (df['Viavel'].map({True: 'Suporta VE', False: 'Requer Expansão'})
                           .value_counts().reset_index())
            viab_counts.columns = ['Status', 'Contagem']
            fig_pie = px.pie(viab_counts, names='Status', values='Contagem',
                title='Proporção de Viabilidade dos Municípios', hole=0.45,
                color='Status',
                color_discrete_map={'Suporta VE': COR_VERDE, 'Requer Expansão': COR_VERMELHO})
            fig_pie.update_layout(**LAYOUT_BASE)
            st.plotly_chart(fig_pie, use_container_width=True)

        df_saldo    = df[['Concelho', 'D']].dropna().sort_values('D', ascending=False)
        df_extremos = pd.concat([df_saldo.head(5), df_saldo.tail(5)])
        cores = [COR_VERDE if v >= 0 else COR_VERMELHO for v in df_extremos['D']]
        fig_saldo = go.Figure(go.Bar(x=df_extremos['Concelho'], y=df_extremos['D'],
                                     marker_color=cores))
        fig_saldo.update_layout(**LAYOUT_BASE,
            title='Saldo de Viabilidade (D) — Top e Bottom 5',
            xaxis=dict(title='Município', gridcolor='#e9ecef'),
            yaxis=dict(title='D (kW)', gridcolor='#e9ecef'),
            shapes=[dict(type='line', x0=-0.5, x1=len(df_extremos)-0.5, y0=0, y1=0,
                         line=dict(color='#6c757d', width=1.5, dash='dash'))])
        st.plotly_chart(fig_saldo, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Análise Exploratória (4.3 do enunciado, usa df_pt_raw como Jupyter)
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Análise Exploratória de Dados")

        # 4.3.1 Mix Tecnológico
        st.markdown("#### 4.3.1 · Mix Tecnológico da Iluminação Pública")
        col_a, col_b = st.columns(2)
        with col_a:
            if df_ip_raw is not None and 'Tipo de Lâmpada' in df_ip_raw.columns:
                df_ip_plot = df_ip_raw.copy()
                df_ip_plot['Categoria_Grafico'] = df_ip_plot['Tipo de Lâmpada'].apply(
                    lambda x: x if x in ['Sódio', 'Mercúrio', 'LED'] else 'Outros')
                df_ip_plot['Potencia_kW'] = (
                    pd.to_numeric(df_ip_plot['Potência Instalada Total (W)'], errors='coerce').fillna(0) / 1000)
                mix_data = df_ip_plot.groupby('Categoria_Grafico')['Potencia_kW'].sum().reset_index()
                ordem_manual = {'Sódio': 0, 'Mercúrio': 1, 'LED': 2, 'Outros': 3}
                mix_data['Ordem'] = mix_data['Categoria_Grafico'].map(ordem_manual)
                mix_data = mix_data.sort_values('Ordem')
                colors_map = {'Sódio': '#ff9999', 'Mercúrio': '#ffcc99', 'LED': '#66b3ff', 'Outros': '#99ff99'}
                fig_pie_tec = go.Figure(go.Pie(
                    labels=mix_data['Categoria_Grafico'],
                    values=mix_data['Potencia_kW'],
                    hole=0.0,
                    marker_colors=[colors_map.get(lbl) for lbl in mix_data['Categoria_Grafico']],
                    textinfo='percent+label',
                    textfont_size=12,
                    pull=[0.02] * len(mix_data)
                ))
                fig_pie_tec.update_layout(**LAYOUT_BASE,
                    title='Mix Tecnológico da Iluminação Pública', showlegend=True,
                    legend=dict(orientation='h', y=-0.1))
                st.plotly_chart(fig_pie_tec, use_container_width=True)
            else:
                st.warning('Dados de iluminação pública não disponíveis para o mix tecnológico.')

        with col_b:
            top10_inef = df.nlargest(10, 'P_IP_Inef').sort_values('P_IP_Inef', ascending=True)
            fig_bar_inef = px.bar(top10_inef, x='P_IP_Inef', y='Concelho', orientation='h',
                title='Top 10 Municípios por Potência Ineficiente (kW)',
                labels={'P_IP_Inef': 'Potência Ineficiente (kW)', 'Concelho': ''},
                color='P_IP_Inef', color_continuous_scale='Reds_r')
            fig_bar_inef.update_layout(**LAYOUT_BASE, coloraxis_showscale=False,
                xaxis=dict(gridcolor='#e9ecef'), yaxis=dict(gridcolor='#e9ecef'))
            st.plotly_chart(fig_bar_inef, use_container_width=True)

        st.markdown("---")

        # 4.3.2 Boxplots por Distrito — usa df_pt_raw (PTDs individuais), como no Jupyter
        st.markdown("#### 4.3.2 · Distribuição do Nível de Utilização dos PTDs por Distrito")
        st.caption("Aveiro (cod. 1), Lisboa (cod. 11), Porto (cod. 13), Setúbal (cod. 15)")

        col_c, col_d = st.columns(2)
        with col_c:
            if df_base is not None and 'CodDistrito' in df_base.columns and 'Util_Media' in df_base.columns:
                df_box = df_base[df_base['CodDistrito'].isin([1, 11, 13, 15])].copy()
                df_box['Distrito_Nome'] = df_box['CodDistrito'].map({1: 'Aveiro', 11: 'Lisboa', 13: 'Porto', 15: 'Setúbal'})

                fig_box, ax_box = plt.subplots(figsize=(8, 5))
                sns.boxplot(x='Distrito_Nome', y='Util_Media', data=df_box,
                            order=['Aveiro', 'Lisboa', 'Porto', 'Setúbal'], ax=ax_box)
                ax_box.set_title('Distribuição do Nível de Utilização dos PTDs por Distrito')
                ax_box.set_ylabel('Utilização Média (Decimal)')
                ax_box.set_xlabel('')
                plt.tight_layout()
                st.pyplot(fig_box)
                plt.close(fig_box)

                variab = df_box.groupby('Distrito_Nome')['Util_Media'].std().sort_values(ascending=False)
                st.info(f"Distrito com maior variabilidade: "
                        f"**{variab.index[0]}** (σ = {variab.iloc[0]:.4f})")
            else:
                st.warning("Dados consolidados de PTD não disponíveis.")

        # 4.3.3 Outliers — usa df_pt_raw, como no Jupyter
        with col_d:
            st.markdown("##### 4.3.3 · Outliers nos Níveis de Ocupação da Rede")
            util_series = (df_pt_raw['Util_Decimal'].dropna()
                           if df_pt_raw is not None and 'Util_Decimal' in df_pt_raw.columns
                           else df['Util_Media'].dropna())

            fig_out, ax_out = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=util_series, color='lightgreen', fliersize=6, linewidth=1.5, ax=ax_out)
            ax_out.set_title('Identificação de Outliers nos Níveis de Ocupação (PTDs)')
            ax_out.set_xlabel('Nível de Utilização (Decimal)')
            ax_out.axvline(x=1.0, color='red', linestyle='--', label='Capacidade Crítica (100%)')
            ax_out.legend()
            plt.tight_layout()
            st.pyplot(fig_out)
            plt.close(fig_out)

            q1 = util_series.quantile(0.25); q3 = util_series.quantile(0.75)
            iqr = q3 - q1
            n_out = ((util_series < q1-1.5*iqr) | (util_series > q3+1.5*iqr)).sum()
            n_crit = (util_series >= 1.0).sum()
            st.info(f"Outliers detectados: {n_out} PTDs  |  "
                    f"PTDs a ≥ 100% capacidade: {n_crit}")

        st.markdown("---")

        # 4.3.4 Estatísticas descritivas — usa df_pt_raw como no Jupyter
        st.markdown("#### 4.3.4 · Estatísticas Descritivas por Concelho")
        st.caption("Calculadas sobre os PTDs individuais de cada concelho (igual ao Jupyter)")
        concelhos_foco = ['Coimbra', 'Évora', 'Braga', 'Faro']
        if (df_pt_raw is not None and 'Util_Decimal' in df_pt_raw.columns
                and 'Concelho' in df_pt_raw.columns):
            df_pt_foco = df_pt_raw[df_pt_raw['Concelho'].isin(concelhos_foco)]
            rows = []
            for c in concelhos_foco:
                vals = df_pt_foco[df_pt_foco['Concelho'] == c]['Util_Decimal'].dropna()
                if len(vals) > 0:
                    rows.append({
                        'Concelho':      c,
                        'Média':         round(vals.mean(), 4),
                        'Desvio Padrão': round(vals.std(), 4),
                        'Q1':            round(vals.quantile(0.25), 4),
                        'Mediana':       round(vals.median(), 4),
                        'Q3':            round(vals.quantile(0.75), 4),
                        'Assimetria':    round(float(skew(vals)), 4),
                        'Curtose':       round(float(kurtosis(vals)), 4),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index('Concelho'), use_container_width=True)
        else:
            st.info("Dados individuais de PTD não disponíveis.")

        st.markdown("---")

        # 4.4.4 Correlação de Pearson
        st.markdown("#### 4.4.4 · Correlação de Pearson — Capacidade PTD vs Iluminação Pública")
        st.caption("Relação linear entre capacidade de transformação instalada e carga de IP (todos os concelhos)")
        df_corr_plot = df[['Cap_PTD', 'P_IP_Total']].dropna()
        if len(df_corr_plot) > 2:
            r_val, p_val_p = scipy_stats.pearsonr(df_corr_plot['Cap_PTD'], df_corr_plot['P_IP_Total'])
            fig_p = px.scatter(df_corr_plot, x='Cap_PTD', y='P_IP_Total',
                trendline='ols', opacity=0.6,
                title='Relação entre Capacidade de Transformação e Iluminação Pública',
                labels={'Cap_PTD': 'Capacidade Total de Transformação (kVA)',
                        'P_IP_Total': 'Carga Total de IP (kW)'},
                color_discrete_sequence=[COR_AZUL])
            fig_p.update_traces(selector=dict(mode='lines'),
                                line=dict(color=COR_VERMELHO, width=2))
            fig_p.update_layout(**LAYOUT_BASE,
                xaxis=dict(gridcolor='#e9ecef'), yaxis=dict(gridcolor='#e9ecef'))
            st.plotly_chart(fig_p, use_container_width=True)

            cr1, cr2, cr3 = st.columns(3)
            cr1.metric("Coeficiente de Pearson (r)", f"{r_val:.4f}")
            cr2.metric("p-value", f"{p_val_p:.2e}")
            conclusao = ("Rejeitamos H₀ — relação linear estatisticamente significativa."
                         if p_val_p < 0.05
                         else "Não rejeitamos H₀ — relação não significativa.")
            cr3.markdown(f"<br>{conclusao}", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — Mapa (PTDs + IP + locais potenciais)
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Mapa das Zonas Analisadas")
        st.caption(
            "Azul: carga de iluminação pública do município. "
            "Verde: município viável para integração de VE. "
            "Vermelho: município que requer expansão da rede. "
            "Estrela: Top 15 locais potenciais para carregamento. "
            "Tamanho proporcional à potência IP / número de PTDs."
        )

        cols_mapa = [c for c in ['lat', 'lon', 'Concelho', 'Viavel', 'D',
                                  'N_PTDs', 'P_Folga', 'P_IP_Total', 'P_IP_Inef', 'Cap_PTD_kW']
                     if c in df.columns]
        df_map = df[cols_mapa].dropna(subset=['lat', 'lon'])

        if not df_map.empty:
            fig_map = go.Figure()

            # Camada 1 — Iluminação Pública
            p_max = df_map['P_IP_Total'].quantile(0.95)
            fig_map.add_trace(go.Scattermapbox(
                lat=df_map['lat'], lon=df_map['lon'], mode='markers',
                marker=dict(
                    size=(df_map['P_IP_Total'].clip(upper=p_max) / p_max * 12 + 4).clip(4, 16),
                    color=COR_AZUL, opacity=0.45),
                text=df_map['Concelho'],
                hovertemplate="<b>%{text}</b><br>IP Total: %{customdata[0]:.1f} kW<br>"
                              "IP Ineficiente: %{customdata[1]:.1f} kW<extra>Iluminação Pública</extra>",
                customdata=df_map[['P_IP_Total', 'P_IP_Inef']].values,
                name='Iluminação Pública'))

            # Camada 2 — Inviáveis
            df_inv = df_map[~df_map['Viavel']]
            if not df_inv.empty:
                fig_map.add_trace(go.Scattermapbox(
                    lat=df_inv['lat'], lon=df_inv['lon'], mode='markers',
                    marker=dict(size=df_inv['N_PTDs'].clip(upper=50)/2+5,
                                color=COR_VERMELHO, opacity=0.7),
                    text=df_inv['Concelho'],
                    hovertemplate="<b>%{text}</b><br>PTDs: %{customdata[0]}<br>"
                                  "Folga: %{customdata[1]:.0f} kW<br>Saldo D: %{customdata[2]:.0f} kW"
                                  "<extra>Requer expansão</extra>",
                    customdata=df_inv[['N_PTDs', 'P_Folga', 'D']].values,
                    name='Requer expansão da rede'))

            # Camada 3 — Viáveis
            df_via = df_map[df_map['Viavel']]
            if not df_via.empty:
                fig_map.add_trace(go.Scattermapbox(
                    lat=df_via['lat'], lon=df_via['lon'], mode='markers',
                    marker=dict(size=df_via['N_PTDs'].clip(upper=50)/2+5,
                                color=COR_VERDE, opacity=0.8),
                    text=df_via['Concelho'],
                    hovertemplate="<b>%{text}</b><br>PTDs: %{customdata[0]}<br>"
                                  "Folga: %{customdata[1]:.0f} kW<br>Saldo D: %{customdata[2]:.0f} kW"
                                  "<extra>Viável para VE</extra>",
                    customdata=df_via[['N_PTDs', 'P_Folga', 'D']].values,
                    name='Suporta integração VE'))

            # Camada 4 — Top 15 potenciais
            df_pot = df_map[df_map['Viavel']].nlargest(15, 'D')
            if not df_pot.empty:
                fig_map.add_trace(go.Scattermapbox(
                    lat=df_pot['lat'], lon=df_pot['lon'], mode='markers',
                    marker=dict(size=16, color=COR_AMARELO, symbol='star', opacity=0.95),
                    text=df_pot['Concelho'],
                    hovertemplate="<b>%{text}</b><br>Saldo D: %{customdata[0]:.0f} kW"
                                  "<extra>Local potencial para carregamento</extra>",
                    customdata=df_pot[['D']].values,
                    name='Local potencial (Top 15)'))

            fig_map.update_layout(
                mapbox=dict(style='carto-positron', zoom=5.5, center=dict(lat=39.5, lon=-8.0)),
                margin=dict(r=0, t=0, l=0, b=0),
                legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#e2e6ea',
                            borderwidth=1, x=0.01, y=0.99, font=dict(size=11)),
                height=580)
            st.plotly_chart(fig_map, use_container_width=True)

            if not df_pot.empty:
                st.markdown("**Top 15 municípios com maior potencial para carregadores VE**")
                cols_tab = [c for c in ['Concelho', 'N_PTDs', 'P_Folga', 'Delta_P_LED', 'P_VE', 'D']
                            if c in df_pot.columns]
                st.dataframe(df_pot[cols_tab].sort_values('D', ascending=False).round(1),
                             use_container_width=True, hide_index=True)
        else:
            st.warning("Coordenadas geográficas não disponíveis no ficheiro PTD_data.xlsx.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — Dados tabelares
    # ════════════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("Dados Tabelares Completos")
        cols_show = [c for c in [
            'CodDistritoConcelho', 'CodDistrito', 'Nome_Distrito', 'Concelho',
            'P_IP_Total', 'P_IP_Inef', 'Rate_Ineficiencia',
            'N_PTDs', 'Cap_PTD', 'Util_Media',
            'Delta_P_LED', 'P_Folga', 'P_VE', 'D', 'Viavel'
        ] if c in df.columns]

        filtro = st.radio("Filtrar por viabilidade",
                          ["Todos", "Viáveis", "Requerem expansão"], horizontal=True)
        df_tab = df.copy()
        if filtro == "Viáveis":
            df_tab = df_tab[df_tab['Viavel'] == True]
        elif filtro == "Requerem expansão":
            df_tab = df_tab[df_tab['Viavel'] == False]

        st.dataframe(df_tab[cols_show].round(4), use_container_width=True, hide_index=True)
        st.caption(f"{len(df_tab)} registos mostrados de {len(df)} no total.")

else:
    st.error("Não foi possível carregar os dados. "
             "Verifique se IP_data.xlsx e PTD_data.xlsx estão na mesma pasta que este script.")