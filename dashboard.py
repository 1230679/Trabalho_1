import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import re

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Gestão Energética & Veículos Elétricos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZADO ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    h1 {
        font-weight: 600;
        font-size: 1.6rem;
        letter-spacing: -0.02em;
        color: #0f1923;
        border-bottom: 3px solid #1a6cf0;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }

    h2, h3 {
        font-weight: 500;
        letter-spacing: -0.01em;
        color: #0f1923;
    }

    [data-testid="metric-container"] {
        background: #f8f9fb;
        border: 1px solid #e2e6ea;
        border-left: 4px solid #1a6cf0;
        border-radius: 4px;
        padding: 1rem 1.2rem;
    }

    [data-testid="metric-container"] label {
        font-size: 0.72rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6c757d;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        color: #0f1923;
    }

    [data-testid="stSidebar"] {
        background: #0f1923;
    }

    [data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }

    [data-testid="stSidebar"] h2 {
        color: #ffffff !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border-bottom: 1px solid #30363d;
        padding-bottom: 0.5rem;
    }

    hr {
        border: none;
        border-top: 1px solid #e2e6ea;
        margin: 1.5rem 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #e2e6ea;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 0.6rem 1.4rem;
        color: #6c757d;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
    }

    .stTabs [aria-selected="true"] {
        color: #1a6cf0 !important;
        border-bottom: 2px solid #1a6cf0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CARREGAMENTO E PROCESSAMENTO DE DADOS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent
    try:
        ip_path  = BASE_DIR / 'IP_data.xlsx'
        ptd_path = BASE_DIR / 'PTD_data.xlsx'

        df_ip = pd.read_excel(ip_path,  na_values=['N/D', 'ND', '-'])
        df_pt = pd.read_excel(ptd_path, na_values=['N/D', 'ND', '-'])

        # ── ETL Iluminação Pública ──────────────────────────────────────────
        df_ip['Is_Ineficiente'] = df_ip['Tipo de Lâmpada'].apply(
            lambda x: 1 if x in ['Sódio', 'Mercúrio'] else 0
        )
        df_ip['Potencia_kW'] = (
            pd.to_numeric(df_ip['Potência Instalada Total (W)'], errors='coerce').fillna(0) / 1000
        )

        # Guardar todas as colunas geográficas disponíveis para o groupby
        colunas_geo_ip = [
            c for c in ['Distrito', 'CodDistrito', 'Concelho', 'CodDistritoConcelho']
            if c in df_ip.columns
        ]

        df_ip_grouped = df_ip.groupby(colunas_geo_ip).agg(
            P_IP_Total    = ('Potencia_kW', 'sum'),
            P_IP_Inef     = ('Potencia_kW',
                             lambda x: x[df_ip.loc[x.index, 'Is_Ineficiente'] == 1].sum()),
            N_Lampadas_Total = ('Tipo de Lâmpada', 'count'),
            N_Lampadas_Inef  = ('Tipo de Lâmpada',
                                lambda x: x.isin(['Sódio', 'Mercúrio']).sum()),
            N_Lampadas_LED   = ('Tipo de Lâmpada',
                                lambda x: x.str.upper().str.contains('LED', na=False).sum()),
        ).reset_index()

        # ── ETL Postos de Transformação ─────────────────────────────────────
        def convert_utilizacao(valor):
            if isinstance(valor, str):
                v = valor.replace('%', '').strip()
                if '-' in v:
                    return float(v.split('-')[-1]) / 100
                if '+' in v:
                    return float(v.replace('+', '')) / 100
                nums = re.findall(r'\d+', v)
                if nums:
                    return float(nums[-1]) / 100
            elif isinstance(valor, (int, float)):
                return float(valor)
            return np.nan

        df_pt['Util_Decimal'] = df_pt['Nível de Utilização [%]'].apply(convert_utilizacao)
        df_pt = df_pt.dropna(subset=['Util_Decimal'])

        if 'Coordenadas Geográficas' in df_pt.columns:
            coords = df_pt['Coordenadas Geográficas'].str.split(',', expand=True)
            df_pt['lat'] = pd.to_numeric(coords[0], errors='coerce')
            df_pt['lon'] = pd.to_numeric(coords[1], errors='coerce')
        else:
            df_pt['lat'], df_pt['lon'] = np.nan, np.nan

        colunas_geo_ptd = [
            c for c in ['CodDistritoConcelho', 'Concelho'] if c in df_pt.columns
        ]

        # Escolher a chave de merge (preferir CodDistritoConcelho, fallback Concelho)
        chave_merge = (
            'CodDistritoConcelho'
            if 'CodDistritoConcelho' in df_ip.columns and 'CodDistritoConcelho' in df_pt.columns
            else 'Concelho'
        )

        df_pt_grouped = df_pt.groupby(colunas_geo_ptd).agg(
            Cap_PTD  = ('Potência instalada [kVA]', 'sum'),
            Util_Media = ('Util_Decimal', 'mean'),
            N_PTDs   = ('Código de Instalação', 'count'),
            lat      = ('lat', 'mean'),
            lon      = ('lon', 'mean')
        ).reset_index()

        df_final = pd.merge(df_ip_grouped, df_pt_grouped, on=chave_merge, how='inner')

        # ── FIX: garantir que a coluna 'Concelho' existe após o merge ───────
        # Quando o merge é por CodDistritoConcelho, pode haver duas colunas
        # 'Concelho_x' e 'Concelho_y'. Consolidamo-las numa única 'Concelho'.
        if 'Concelho' not in df_final.columns:
            if 'Concelho_x' in df_final.columns:
                df_final['Concelho'] = df_final['Concelho_x']
            elif 'Concelho_y' in df_final.columns:
                df_final['Concelho'] = df_final['Concelho_y']
            else:
                # Último recurso: usar o código como nome
                df_final['Concelho'] = df_final[chave_merge].astype(str)

        # Limpar colunas duplicadas se existirem
        for sufixo in ['_x', '_y']:
            col = f'Concelho{sufixo}'
            if col in df_final.columns:
                df_final.drop(columns=[col], inplace=True)

        return df_final, df_pt, df_ip

    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None, None


# ─────────────────────────────────────────────
# CÁLCULO DE CENÁRIOS
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
    df['Taxa_Inef']   = (df['N_Lampadas_Inef'] / df['N_Lampadas_Total']) * 100
    df['Rate_Inef']   = df['P_IP_Inef'] / df['P_IP_Total'].replace(0, np.nan)

    return df


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
    font_family  = 'IBM Plex Sans, sans-serif',
    paper_bgcolor = 'rgba(0,0,0,0)',
    plot_bgcolor  = 'rgba(0,0,0,0)',
    margin        = dict(l=10, r=10, t=40, b=10),
)


# ─────────────────────────────────────────────
# HELPER: nome do município (seguro)
# ─────────────────────────────────────────────
def get_concelho_col(df):
    """Devolve 'Concelho' se existir, caso contrário usa o índice como string."""
    return 'Concelho' if 'Concelho' in df.columns else None


# ─────────────────────────────────────────────
# INTERFACE PRINCIPAL
# ─────────────────────────────────────────────
df_base, df_pt_raw, df_ip_raw = load_data()

if df_base is not None:

    st.title("Dashboard de Eficiência Energética e Mobilidade Elétrica")

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    st.sidebar.markdown("## Parametrização de Cenários")
    st.sidebar.markdown("---")

    st.sidebar.markdown("**Iluminação Pública**")
    sim_led = st.sidebar.slider(
        "Eficiência de substituição LED (%)", 0, 100, 65, 5
    ) / 100

    st.sidebar.markdown("**Veículos Elétricos**")
    sim_ve_perc = st.sidebar.slider(
        "Postos de carregamento por PTD (%)", 0, 100, 60, 5
    ) / 100
    sim_ve_pwr = st.sidebar.select_slider(
        "Potência do carregador (kW)", options=[3.7, 7.4, 11, 22], value=22
    )

    df = compute_scenario(df_base, sim_led, sim_ve_perc, sim_ve_pwr)

    # ── MÉTRICAS GLOBAIS ──────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Potência Libertada (LED)",   f"{df['Delta_P_LED'].sum():,.0f} kW")
    col2.metric("Folga Total na Rede",        f"{df['P_Folga'].sum():,.0f} kW")
    col3.metric("Carga Projetada (VE)",       f"{df['P_VE'].sum():,.0f} kW")
    col4.metric("Saldo Final de Viabilidade", f"{df['D'].sum():,.0f} kW")
    viab_pct = (df['Viavel'].sum() / len(df)) * 100 if len(df) > 0 else 0
    col5.metric(
        "Municípios Viáveis",
        f"{df['Viavel'].sum()} / {len(df)}",
        f"{viab_pct:.1f}%"
    )
    st.markdown("---")

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Perfis de Consumo",
        "Capacidade PTD & VE",
        "Analise Exploratoria",
        "Mapa",
        "Dados"
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Perfis Horários de Consumo IP
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Perfis Horários de Consumo — Iluminação Pública")
        st.caption(
            "Perfil simplificado: iluminação activa entre as 18h e as 6h. "
            "Comparação entre o estado actual e após substituição por LED."
        )

        horas  = list(range(24))
        perfil = [1 if (h >= 18 or h <= 6) else 0 for h in horas]

        p_antes  = df['P_IP_Antes'].sum()
        p_depois = df['P_IP_Depois'].sum()

        fig_perfil = go.Figure()
        fig_perfil.add_trace(go.Scatter(
            x=horas, y=[p_antes  * p for p in perfil],
            mode='lines', name='Antes da modernização',
            fill='tozeroy',
            line=dict(color=COR_LARANJA, width=2),
            fillcolor='rgba(255,107,53,0.15)'
        ))
        fig_perfil.add_trace(go.Scatter(
            x=horas, y=[p_depois * p for p in perfil],
            mode='lines', name='Depois (LED)',
            fill='tozeroy',
            line=dict(color=COR_VERDE, width=2),
            fillcolor='rgba(0,168,107,0.15)'
        ))
        fig_perfil.update_layout(
            **LAYOUT_BASE,
            xaxis=dict(title="Hora do dia", tickmode='linear', dtick=2, gridcolor='#e9ecef'),
            yaxis=dict(title="Carga IP agregada (kW)", gridcolor='#e9ecef'),
            legend=dict(orientation='h', y=1.05),
        )
        st.plotly_chart(fig_perfil, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Potência antes",  f"{p_antes:,.0f} kW")
        c2.metric("Potência depois", f"{p_depois:,.0f} kW")
        poup_pct = ((p_antes - p_depois) / p_antes * 100) if p_antes > 0 else 0
        c3.metric("Reducao", f"{p_antes - p_depois:,.0f} kW", f"-{poup_pct:.1f}%")

        st.markdown("---")
        st.subheader("Mix Tecnológico da Iluminação Pública")

        col_mix1, col_mix2 = st.columns(2)
        with col_mix1:
            total_led  = df['P_IP_Total'].sum() - df['P_IP_Inef'].sum()
            total_inef = df['P_IP_Inef'].sum()
            fig_pie_mix = go.Figure(go.Pie(
                labels=['LED / Eficiente', 'Sódio / Mercúrio (Ineficiente)'],
                values=[total_led, total_inef],
                hole=0.45,
                marker_colors=[COR_VERDE, COR_LARANJA],
                textinfo='percent+label',
                textfont_size=12
            ))
            fig_pie_mix.update_layout(
                **LAYOUT_BASE,
                title='Potência por Tecnologia (kW)',
                showlegend=False
            )
            st.plotly_chart(fig_pie_mix, use_container_width=True)

        with col_mix2:
            top_inef  = df.nlargest(10, 'P_IP_Inef').sort_values('P_IP_Inef', ascending=True)
            nome_col  = get_concelho_col(top_inef)
            y_axis    = nome_col if nome_col else top_inef.index
            fig_inef  = px.bar(
                top_inef, x='P_IP_Inef', y=y_axis,
                orientation='h',
                title='Top 10 Municípios — Potência Ineficiente (kW)',
                labels={'P_IP_Inef': 'Potência ineficiente (kW)'},
                color_discrete_sequence=[COR_LARANJA]
            )
            fig_inef.update_layout(**LAYOUT_BASE,
                                   yaxis=dict(gridcolor='#e9ecef'),
                                   xaxis=dict(gridcolor='#e9ecef'))
            st.plotly_chart(fig_inef, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Capacidade PTD & Integração VE
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Capacidade Instalada e Disponível nos PTD")

        col_ptd1, col_ptd2 = st.columns(2)

        with col_ptd1:
            top10 = df.nlargest(10, 'Cap_PTD_kW').copy()
            if 'Concelho' not in top10.columns:
                top10['Concelho'] = top10.index.astype(str)
            fig_cap = go.Figure()
            fig_cap.add_trace(go.Bar(
                x=top10['Concelho'], y=top10['P_Ocupada'],
                name='Carga Actual', marker_color=COR_CINZA
            ))
            fig_cap.add_trace(go.Bar(
                x=top10['Concelho'], y=top10['P_Folga'],
                name='Capacidade Disponível', marker_color=COR_VERDE
            ))
            fig_cap.update_layout(
                **LAYOUT_BASE,
                barmode='stack',
                title='Top 10 Municípios — Capacidade PTD (kW)',
                xaxis=dict(title='Município', gridcolor='#e9ecef'),
                yaxis=dict(title='Potência (kW)', gridcolor='#e9ecef'),
                legend=dict(orientation='h', y=1.05)
            )
            st.plotly_chart(fig_cap, use_container_width=True)

        with col_ptd2:
            fig_hist = px.histogram(
                df, x='Util_Media', nbins=20,
                title='Distribuição do Nível de Utilização Médio dos PTD',
                labels={'Util_Media': 'Nível de utilização', 'count': 'N.º municípios'},
                color_discrete_sequence=[COR_AZUL]
            )
            fig_hist.update_layout(
                **LAYOUT_BASE,
                xaxis=dict(tickformat='.0%', gridcolor='#e9ecef'),
                yaxis=dict(gridcolor='#e9ecef')
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")
        st.subheader("Cenários de Integração de Carregadores VE e Impacto na Rede")
        st.caption(
            f"Carregador seleccionado: **{sim_ve_pwr} kW** | "
            f"Cobertura por PTD: **{sim_ve_perc*100:.0f}%** | "
            f"Factor de simultaneidade: 60%"
        )

        col_ve1, col_ve2 = st.columns(2)

        with col_ve1:
            top10_ve = df.nlargest(10, 'Cap_PTD_kW').copy()
            if 'Concelho' not in top10_ve.columns:
                top10_ve['Concelho'] = top10_ve.index.astype(str)
            fig_ve = go.Figure()
            fig_ve.add_trace(go.Bar(
                x=top10_ve['Concelho'], y=top10_ve['P_Ocupada'],
                name='Carga Actual', marker_color=COR_CINZA
            ))
            fig_ve.add_trace(go.Bar(
                x=top10_ve['Concelho'], y=top10_ve['P_VE'],
                name='Impacto VE', marker_color=COR_VERMELHO
            ))
            fig_ve.add_trace(go.Bar(
                x=top10_ve['Concelho'],
                y=(top10_ve['P_Folga'] + top10_ve['Delta_P_LED'] - top10_ve['P_VE']).clip(lower=0),
                name='Margem Restante', marker_color=COR_VERDE
            ))
            fig_ve.update_layout(
                **LAYOUT_BASE,
                barmode='stack',
                title='Impacto da Carga VE no PTD (Top 10)',
                xaxis=dict(title='Município', gridcolor='#e9ecef'),
                yaxis=dict(title='Potência (kW)', gridcolor='#e9ecef'),
                legend=dict(orientation='h', y=1.05)
            )
            st.plotly_chart(fig_ve, use_container_width=True)

        with col_ve2:
            top_led  = df.nlargest(10, 'Delta_P_LED').sort_values('Delta_P_LED', ascending=True)
            nome_col = get_concelho_col(top_led)
            fig_led  = px.bar(
                top_led,
                x='Delta_P_LED',
                y=nome_col if nome_col else top_led.index,
                orientation='h',
                title='Top 10 Municípios — Potencial de Poupança LED (kW)',
                labels={'Delta_P_LED': 'Potência libertada (kW)', 'y': 'Município'},
                color_discrete_sequence=[COR_AMARELO]
            )
            fig_led.update_layout(
                **LAYOUT_BASE,
                xaxis=dict(gridcolor='#e9ecef'),
                yaxis=dict(gridcolor='#e9ecef')
            )
            st.plotly_chart(fig_led, use_container_width=True)

        st.markdown("---")
        st.subheader("Viabilidade por Município")

        col_viab1, col_viab2 = st.columns(2)

        with col_viab1:
            viab_counts = (
                df['Viavel']
                .map({True: 'Suporta VE', False: 'Requer Expansão da Rede'})
                .value_counts()
                .reset_index()
            )
            viab_counts.columns = ['Status', 'Contagem']
            fig_pie_viab = px.pie(
                viab_counts, names='Status', values='Contagem',
                title='Proporção de Viabilidade dos Municípios',
                color='Status',
                color_discrete_map={
                    'Suporta VE': COR_VERDE,
                    'Requer Expansão da Rede': COR_VERMELHO
                },
                hole=0.45
            )
            fig_pie_viab.update_layout(**LAYOUT_BASE)
            st.plotly_chart(fig_pie_viab, use_container_width=True)

        with col_viab2:
            # ── FIX PRINCIPAL ─────────────────────────────────────────────
            # Garantir que 'Concelho' existe antes de seleccionar as colunas
            df_work = df.copy()
            if 'Concelho' not in df_work.columns:
                df_work['Concelho'] = df_work.index.astype(str)

            df_saldo = df_work[['Concelho', 'D']].dropna().sort_values('D', ascending=False)
            top5     = df_saldo.head(5)
            bot5     = df_saldo.tail(5)
            df_extremos = pd.concat([top5, bot5])
            cores = [COR_VERDE if v >= 0 else COR_VERMELHO for v in df_extremos['D']]

            fig_saldo = go.Figure(go.Bar(
                x=df_extremos['Concelho'],
                y=df_extremos['D'],
                marker_color=cores
            ))
            fig_saldo.update_layout(
                **LAYOUT_BASE,
                title='Saldo de Viabilidade (D) — Top e Bottom 5',
                xaxis=dict(title='Município', gridcolor='#e9ecef'),
                yaxis=dict(title='D (kW)', gridcolor='#e9ecef'),
                shapes=[dict(
                    type='line', x0=-0.5, x1=len(df_extremos) - 0.5,
                    y0=0, y1=0,
                    line=dict(color='#6c757d', width=1.5, dash='dash')
                )]
            )
            st.plotly_chart(fig_saldo, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Análise Exploratória
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Análise Exploratória de Dados")

        col_eda1, col_eda2 = st.columns(2)

        with col_eda1:
            fig_scatter = px.scatter(
                df, x='Taxa_Inef', y='Util_Media',
                size='Cap_PTD_kW',
                color='Viavel',
                hover_name='Concelho' if 'Concelho' in df.columns else None,
                title='Ineficiência da Iluminação vs Ocupação do PTD',
                labels={
                    'Taxa_Inef': 'Taxa de ineficiência IP (%)',
                    'Util_Media': 'Utilização média PTD'
                },
                color_discrete_map={True: COR_VERDE, False: COR_VERMELHO}
            )
            fig_scatter.update_layout(
                **LAYOUT_BASE,
                xaxis=dict(gridcolor='#e9ecef'),
                yaxis=dict(tickformat='.0%', gridcolor='#e9ecef'),
                legend_title_text='Viável'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_eda2:
            fig_corr = px.scatter(
                df, x='P_IP_Total', y='Cap_PTD_kW',
                color='Viavel',
                hover_name='Concelho' if 'Concelho' in df.columns else None,
                title='Capacidade PTD vs Potência IP Total',
                labels={
                    'P_IP_Total': 'Potência IP total (kW)',
                    'Cap_PTD_kW': 'Capacidade PTD (kW)'
                },
                color_discrete_map={True: COR_AZUL, False: COR_VERMELHO},
                trendline='ols'
            )
            fig_corr.update_layout(
                **LAYOUT_BASE,
                xaxis=dict(gridcolor='#e9ecef'),
                yaxis=dict(gridcolor='#e9ecef'),
                legend_title_text='Viável'
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        col_eda3, col_eda4 = st.columns(2)

        with col_eda3:
            if 'Distrito' in df.columns:
                distritos_alvo = ['Lisboa', 'Porto', 'Aveiro', 'Setúbal']
                df_box = df[df['Distrito'].isin(distritos_alvo)]
                if not df_box.empty:
                    fig_box = px.box(
                        df_box, x='Distrito', y='Util_Media',
                        title='Distribuição do Nível de Utilização por Distrito',
                        labels={'Util_Media': 'Utilização média'},
                        color='Distrito',
                        color_discrete_sequence=[COR_AZUL, COR_VERDE, COR_LARANJA, COR_CINZA]
                    )
                    fig_box.update_layout(
                        **LAYOUT_BASE,
                        showlegend=False,
                        yaxis=dict(tickformat='.0%', gridcolor='#e9ecef'),
                        xaxis=dict(gridcolor='#e9ecef')
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("Sem dados para os distritos seleccionados.")
            else:
                st.info("Coluna 'Distrito' não disponível no dataset consolidado.")

        with col_eda4:
            fig_rel = px.scatter(
                df, x='Delta_P_LED', y='P_Folga',
                size='N_PTDs',
                color='Viavel',
                hover_name='Concelho' if 'Concelho' in df.columns else None,
                title='Potência Libertada (LED) vs Folga Disponível na Rede',
                labels={
                    'Delta_P_LED': 'Potência libertada LED (kW)',
                    'P_Folga': 'Folga na rede (kW)'
                },
                color_discrete_map={True: COR_VERDE, False: COR_VERMELHO}
            )
            fig_rel.update_layout(
                **LAYOUT_BASE,
                xaxis=dict(gridcolor='#e9ecef'),
                yaxis=dict(gridcolor='#e9ecef'),
                legend_title_text='Viável'
            )
            st.plotly_chart(fig_rel, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — Mapa
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Mapa das Zonas Analisadas")
        st.caption(
            "Verde: município viável para integração de VE. "
            "Vermelho: município que requer expansão da rede. "
            "O tamanho do círculo representa o número de PTDs do município. "
            "Locais potenciais para carregamento assinalados com estrela."
        )

        colunas_mapa = [
            c for c in ['lat', 'lon', 'Concelho', 'Viavel', 'D', 'N_PTDs',
                        'P_Folga', 'Delta_P_LED', 'Cap_PTD_kW', 'P_IP_Total']
            if c in df.columns
        ]
        df_map = df[colunas_mapa].dropna(subset=['lat', 'lon'])

        if not df_map.empty:
            fig_map = go.Figure()

            df_inviavel = df_map[~df_map['Viavel']]
            if not df_inviavel.empty:
                fig_map.add_trace(go.Scattermapbox(
                    lat=df_inviavel['lat'],
                    lon=df_inviavel['lon'],
                    mode='markers',
                    marker=dict(
                        size=df_inviavel['N_PTDs'].clip(upper=50) / 2 + 5
                              if 'N_PTDs' in df_inviavel.columns else 10,
                        color=COR_VERMELHO,
                        opacity=0.7
                    ),
                    text=df_inviavel['Concelho'] if 'Concelho' in df_inviavel.columns else None,
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "PTDs: %{customdata[0]}<br>"
                        "Folga: %{customdata[1]:.0f} kW<br>"
                        "Saldo D: %{customdata[2]:.0f} kW<br>"
                        "<extra>Requer expansão</extra>"
                    ),
                    customdata=df_inviavel[['N_PTDs', 'P_Folga', 'D']].values
                              if all(c in df_inviavel.columns for c in ['N_PTDs', 'P_Folga', 'D'])
                              else None,
                    name='Requer expansão da rede'
                ))

            df_viavel = df_map[df_map['Viavel']]
            if not df_viavel.empty:
                fig_map.add_trace(go.Scattermapbox(
                    lat=df_viavel['lat'],
                    lon=df_viavel['lon'],
                    mode='markers',
                    marker=dict(
                        size=df_viavel['N_PTDs'].clip(upper=50) / 2 + 5
                              if 'N_PTDs' in df_viavel.columns else 10,
                        color=COR_VERDE,
                        opacity=0.8
                    ),
                    text=df_viavel['Concelho'] if 'Concelho' in df_viavel.columns else None,
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "PTDs: %{customdata[0]}<br>"
                        "Folga: %{customdata[1]:.0f} kW<br>"
                        "Saldo D: %{customdata[2]:.0f} kW<br>"
                        "<extra>Viável para VE</extra>"
                    ),
                    customdata=df_viavel[['N_PTDs', 'P_Folga', 'D']].values
                              if all(c in df_viavel.columns for c in ['N_PTDs', 'P_Folga', 'D'])
                              else None,
                    name='Suporta integração VE'
                ))

            df_pot = df_map[df_map['Viavel']].nlargest(15, 'D') if 'D' in df_map.columns else pd.DataFrame()
            if not df_pot.empty:
                fig_map.add_trace(go.Scattermapbox(
                    lat=df_pot['lat'],
                    lon=df_pot['lon'],
                    mode='markers',
                    marker=dict(size=16, color=COR_AMARELO, symbol='star', opacity=0.95),
                    text=df_pot['Concelho'] if 'Concelho' in df_pot.columns else None,
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Saldo D: %{customdata[0]:.0f} kW<br>"
                        "<extra>Local potencial para carregamento</extra>"
                    ),
                    customdata=df_pot[['D']].values,
                    name='Local potencial para carregamento (Top 15)'
                ))

            fig_map.update_layout(
                mapbox=dict(style='carto-positron', zoom=5.5, center=dict(lat=39.5, lon=-8.0)),
                margin=dict(r=0, t=0, l=0, b=0),
                legend=dict(
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#e2e6ea',
                    borderwidth=1,
                    x=0.01, y=0.99,
                    xanchor='left', yanchor='top',
                    font=dict(size=11)
                ),
                height=580
            )
            st.plotly_chart(fig_map, use_container_width=True)

            if not df_pot.empty:
                st.markdown("**Locais com maior potencial para instalação de carregadores VE (Top 15)**")
                cols_tabela = [c for c in ['Concelho', 'N_PTDs', 'P_Folga', 'Delta_P_LED', 'P_VE', 'D']
                               if c in df_pot.columns]
                st.dataframe(
                    df_pot[cols_tabela].sort_values('D', ascending=False).round(1),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning(
                "Coordenadas geográficas não disponíveis. "
                "Verifique se a coluna 'Coordenadas Geográficas' existe no ficheiro PTD_data.xlsx."
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — Dados Tabelares
    # ════════════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("Dados Tabelares Completos")

        cols_mostrar = [
            c for c in [
                'Concelho', 'Distrito',
                'N_Lampadas_Total', 'N_Lampadas_Inef', 'Taxa_Inef',
                'P_IP_Total', 'P_IP_Inef', 'Delta_P_LED',
                'N_PTDs', 'Cap_PTD', 'Cap_PTD_kW', 'Util_Media',
                'P_Ocupada', 'P_Folga', 'P_VE',
                'D', 'Viavel'
            ] if c in df.columns
        ]

        filtro = st.radio(
            "Filtrar por viabilidade",
            options=["Todos", "Viáveis", "Requerem expansão"],
            horizontal=True
        )
        df_tabela = df.copy()
        if filtro == "Viáveis":
            df_tabela = df_tabela[df_tabela['Viavel'] == True]
        elif filtro == "Requerem expansão":
            df_tabela = df_tabela[df_tabela['Viavel'] == False]

        st.dataframe(
            df_tabela[cols_mostrar].round(2),
            use_container_width=True,
            hide_index=True
        )
        st.caption(f"{len(df_tabela)} registos mostrados de {len(df)} no total.")

else:
    st.error(
        "Não foi possível carregar os dados. "
        "Certifique-se de que os ficheiros IP_data.xlsx e PTD_data.xlsx "
        "estão na mesma pasta que este script."
    )