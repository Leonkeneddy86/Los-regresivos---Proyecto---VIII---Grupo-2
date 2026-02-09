import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Car Price Dashboard", layout="wide")

DATA_PATH = "data"

PRICE_COL = "price"
YEAR_COL = "model_year"
EXT_COL = "ext_col"
INT_COL = "int_col"
TRANS_COL = "transmission"
BRAND_COL = "brand"

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
if not files:
    st.error("❌ No hay archivos CSV en la carpeta /data")
    st.stop()

st.sidebar.title("📦 Dataset")
selected_file = st.sidebar.selectbox("Selecciona el dataset", files)
df = load_data(os.path.join(DATA_PATH, selected_file))

# =========================
# BASIC CLEANING
# =========================
df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")

df[TRANS_COL] = df[TRANS_COL].apply(
    lambda x: "-" if str(x).strip() == "-" else str(x).strip()
)

df = df[df[PRICE_COL] > 0].copy()

# =========================
# SIDEBAR FILTERS (CASCADA)
# =========================
st.sidebar.title("Filtros")

# -------- BRAND (manda sobre todo)
brands = df[BRAND_COL].value_counts().head(25).index.tolist()
selected_brand = st.sidebar.selectbox("Marca", brands)

df_b = df[df[BRAND_COL] == selected_brand].copy()

# -------- YEAR (slider como antes)
year_min, year_max = int(df_b[YEAR_COL].min()), int(df_b[YEAR_COL].max())
year_range = st.sidebar.slider(
    "Año",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

df_y = df_b[df_b[YEAR_COL].between(*year_range)].copy()

# -------- PRICE (slider como antes)
price_min, price_max = float(df_y[PRICE_COL].min()), float(df_y[PRICE_COL].max())
price_range = st.sidebar.slider(
    "Precio",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, price_max)
)

df_p = df_y[df_y[PRICE_COL].between(*price_range)].copy()

# -------- TRANSMISSION (dropdown)
trans_opts = sorted(df_p[TRANS_COL].unique().tolist())
sel_trans = st.sidebar.selectbox("Transmisión", ["All"] + trans_opts)

df_t = df_p if sel_trans == "All" else df_p[df_p[TRANS_COL] == sel_trans]

# -------- EXTERIOR COLOR (dropdown)
ext_opts = df_t[EXT_COL].value_counts().head(20).index.tolist()
sel_ext = st.sidebar.selectbox("Color exterior", ["All"] + ext_opts)

df_e = df_t if sel_ext == "All" else df_t[df_t[EXT_COL] == sel_ext]

# -------- INTERIOR COLOR (dropdown)
int_opts = df_e[INT_COL].value_counts().head(20).index.tolist()
sel_int = st.sidebar.selectbox("Color interior", ["All"] + int_opts)

df_f = df_e if sel_int == "All" else df_e[df_e[INT_COL] == sel_int]

if df_f.empty:
    st.warning("⚠️ No hay datos con estos filtros. Amplía rangos o selecciona 'All'.")
    st.stop()

# =========================
# HEADER + KPIs
# =========================
st.title("🚗 Car Price Dashboard")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Marca", selected_brand)
k2.metric("Registros", f"{len(df_f):,}")
k3.metric("Precio medio", f"{df_f[PRICE_COL].mean():,.0f}")
k4.metric("Precio mediano", f"{df_f[PRICE_COL].median():,.0f}")

st.divider()

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📊 Exploración", "🎨 Colores (media)", "🧾 Datos"])

# -------- TAB 1 --------
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribución del precio")
        fig, ax = plt.subplots()
        ax.hist(df_f[PRICE_COL], bins=40)
        ax.set_xlabel("Price")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with c2:
        st.subheader("Año vs Precio")
        fig, ax = plt.subplots()
        ax.scatter(df_f[YEAR_COL], df_f[PRICE_COL], alpha=0.4)
        ax.set_yscale("log")
        ax.set_xlabel("Model year")
        ax.set_ylabel("Price (log)")
        st.pyplot(fig)

    st.subheader("Precio por transmisión")
    fig, ax = plt.subplots(figsize=(10,4))
    sns.boxplot(data=df_f, x=TRANS_COL, y=PRICE_COL, ax=ax)
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)

# -------- TAB 2 --------
with tab2:
    st.subheader("Top 5 colores exteriores más caros (media)")
    top_ext = df_f[EXT_COL].value_counts().head(10).index
    top5_ext = (
        df_f[df_f[EXT_COL].isin(top_ext)]
        .groupby(EXT_COL)[PRICE_COL]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )

    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=top5_ext.index, y=top5_ext.values, ax=ax)
    ax.set_xlabel("Exterior color")
    ax.set_ylabel("Mean price")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)

    st.subheader("Top 5 colores interiores más caros (media)")
    top_int = df_f[INT_COL].value_counts().head(10).index
    top5_int = (
        df_f[df_f[INT_COL].isin(top_int)]
        .groupby(INT_COL)[PRICE_COL]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )

    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=top5_int.index, y=top5_int.values, ax=ax)
    ax.set_xlabel("Interior color")
    ax.set_ylabel("Mean price")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)

# -------- TAB 3 --------
with tab3:
    st.subheader("Dataset filtrado")
    st.dataframe(df_f, use_container_width=True)

    st.download_button(
        "⬇️ Descargar CSV filtrado",
        df_f.to_csv(index=False).encode("utf-8"),
        file_name=f"cars_filtered_{selected_brand}.csv",
        mime="text/csv"
    )