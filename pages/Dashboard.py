import os
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Dashboard de Satisfacción", layout="wide")
st.title("📈 Dashboard — Satisfacción de Pasajeros")

# Posibles datasets (prioriza data/clean_data.csv)
DATA_OPTIONS = []
if os.path.exists("data/clean_data.csv"):
    DATA_OPTIONS.append("data/clean_data.csv")
if os.path.exists("airline_passenger_satisfaction.csv"):
    DATA_OPTIONS.append("airline_passenger_satisfaction.csv")

if not DATA_OPTIONS:
    st.error("❌ No hay datasets disponibles. Añade `data/clean_data.csv` o `airline_passenger_satisfaction.csv`.")
    st.stop()

st.sidebar.title("📦 Dataset")
selected_file = st.sidebar.selectbox("Selecciona el dataset", DATA_OPTIONS)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(selected_file)

# Normalizar nombre de la columna de satisfacción
if "satisfaction" not in df.columns:
    st.error("El dataset no contiene la columna 'satisfaction'.")
    st.stop()

# Crear columna booleana para satisfacción
df["is_satisfied"] = df["satisfaction"].str.lower() == "satisfied"

# =========================
# SIDEBAR — FILTROS
# =========================
st.sidebar.header("Filtros")

# Clase
class_opts = [c for c in sorted(df["Class"].dropna().unique())]
sel_class = st.sidebar.selectbox("Clase", ["All"] + class_opts)
if sel_class != "All":
    df = df[df["Class"] == sel_class]

# Tipo de viaje
travel_opts = [c for c in sorted(df["Type of Travel"].dropna().unique())]
sel_travel = st.sidebar.selectbox("Tipo de viaje", ["All"] + travel_opts)
if sel_travel != "All":
    df = df[df["Type of Travel"] == sel_travel]

# Customer Type
cust_opts = [c for c in sorted(df["Customer Type"].dropna().unique())]
sel_cust = st.sidebar.selectbox("Tipo de cliente", ["All"] + cust_opts)
if sel_cust != "All":
    df = df[df["Customer Type"] == sel_cust]

# Rango de edad
age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
sel_age = st.sidebar.slider("Edad", min_value=age_min, max_value=age_max, value=(age_min, age_max))
df = df[df["Age"].between(*sel_age)]

# Rango de distancia de vuelo
fd_min, fd_max = int(df["Flight Distance"].min()), int(df["Flight Distance"].max())
sel_fd = st.sidebar.slider("Flight Distance", min_value=fd_min, max_value=fd_max, value=(fd_min, fd_max))
df = df[df["Flight Distance"].between(*sel_fd)]

if df.empty:
    st.warning("⚠️ No hay datos con estos filtros. Ajusta los filtros.")
    st.stop()

# =========================
# KPIs
# =========================
st.subheader("KPIs")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros", f"{len(df):,}")
pct = df["is_satisfied"].mean() * 100
col2.metric("% Satisfechos", f"{pct:.1f}%")
col3.metric("Edad media", f"{df['Age'].mean():.1f}")
col4.metric("Dist. vuelo media", f"{df['Flight Distance'].mean():.0f}")

st.divider()

# =========================
# VISUALIZACIONES
# =========================
tab1, tab2, tab3 = st.tabs(["📊 Resumen", "🔎 Servicios", "🧾 Datos"])

with tab1:
    st.subheader("Distribución de satisfacción")
    sat_counts = df["satisfaction"].value_counts().reset_index()
    sat_counts.columns = ["satisfaction", "count"]
    chart = alt.Chart(sat_counts).mark_bar().encode(
        x=alt.X("satisfaction:N", sort="-y"),
        y="count:Q",
        color="satisfaction:N",
        tooltip=["satisfaction", "count"]
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Edad vs Flight Distance (coloreado por satisfacción)")
    scatter = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x="Age:Q",
            y="Flight Distance:Q",
            color=alt.Color("satisfaction:N"),
            tooltip=["Age", "Flight Distance", "satisfaction"]
        )
    )
    st.altair_chart(scatter.interactive(), use_container_width=True)

with tab2:
    st.subheader("Promedio de puntuaciones por servicio")
    service_cols = [
        'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
        'Inflight entertainment', 'On-board service', 'Leg room service',
        'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
    ]
    # Filtrar columnas que existan en el dataset
    service_cols = [c for c in service_cols if c in df.columns]
    svc_mean = df[service_cols].mean().reset_index()
    svc_mean.columns = ["service", "mean"]
    svc_chart = alt.Chart(svc_mean).mark_bar().encode(
        x=alt.X("mean:Q"),
        y=alt.Y("service:N", sort="-x"),
        tooltip=["service", "mean"]
    )
    st.altair_chart(svc_chart, use_container_width=True)

    st.subheader("Satisfacción por Clase")
    cls = df.groupby("Class")["is_satisfied"].mean().reset_index()
    cls["pct_satisfied"] = cls["is_satisfied"] * 100
    cls_chart = alt.Chart(cls).mark_bar().encode(
        x=alt.X("Class:N"),
        y=alt.Y("pct_satisfied:Q"),
        color=alt.Color("Class:N"),
        tooltip=["Class", "pct_satisfied"]
    )
    st.altair_chart(cls_chart, use_container_width=True)

with tab3:
    st.subheader("Dataset filtrado")
    st.dataframe(df.reset_index(drop=True), use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV filtrado", csv, file_name="passenger_satisfaction_filtered.csv", mime="text/csv")
