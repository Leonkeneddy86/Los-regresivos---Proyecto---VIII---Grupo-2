import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title('Predicción')

class LabelEncoder:
    def __init__(self):
        self.classes_ = []
        self.class_to_int = {}

    def fit(self, values):
        uniques = pd.Series(values).dropna().unique().tolist()
        self.classes_ = list(uniques)
        self.class_to_int = {v: i for i, v in enumerate(self.classes_)}
        return self

    def transform(self, values):
        return [self.class_to_int.get(v, -1) for v in values]

    def transform_scalar(self, v):
        return self.class_to_int.get(v, -1)

# --- POSTPROCESADOR ---
class PostProcesador:
    def round_post(self, y):
        return np.round(y).astype(int)

# --- CARGA DEL MODELO ---
try:
    pipeline = joblib.load('model.pkl')
    st.success("Modelo cargado correctamente")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

post = PostProcesador()

# --- CARGA DE CATEGORÍAS (para poblar selectboxes y encoders) ---
try:
    df = pd.read_csv('data/clean_data.csv')
except Exception as e:
    st.error(f"No se pudo leer 'data/clean_data.csv': {e}")
    st.stop()

gender_enc = LabelEncoder().fit(sorted(df['Gender'].dropna().unique()))
cust_enc = LabelEncoder().fit(sorted(df['Customer Type'].dropna().unique()))
travel_enc = LabelEncoder().fit(sorted(df['Type of Travel'].dropna().unique()))
class_enc = LabelEncoder().fit(sorted(df['Class'].dropna().unique()))
satisfaction_enc = LabelEncoder().fit(sorted(df['satisfaction'].dropna().unique()))


# --- FORMULARIO DE PREDICCIÓN ---
with st.form("prediction_form"):
    st.header("Ingrese los datos del vehículo")

    gender = st.number_input("Indique Genero", min_value=1990, max_value=2025, step=1)
    client = st.number_input("Tipo de cliente", min_value=0, step=1000)
    travel = st.number_input("Tipo de viaje", min_value=0, max_value=16, step=1)
    class_travel = st.number_input("Clase", min_value=0, step=10)
    satisfaction = st.number_input("Tipo de satisfaccion", min_value=0.0, max_value=20.0, step=0.1)

    gender = st.selectbox("Genero", gender_enc.classes_)
    client = st.selectbox("Tipo de cliente", cust_enc.classes_)
    travel = st.selectbox("Tipo de viaje", travel_enc.classes_)
    class_travel = st.selectbox("Tipo de clase", class_enc.classes_)
    satisfaction = st.selectbox("Grado de satisfaccion", satisfaction_enc.classes_)
    submit_button = st.form_submit_button("Predecir Precio")

if submit_button:
    try:
        # Codificar los valores categóricos
        gender_encoded = gender_enc.transform_scalar(gender)
        client_encoded = cust_enc.transform_scalar(client)
        travel_encoded = travel_enc.transform_scalar(travel)
        class_encoded = class_enc.transform_scalar(class_travel)
        satisfaction_encoded = satisfaction_enc.transform_scalar(satisfaction)

        # Crear diccionario con las características en el ORDEN EXACTO del modelo
        input_dict = {
            "Gender_label": [gender_encoded],
            "Customer_Type_label": [client_encoded],
            "Type_of_Travel_label": [travel_encoded],
            "Class_label": [class_encoded],
            "Age": [0],
            "Flight Distance": [0],
            "Inflight wifi service": [0],
            "Departure/Arrival time convenient": [0],
            "Ease of Online booking": [0],
            "Gate location": [0],
            "Food and drink": [0],
            "Online boarding": [0],
            "Seat comfort": [0],
            "Inflight entertainment": [0],
            "On-board service": [0],
            "Leg room service": [0],
            "Baggage handling": [0],
            "Checkin service": [0],
            "Inflight service": [0],
            "Cleanliness": [0],
            "Departure Delay in Minutes": [0],
            "Arrival Delay in Minutes": [0],
        }

        input_data = pd.DataFrame(input_dict)

        prediccion = pipeline.predict(input_data)
        prediccion_final = post.round_post(prediccion)[0]

        st.success(f"Precio predicho: ${prediccion_final:,.2f}")

        with st.expander("Detalles de la predicción"):
            st.write(f"**Genero:** {gender}")
            st.write(f"**Tipo de cliente:** {client}")
            st.write(f"**Tipo de viaje:** {travel}")
            st.write(f"**Tipo de clase:** {class_travel}")
            st.write(f"**Satisfaccion:** {satisfaction}")

    except Exception as e:
        st.error(f"Error al hacer la predicción: {e}")