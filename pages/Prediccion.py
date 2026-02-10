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
    st.title("📊 Predicción de Satisfacción del Cliente")
    st.markdown("Ingrese los datos del cliente para determinar su nivel de satisfacción.")
    st.header("Ingrese los datos")

    # Usar las clases del encoder para garantizar consistencia con el entrenamiento
    gender = st.selectbox("Indique Genero", options=gender_enc.classes_)
    client = st.selectbox("Tipo de cliente", options=cust_enc.classes_)
    travel = st.selectbox("Tipo de viaje", options=travel_enc.classes_)
    class_travel = st.selectbox("Tipo de Clase", options=class_enc.classes_)

    # Inputs numéricos (ajustar rangos/valores por defecto según su dataset)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    flight_distance = st.number_input("Distancia del vuelo", min_value=0, value=0)
    inflight_wifi = st.slider("Servicio de wifi a bordo", 0, 5, 3)
    departure_arrival_time = st.slider("Conveniencia de horario salida/llegada", 0, 5, 3)
    ease_online_booking = st.slider("Facilidad de reserva online", 0, 5, 3)
    gate_location = st.slider("Ubicación de la puerta", 0, 5, 3)
    food_and_drink = st.slider("Comida y bebida", 0, 5, 3)
    online_boarding = st.slider("Embarque online", 0, 5, 3)
    seat_comfort = st.slider("Comodidad del asiento", 0, 5, 3)
    inflight_entertainment = st.slider("Entretenimiento a bordo", 0, 5, 3)
    on_board_service = st.slider("Servicio a bordo", 0, 5, 3)
    leg_room_service = st.slider("Espacio para las piernas", 0, 5, 3)
    baggage_handling = st.slider("Manejo de equipaje", 0, 5, 3)
    checkin_service = st.slider("Servicio de check-in", 0, 5, 3)
    inflight_service = st.slider("Servicio en vuelo", 0, 5, 3)
    cleanliness = st.slider("Limpieza", 0, 5, 3)
    departure_delay = st.number_input("Retraso en salida (minutos)", min_value=0, value=0)
    arrival_delay = st.number_input("Retraso en llegada (minutos)", min_value=0, value=0)

    submit_button = st.form_submit_button("Predecir")

if submit_button:
    try:
        # Codificar los valores categóricos
        gender_encoded = gender_enc.transform_scalar(gender)
        client_encoded = cust_enc.transform_scalar(client)
        travel_encoded = travel_enc.transform_scalar(travel)
        class_encoded = class_enc.transform_scalar(class_travel)

        # Crear diccionario con las características en el ORDEN EXACTO del modelo
        input_dict = {
            "Gender_label": [gender_encoded],
            "Customer_Type_label": [client_encoded],
            "Type_of_Travel_label": [travel_encoded],
            "Class_label": [class_encoded],
            "Age": [age],
            "Flight Distance": [flight_distance],
            "Inflight wifi service": [inflight_wifi],
            "Departure/Arrival time convenient": [departure_arrival_time],
            "Ease of Online booking": [ease_online_booking],
            "Gate location": [gate_location],
            "Food and drink": [food_and_drink],
            "Online boarding": [online_boarding],
            "Seat comfort": [seat_comfort],
            "Inflight entertainment": [inflight_entertainment],
            "On-board service": [on_board_service],
            "Leg room service": [leg_room_service],
            "Baggage handling": [baggage_handling],
            "Checkin service": [checkin_service],
            "Inflight service": [inflight_service],
            "Cleanliness": [cleanliness],
            "Departure Delay in Minutes": [departure_delay],
            "Arrival Delay in Minutes": [arrival_delay],
        }

        input_data = pd.DataFrame(input_dict)

        # Predicción
        pred = pipeline.predict(input_data)
        pred_int = int(np.round(pred[0])) if pred.dtype.kind in 'f' else int(pred[0])

        # Inversa del encoder para obtener la etiqueta original
        try:
            raw_label = satisfaction_enc.classes_[pred_int]
        except Exception:
            raw_label = str(pred_int)

        # Mapeo a etiquetas
        label_map = {
            "satisfied": "Satisfecho",
            "neutral": "Neutral",
            "neutral or dissatisfied": "Neutral",
            "dissatisfied": "Insatisfecho",
            "unsatisfied": "Insatisfecho",
            "not satisfied": "Insatisfecho"
        }
        label = label_map.get(str(raw_label).lower(), str(raw_label))

        st.success(f"Predicción: {label}")

        # Si el modelo soporta probabilidades, mostrar
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(input_data)[0]
            st.write("Probabilidades:", probs)

        with st.expander("Detalles de la predicción"):
            st.write(f"**Genero:** {gender}")
            st.write(f"**Tipo de cliente:** {client}")
            st.write(f"**Tipo de viaje:** {travel}")
            st.write(f"**Tipo de clase:** {class_travel}")

    except Exception as e:
        st.error(f"Error al hacer la predicción: {e}")