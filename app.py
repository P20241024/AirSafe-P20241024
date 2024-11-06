import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Cargar el modelo y el escalador
stacking_model = joblib.load('/content/drive/MyDrive/model_randomForestRegressor.pkl')  # Modifica esta ruta
scaler = joblib.load('/content/drive/MyDrive/scaler.pkl')  # Asegúrate de tener el escalador guardado también

# Configurar la página de Streamlit
st.title("Predicción de CO₂ en Entornos Subterráneos")
st.write("Este sistema permite predecir la concentración de CO₂ basándose en parámetros ambientales.")

# Entradas para predicción
st.sidebar.header("Parámetros de entrada")
year = st.sidebar.selectbox("Año", options=[2022, 2023, 2024], index=0)
month = st.sidebar.selectbox("Mes", options=list(range(1, 13)), index=3)
day = st.sidebar.selectbox("Día", options=list(range(1, 29)), index=19)
hour = st.sidebar.selectbox("Hora", options=list(range(0, 24)), index=14)

temperature = st.sidebar.slider("Temperatura (°C)", 0.0, 40.0, 25.0)
humidity = st.sidebar.slider("Humedad (%)", 0.0, 100.0, 50.0)
light_intensity = st.sidebar.slider("Intensidad de luz (lux)", 0.0, 2000.0, 500.0)

# Preparar los datos de entrada
input_data = pd.DataFrame({
    'Year': [year],
    'Month': [month],
    'Day': [day],
    'Hour': [hour],
    'Temperature': [temperature],
    'Humidity': [humidity],
    'Light intensity': [light_intensity]
})

# Escalar los datos
input_data_scaled = scaler.transform(input_data)

# Realizar predicción
co2_prediction = stacking_model.predict(input_data_scaled)[0]
st.subheader("Predicción de CO₂:")
st.metric("Concentración de CO₂ predicha (ppm)", f"{co2_prediction:.2f}")

# Simulación y gráficos
st.subheader("Predicción de CO₂ para las próximas horas")
num_hours = st.slider("Horas a predecir", 1, 12, 6)
future_predictions = []

# Generar predicciones para las próximas horas
current_time = datetime(year, month, day, hour)
for i in range(num_hours):
    future_time = current_time + timedelta(hours=i)
    input_data['Year'] = future_time.year
    input_data['Month'] = future_time.month
    input_data['Day'] = future_time.day
    input_data['Hour'] = future_time.hour
    input_data_scaled = scaler.transform(input_data)
    future_co2 = stacking_model.predict(input_data_scaled)[0]
    future_predictions.append((future_time.strftime("%Y-%m-%d %H:%M"), future_co2))

# Mostrar resultados en tabla y gráficos
future_df = pd.DataFrame(future_predictions, columns=["Fecha y Hora", "CO₂ (ppm)"])
st.table(future_df)

# Gráfico de predicciones
st.line_chart(future_df.set_index("Fecha y Hora"))

st.subheader("Aviso de Seguridad")
threshold = 400
if co2_prediction > threshold:
    st.error("¡Alerta! La concentración de CO₂ para la jornada establecida supera el límite de seguridad.")
else:
    st.success("La concentración de CO₂ para la jornada establecida está dentro de los límites de seguridad.")
