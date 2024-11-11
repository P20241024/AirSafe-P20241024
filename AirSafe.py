import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
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

# Crear el gráfico de barras con colores específicos para los niveles de CO₂
st.subheader("Gráfico de Predicciones de CO₂")
limite_co2 = 400  # Límite de seguridad para el CO₂ en ppm
horas_del_dia_str = [pred[0][-5:] for pred in future_predictions]  # Extraer solo la hora de cada timestamp
predicciones_por_hora = [pred[1] for pred in future_predictions]
colores = ['red' if pred > limite_co2 else 'green' for pred in predicciones_por_hora]

# Crear el gráfico de barras
plt.figure(figsize=(8, 6))
barras = plt.bar(horas_del_dia_str, predicciones_por_hora, color=colores)

# Añadir etiquetas en cada barra para mostrar el valor de la predicción
for barra, pred in zip(barras, predicciones_por_hora):
    altura = barra.get_height()
    color_texto = 'red' if pred > limite_co2 else 'green'
    etiqueta = f"{pred:.2f}\n{'CO2 Peligroso' if pred > limite_co2 else 'CO2 Seguro'}"
    plt.text(barra.get_x() + barra.get_width() / 2, altura - 3, etiqueta, ha='center', va='bottom', fontsize=10, color=color_texto)

# Configurar el gráfico
plt.xlabel('Hora del día')
plt.ylabel('Predicciones de CO₂ (ppm)')
plt.title(f'Predicciones de CO₂ a lo largo del día - {current_time.strftime("%Y-%m-%d")}')
plt.ylim(0, max(predicciones_por_hora) * 1.1)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Mostrar el gráfico en Streamlit
st.pyplot(plt)


st.subheader("Aviso de Seguridad")
threshold = 400
if co2_prediction > threshold:
    st.error("¡Alerta! La concentración de CO₂ para la jornada establecida supera el límite de seguridad.")
else:
    st.success("La concentración de CO₂ para la jornada establecida está dentro de los límites de seguridad.")
