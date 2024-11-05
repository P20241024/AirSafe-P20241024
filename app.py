import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Cargar modelos pre-entrenados
modelo_stacking = joblib.load('/content/drive/MyDrive/model_stacking.pkl')
modelo_linear_regression = joblib.load('/content/drive/MyDrive/model_linearRegression.pkl')
modelo_random_forest = joblib.load('/content/drive/MyDrive/model_randomForestRegressor.pkl')
modelo_svr = joblib.load('/content/drive/MyDrive/model_svr.pkl')

# Título de la app
st.title("Predicción de Concentración de CO₂")

# Cargar datos de muestra
@st.cache
def cargar_datos():
    df = pd.read_excel("/content/drive/MyDrive/Data/data_1k_tesis_CO2.xlsx")
    return df

df = cargar_datos()

# Ajustes en los datos
df['Temperature'] = df['Temperature'].apply(lambda x: max(0, x))
df['Humidity'] = df['Humidity'].apply(lambda x: max(0, x))
df['Light intensity'] = df['Light intensity'].apply(lambda x: max(0, x))
df = df.dropna()

# Obtener valores de entrada del usuario
st.sidebar.header("Parámetros de Entrada")
temperature = st.sidebar.slider("Temperatura (°C)", float(df['Temperature'].min()), float(df['Temperature'].max()), float(df['Temperature'].mean()))
humidity = st.sidebar.slider("Humedad (%)", float(df['Humidity'].min()), float(df['Humidity'].max()), float(df['Humidity'].mean()))
light_intensity = st.sidebar.slider("Intensidad de Luz (lux)", float(df['Light intensity'].min()), float(df['Light intensity'].max()), float(df['Light intensity'].mean()))
year = st.sidebar.selectbox("Año", [2018, 2019, 2020])
month = st.sidebar.selectbox("Mes", list(range(1, 13)))
day = st.sidebar.selectbox("Día", list(range(1, 29)))
hour = st.sidebar.slider("Hora del día", 0, 23, 12)

# Función para predicción
def predecir_concentracion_co2(modelo, year, month, day, hour, temp, humidity, light_intensity):
    data = pd.DataFrame({
        'Year': [year], 'Month': [month], 'Day': [day], 'Hour': [hour],
        'Temperature': [temp], 'Humidity': [humidity], 'Light intensity': [light_intensity]
    })
    return modelo.predict(data)[0]

# Predicciones de diferentes modelos
stacking_pred = predecir_concentracion_co2(modelo_stacking, year, month, day, hour, temperature, humidity, light_intensity)
linear_pred = predecir_concentracion_co2(modelo_linear_regression, year, month, day, hour, temperature, humidity, light_intensity)
rf_pred = predecir_concentracion_co2(modelo_random_forest, year, month, day, hour, temperature, humidity, light_intensity)
svr_pred = predecir_concentracion_co2(modelo_svr, year, month, day, hour, temperature, humidity, light_intensity)

# Mostrar predicciones
st.subheader("Resultados de Predicción")
st.write(f"Predicción de CO₂ usando Stacking: {stacking_pred:.2f} ppm")
st.write(f"Predicción de CO₂ usando Regressión Lineal: {linear_pred:.2f} ppm")
st.write(f"Predicción de CO₂ usando Random Forest: {rf_pred:.2f} ppm")
st.write(f"Predicción de CO₂ usando SVR: {svr_pred:.2f} ppm")

# Comparación de métricas de los modelos
st.subheader("Métricas de Modelos")
metrics = {
    "Model": ["Stacking", "Linear Regression", "Random Forest", "SVR"],
    "Predicción CO₂ (ppm)": [stacking_pred, linear_pred, rf_pred, svr_pred]
}
df_metrics = pd.DataFrame(metrics)
st.write(df_metrics)

# Gráfico de comparación de predicciones
st.subheader("Comparación de Predicciones")
fig, ax = plt.subplots()
ax.bar(df_metrics['Model'], df_metrics['Predicción CO₂ (ppm)'], color=['blue', 'green', 'orange', 'purple'])
ax.set_ylabel("CO₂ Predicho (ppm)")
st.pyplot(fig)

# Predicción a lo largo del día
st.subheader("Predicciones de CO₂ a lo largo del día")
horas = [hour + i for i in range(6)]  # Incremento de 6 horas
predicciones_dia = [predecir_concentracion_co2(modelo_stacking, year, month, day, h % 24, temperature, humidity, light_intensity) for h in horas]

fig, ax = plt.subplots()
ax.plot(horas, predicciones_dia, marker='o')
ax.set_xlabel("Hora")
ax.set_ylabel("Predicción de CO₂ (ppm)")
st.pyplot(fig)
