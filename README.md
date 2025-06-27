# Churn Prediction con LSTM en GPU

Este proyecto aplica una red neuronal tipo LSTM para predecir la recurrencia de clientes semanales en base a su historial de compras. Utiliza **TensorFlow con soporte para GPU** para acelerar el entrenamiento.

## Requisitos

- Python 3.9+
- TensorFlow (con soporte para GPU)
- Acceso a SQL Server
- Keras
- cuDF (opcional, si se quiere usar GPU también para procesamiento de datos)

Puedes instalar las dependencias con:

```bash
pip install -r requirements.txt

## Nota

Funciona mejor en Jupyter
