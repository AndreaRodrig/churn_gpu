import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
import time

# Verificación de disponibilidad de GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Parámetros
ncli = 200  # Número total de clientes a evaluar
pcli = 10   # Punto de inicio para la evaluación (salta primeros 10)

# Conexión a SQL Server
connection_string = (
    "Driver={***};"
    "Server=***;"
    "Database=***;"
    "UID=***;"
    "PWD=***;"
    "Encrypt=***;"
    "TrustServerCertificate=***;"
)

connection = pyodbc.connect(connection_string)
cursor = connection.cursor()

# Cargar clientes activos con más de 20 ventas en los últimos 2 años
with open('clientes_freq4.sql', 'r') as data:
    compradores = data.read()

cursor.execute(compradores)
comprador = list(set([row.iIdCliente for row in cursor.fetchall()]))

# Inicializar métricas
TP, TN, FP, FN = 0, 0, 0, 0
k = 0

def count(vec1, vec2):
    if np.sum(vec1) == 0 and np.sum(vec2) == 0:
        return 1
    elif np.sum(vec1) > 0 and np.sum(vec2) == 0:
        return 0
    elif np.sum(vec1) > 0 and np.sum(vec2) > 0:
        return 1
    else:
        return 0

start_time = time.time()

# Entrenamiento por cliente
for cliente in comprador[pcli:ncli]:
    with open('client_weekall.sql', 'r', encoding='utf-8') as file:
        query = file.read()

    df = pd.read_sql(query, connection, params=(cliente,))
    random = df['TotalTickets'].to_numpy()
    random[random > 0] = 1

    M = len(random)
    N = M - 5
    sequence = random[0:N]
    N = len(sequence)

    X, y = [], []
    window_size = 4
    for i in range(len(sequence) - window_size - 4 + 1):
        X.append(sequence[i:i+window_size])
        y.append(sequence[i+window_size:i+window_size+4])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Modelo LSTM
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(window_size, 1)),
        Dense(32, activation='relu'),
        Dense(4)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2, verbose=0)

    # Predicción para las siguientes 4 semanas
    input_sequence = np.array(sequence[-window_size:]).reshape((1, window_size, 1))
    predicted = model.predict(input_sequence)
    predicted_binary = (predicted > 0.28).astype(int)

    k += count(predicted_binary[0], random[N:M])

    TP += int((np.sum(predicted_binary[0]) > 0) and (np.sum(random[N:M]) > 0))
    TN += int((np.sum(predicted_binary[0]) == 0) and (np.sum(random[N:M]) == 0))
    FP += int((np.sum(predicted_binary[0]) > 0) and (np.sum(random[N:M]) == 0))
    FN += int((np.sum(predicted_binary[0]) == 0) and (np.sum(random[N:M]) > 0))

# Resultados
p = k / len(comprador[pcli:ncli])
print('Accuracy =', p)

# Matriz de confusión
total_cm = np.array([[TN, FP], [FN, TP]])
labels = np.array([[f"TN: {TN}\n{TN/(TN+FP+FN+TP):.2%}", f"FP: {FP}\n{FP/(TN+FP+FN+TP):.2%}"],
                   [f"FN: {FN}\n{FN/(TN+FP+FN+TP):.2%}", f"TP: {TP}\n{TP/(TN+FP+FN+TP):.2%}"]])

plt.figure(figsize=(6, 5))
sns.heatmap(total_cm, annot=labels, fmt='', cmap='hot', cbar=True,
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"],
            linewidths=1, linecolor='black', square=True)
plt.title("Total Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

end_time = time.time()
print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")
