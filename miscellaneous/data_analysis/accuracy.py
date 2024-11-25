import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Leer los dos archivos
df_real = pd.read_csv("C:\\Users\\urigo\\Documents\\Vakas\\data_analysis\\dataset_di.csv")
df_pred = pd.read_csv("C:\\Users\\urigo\\Documents\\Vakas\\data_analysis\\predictionsTrain5.csv")

# Convertir Timestamp a datetime y establecer como índice
df_real["Timestamp"] = pd.to_datetime(df_real["Timestamp"])
df_pred["Timestamp"] = pd.to_datetime(df_pred["Timestamp"])

df_real.set_index("Timestamp", inplace=True)
df_pred.set_index("Timestamp", inplace=True)

# Asegurarse de que los índices sean iguales
df_combined = df_real.join(df_pred, lsuffix='_real', rsuffix='_pred', how='inner')

# Comparar las categorías de cada cama
for bed in ['A', 'B', 'C']:
    df_combined[f'{bed}_correct'] = df_combined[f'{bed}_real'] == df_combined[f'{bed}_pred']

# Calcular accuracy global y por cama
accuracy_per_bed = {bed: df_combined[f'{bed}_correct'].mean() for bed in ['A', 'B', 'C']}
global_accuracy = sum([df_combined[f'{bed}_correct'].sum() for bed in ['A', 'B', 'C']]) / (3 * len(df_combined))

print("Precisión por cama:", accuracy_per_bed)
print(f"Precisión global: {global_accuracy:.2%}")

plt.style.use('bmh')
# Graficar diferencias en distribuciones
for bed in ['A', 'B', 'C']:
    # Crear un DataFrame para gráficas
    graph_df = pd.DataFrame({
        'Value': pd.concat([df_combined[f'{bed}_real'], df_combined[f'{bed}_pred']]),
        'Type': ['Real'] * len(df_combined) + ['Predicción'] * len(df_combined)
    })

    plt.figure(figsize=(10, 6))
    sns.countplot(x='Value', hue='Type', data=graph_df, palette='pastel')
    plt.title(f"Distribución Real vs Predicción para Cama {bed}")
    plt.xlabel("Categoría")
    plt.ylabel("Conteo")
    plt.legend(title="Tipo", loc="upper right")
    plt.tight_layout()
    plt.show()

# Calcular el porcentaje de error por hora del día
df_combined['Hour'] = df_combined.index.hour  # Extraer la hora del día

# Calcular el error por cama
for bed in ['A', 'B', 'C']:
    df_combined[f'{bed}_error'] = ~df_combined[f'{bed}_correct']  # True si hay error, False si es correcto

# Calcular porcentaje de error por hora
error_by_hour = (
    df_combined.groupby('Hour')[[f'{bed}_error' for bed in ['A', 'B', 'C']]]
    .mean()
    .mean(axis=1)  # Promedio de los errores entre las camas
    * 100  # Convertir a porcentaje
)

# Graficar el histograma
plt.figure(figsize=(12, 6))
sns.barplot(x=error_by_hour.index, y=error_by_hour.values, palette="muted")
plt.title("Porcentaje de Error por Hora del Día")
plt.xlabel("Hora del Día")
plt.ylabel("Porcentaje de Error (%)")
plt.xticks(range(0, 24)) 
plt.tight_layout()
plt.show()

# Calcular el porcentaje de precisión por hora del día
accuracy_by_hour = (
    df_combined.groupby('Hour')[[f'{bed}_correct' for bed in ['A', 'B', 'C']]]
    .mean()
    .mean(axis=1)  # Promedio de las precisiones entre las camas
    * 100  # Convertir a porcentaje
)

# Graficar el histograma de precisión
plt.figure(figsize=(12, 6))
sns.lineplot(x=accuracy_by_hour.index, y=accuracy_by_hour.values, marker='o', palette="muted")
plt.axhline(80, color='red', linestyle='--', label='80% Accuracy')
plt.ylim(0, 100)
plt.title("Porcentaje de Accuracy por Hora del Día")
plt.xlabel("Hora del Día")
plt.ylabel("Porcentaje de Accuracy (%)")
plt.xticks(range(0, 24))  # Asegurarse de que todas las horas estén representadas
plt.legend()
plt.tight_layout()
plt.show()
