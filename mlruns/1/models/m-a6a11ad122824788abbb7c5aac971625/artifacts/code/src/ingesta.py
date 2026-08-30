from pathlib import Path
from ucimlrepo import fetch_ucirepo


# Carpeta donde se encuentra este archivo ingest.py
CURRENT_DIR = Path(__file__).resolve().parent

# Descargar dataset desde UCI
adult = fetch_ucirepo(id=2)

# Obtener características y variable objetivo
X = adult.data.features
y = adult.data.targets

# Unir X e y
data = X.copy()
data[y.columns[0]] = y.iloc[:, 0]

# Guardar dataset raw en la misma carpeta que ingest.py
output_path = CURRENT_DIR / "adult_raw.csv"
data.to_csv(output_path, index=False)

print("Dataset descargado correctamente.")
print(f"Registros: {data.shape[0]}")
print(f"Columnas: {data.shape[1]}")
print(f"Archivo guardado en: {output_path}")