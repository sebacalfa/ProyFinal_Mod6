# Imagen liviana: solo lo necesario para SERVIR el modelo, no para reentrenarlo.
# Por eso no incluye jupyterlab, matplotlib, seaborn ni ucimlrepo:
# esas dependencias son para entrenar/explorar, no para responder /predict.
# Esto es justamente lo que hace que el contenedor funcione bien incluso
# en computadoras de menor capacidad.
FROM python:3.13.14-slim

WORKDIR /app

# Dependencias del sistema necesarias para compilar algunas ruedas de Python
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1) Instalar dependencias primero: Docker cachea esta capa y solo la
#    reconstruye si requirements-api.txt cambia (builds mas rapidos).
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 2) Copiar SOLO el codigo fuente necesario para servir (no notebooks, no CSVs)
COPY src/pipeline_datos.py src/pipeline_datos.py
COPY src/monitoring.py src/monitoring.py
COPY src/api/ src/api/

# 3) Copiar SOLO el modelo entrenado y su manifiesto (no todo resultado_pipeline,
#    que incluye CSVs de varios MB que no hacen falta para servir el modelo).
COPY resultado_pipeline/modelo/production/ resultado_pipeline/modelo/production/

# Necesario para que "import src.pipeline_datos" funcione al desempaquetar
# el modelo (fue guardado con joblib.dump, que exige la clase importable).
ENV PYTHONPATH=/app

EXPOSE 8010

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8010"]
