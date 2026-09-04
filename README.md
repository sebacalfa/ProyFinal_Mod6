# Adult Income: proyecto MLOps de principio a fin

Este proyecto nació como una forma de llevar un modelo de machine learning más allá del notebook. A partir del dataset **Adult / Census Income** de UCI, se entrena un clasificador que estima si el ingreso anual de una persona supera los USD 50 000.

El repositorio reúne el flujo completo: descarga y validación de datos, análisis exploratorio, preparación de variables, comparación de modelos, seguimiento de experimentos con MLflow, una API de inferencia, pruebas automatizadas y monitoreo básico en producción.

## 1. Problema de negocio

El objetivo es estimar la probabilidad de que una persona pertenezca a la clase `>50K` usando variables demográficas, educativas y laborales. También se revisa por separado el desempeño para los grupos `Female` y `Male`, ya que una métrica general puede ocultar diferencias importantes entre subgrupos.

La variable objetivo se codifica de la siguiente forma:

- `0`: ingreso `<=50K`.
- `1`: ingreso `>50K`.

Este modelo es un ejercicio académico y debe entenderse como una herramienta de análisis. No está diseñado para tomar decisiones sensibles sobre personas. Un uso real exigiría, como mínimo, una revisión más profunda de sesgos, riesgos y requisitos legales.

## 2. Dataset

- Fuente: [UCI Machine Learning Repository - Adult](https://archive.ics.uci.edu/dataset/2/adult).
- Identificador utilizado por `ucimlrepo`: `2`.
- Tipo de problema: clasificación binaria.
- Dimensión raw observada: 48 842 filas y 15 columnas.
- Dimensión después de eliminar duplicados: 48 790 filas y 15 columnas.
- Variable objetivo: `income`.
- Proporción observada de la clase positiva: aproximadamente 23,94 %.

El script de ingesta descarga los datos desde UCI, por lo que no es obligatorio depender de un CSV preparado manualmente.

## 3. Qué incluye el proyecto

| Componente | Estado | Evidencia principal |
|---|---:|---|
| Ingesta reproducible | Implementado | `src/ingesta.py` y `src/pipeline_datos.py` |
| Data Quality Gates | Implementado | `src/pipeline_datos.py` y reporte CSV |
| EDA orientado a decisiones | Implementado | `H_EDA_decisiones.ipynb` |
| Feature engineering reutilizable | Implementado | `AdultFeatureEngineer` y pipeline de scikit-learn |
| Modelado y experimentación | Implementado | `src/pipeline_ML.py` y notebook J |
| MLflow Tracking | Implementado | `mlflow.db`, `mlruns/` y registro desde el pipeline |
| Model Registry | Ciclo Experiment → Candidate → Validation → Production | `adult-income-classifier`, gates y exportación de versión aprobada |
| Docker | Implementado | `Dockerfile` y `requirements-api.txt` |
| API de inferencia | Implementada | `src/api/main.py` con `/health`, `/predict` y `/monitoring/system` |
| Pruebas automatizadas | Implementadas | `tests/test_data.py`, `tests/test_model.py`, `tests/test_api.py` y `tests/test_monitoring.py` |
| Monitoreo y simulación de drift | Implementado | `src/monitoring.py` y `src/simulate_production.py` |
| Estrategia de reentrenamiento (trigger) | Implementado | `src/retrain_trigger.py`, `tests/test_retrain_trigger.py` y `resultado_pipeline/monitoring/retrain_decision.json` |

La tabla refleja lo que ya está disponible en el repositorio, no una lista de funcionalidades pendientes.

## 4. Arquitectura implementada

```text
UCI Adult / CSV local
        |
        v
Ingesta reproducible
        |
        v
Normalización y Data Quality Gates
        |
        v
Eliminación de duplicados + división estratificada train/test
        |
        v
Feature engineering + imputación + escalado + One-Hot Encoding
        |
        v
Validación cruzada de cuatro modelos
        |
        v
Selección por Average Precision + umbral out-of-fold
        |
        v
Evaluación final en test + auditoría por sexo
        |
        +------> Artefactos locales
        |
        +------> MLflow Tracking y Model Registry
```

Las transformaciones forman parte del mismo pipeline que utiliza el modelo. Así, la API puede recibir datos sin transformar y aplicar exactamente la misma preparación usada durante el entrenamiento.

## 5. Estructura del proyecto

```text
.
|-- README.md
|-- requirements.txt
|-- Indicaciones.pdf
|-- Proyecto_Final_Mod6.ipynb
|-- H_EDA_decisiones.ipynb
|-- J_Modelo_y_Experimentacion.ipynb
|-- mlflow.db
|-- mlruns/
|-- src/
|   |-- ingesta.py
|   |-- Data_Quality_Gates.py
|   |-- pipeline_datos.py
|   `-- pipeline_ML.py
`-- resultado_pipeline/
    |-- adult_raw.csv
    |-- adult_clean.csv
    |-- pipeline_features.joblib
    |-- reporte_calidad.csv
    |-- particiones y matrices transformadas
    `-- modelo/
        |-- modelo_clasificacion.joblib
        |-- comparacion_modelos_cv.csv
        |-- metricas_test.json
        |-- metricas_subgrupos_sex.csv
        |-- matriz_confusion.png
        |-- curvas_roc_pr.png
        `-- manifest_modelo.json
```

Las carpetas `resultado_pipeline/` y `mlruns/` contienen resultados generados por el flujo y pueden volver a crearse ejecutando los pipelines.

## 6. Instalación

### Requisitos previos

- Python 3.13.14 (versión utilizada por los notebooks y Docker).
- Acceso a internet únicamente si se descargará nuevamente el dataset.
- Git y un entorno virtual son recomendados.

### Windows PowerShell

Desde la carpeta raíz del proyecto:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name adult-mlops --display-name "Adult MLOps"
```

Si PowerShell bloquea temporalmente la activación del entorno virtual, puede ejecutar los comandos usando directamente `.\.venv\Scripts\python.exe`.

### Verificación rápida

```powershell
python -c "import pandas, sklearn, mlflow; print('Dependencias instaladas correctamente')"
```

## 7. Ingesta de datos

Para descargar el dataset desde UCI y guardarlo como `src/adult_raw.csv`:

```powershell
python src/ingesta.py
```

El pipeline principal también descarga los datos cuando no se indica `raw_path`. Para trabajar sin conexión, se puede reutilizar `resultado_pipeline/adult_raw.csv`.

## 8. Calidad, limpieza y feature engineering

Desde la raíz del repositorio, ejecute:

```powershell
python src/pipeline_datos.py
```

Durante esta etapa se realizan las siguientes tareas:

- Normalización de espacios, `?`, cadenas vacías y otros indicadores de faltantes.
- Unificación y codificación binaria de `income`.
- Validación de esquema, cantidad mínima de filas, target, rangos y consistencia educativa.
- Eliminación de duplicados antes de dividir los datos.
- División train/test estratificada con semilla 42.
- Indicadores de ausencia para `workclass`, `occupation` y `native-country`.
- Transformaciones `log1p` para variables con fuerte asimetría.
- Indicadores de presencia de ganancia y pérdida de capital.
- Eliminación de `education`, porque duplica la información de `education-num`.
- Imputación, escalado y codificación One-Hot dentro de un pipeline reutilizable.

Los resultados se guardan en `resultado_pipeline/`. El archivo `manifest.json` resume la configuración utilizada, las dimensiones de los datos y los artefactos generados.

## 9. EDA

El análisis exploratorio está en `H_EDA_decisiones.ipynb` y se puede abrir desde la raíz del proyecto:

```powershell
jupyter lab H_EDA_decisiones.ipynb
```

El notebook revisa duplicados, datos faltantes, desbalance de clases, variables redundantes, valores extremos y categorías poco frecuentes. Las decisiones tomadas durante este análisis se trasladaron a `src/pipeline_datos.py`.

## 10. Entrenamiento y experimentación

### Desde el notebook

```powershell
jupyter lab J_Modelo_y_Experimentacion.ipynb
```

Las celdas deben ejecutarse en orden. No hace falta abrir antes `Proyecto_Final_Mod6.ipynb`, porque el entrenamiento prepara los datos internamente.

### Desde la terminal

Con MLflow habilitado:

```powershell
python src/pipeline_ML.py --raw-path resultado_pipeline/adult_raw.csv
```

Para una ejecución local sin registrar nuevos runs:

```powershell
python src/pipeline_ML.py --raw-path resultado_pipeline/adult_raw.csv --no-mlflow
```

El entrenamiento compara:

1. `DummyClassifier` como baseline.
2. Regresión logística balanceada.
3. Random Forest balanceado.
4. HistGradientBoosting balanceado.

La selección usa cinco folds estratificados y **Average Precision** como métrica principal. El umbral se selecciona con predicciones out-of-fold de entrenamiento y el test se evalúa una sola vez al final.

## 11. MLflow

### Evidencias de J y K

Ejecute `python -m src.pipeline_ML --raw-path resultado_pipeline/adult_raw.csv`
desde la raíz del proyecto. La ejecución crea cuatro runs candidatos y un run
final; conserva las ejecuciones anteriores. Se requiere MLflow instalado.

**J — Experiment Tracking.** Cada candidato registra `algorithm`,
`hyperparameters`, `feature_set`, `random_seed`, `data_version`, métricas promedio
de cinco folds estratificados y su pipeline entrenado. En `evaluation/` conserva
su configuración, predicciones out-of-fold, matriz de confusión y curvas ROC/PR.
La matriz agrupa predicciones OOF a umbral 0.5; las métricas del run son medias por
fold, por lo que no son necesariamente idénticas a las calculadas sobre la matriz
agrupada. El run final registra la evaluación en test al umbral elegido en train.
Todos los runs incluyen snapshots de datos y particiones (`data/`), hashes,
nombres de features y versiones del entorno (`traceability/`) y código fuente.
Los modelos de MLflow 3 se muestran también como modelos asociados al run;
se comprueba su carga y sus probabilidades después de guardarlos.

**K — Model Registry.** `src/model_registry.py` registra el ganador y conserva
en `registry/registry_lifecycle.json` el historial con fecha de cada transición:
`Experiment → Candidate → Validation → Production`. Los alias de MLflow
`candidate`, `validation` y `production` señalan las versiones que alcanzaron
esas etapas. Pueden apuntar a una misma versión; el historial es la evidencia del
orden de las decisiones. Se usan alias y etiquetas en lugar de los antiguos stages.

La selección maximiza la media de la métrica primaria en CV (Average Precision
por defecto); un empate se resuelve por nombre de algoritmo. La política
académica `academic-v1`, configurable en `MLPipelineConfig.promotion_policy`, exige:

| Criterio de aprobación | Límite |
|---|---|
| Average Precision promedio en CV | ≥ 0.80 |
| F1 promedio en CV a umbral 0.5 | ≥ 0.65 |
| Mejora de Average Precision sobre Dummy | ≥ 0.10 |
| Desviación estándar de Average Precision en CV | ≤ 0.03 |
| Modelo descargado desde Registry | Carga y reproduce las probabilidades |
| Exportación para la API | Carga y reproduce las probabilidades |

Estos límites son una decisión académica explícita incorporada al proyecto,
no límites prescritos por el profesor ni una validación comercial independiente.
El test no interviene en los gates. Si una métrica falta, no es finita o incumple
su límite, la versión queda en Validation con rechazo documentado; no se cambia
la versión de producción anterior.

La versión aprobada se descarga desde `models:/adult-income-classifier/<versión>`
y se exporta a `resultado_pipeline/modelo/production/version_<versión>/`.
`production/current.json` identifica la exportación vigente. La API y Docker usan
esa exportación, su umbral y su versión de Registry; verifican el hash del modelo.
El alias Production representa aprobación local para servir, no un despliegue
automático en un servidor externo. Después de un nuevo entrenamiento, reconstruya
la imagen Docker para incluir la exportación actual.

Abra `MLflow_Tracking_y_Model_Registry.ipynb` para mostrar los runs, artefactos,
criterios, historial y alias. Referencia: [alias y flujos de Model Registry](https://mlflow.org/docs/latest/ml/model-registry/workflow).

El tracking utiliza SQLite mediante `sqlite:///mlflow.db`. Para abrir la interfaz:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Después visite [http://127.0.0.1:5000](http://127.0.0.1:5000).

### Conservación y recuperación de artefactos locales

Inicie MLflow desde la raíz de este repositorio para usar la base correcta:
`mlflow.db` (no `miflow.db`). Los metadatos y los archivos se guardan por separado.
Respalde juntos `mlflow.db`, `mlartifacts/`, `mlruns/` y `resultado_pipeline/`,
con el entrenamiento y MLflow detenidos. `mlartifacts/` está excluida de Git;
una copia del repositorio hecha solamente con Git no conserva esa carpeta.
Al mover el proyecto, las rutas absolutas de los runs históricos también
necesitan revisión. Reiniciar la computadora no debería eliminar los archivos.

Si faltan los artefactos del último run seleccionado pero se conservan sus
resultados locales, ejecute `python -m src.recuperar_artefactos_locales`.
El recuperador comprueba el manifiesto y las métricas contra la base de datos,
restaura la evaluación sin sobrescribir archivos distintos y verifica las copias
con SHA-256. No reconstruye los modelos del Registry ni los runs candidatos:
para ellos se necesita el respaldo original o una nueva ejecución del entrenamiento.
El entrenamiento verifica ahora las copias locales de gráficos, configuración
y código inmediatamente después de registrarlas; esta comprobación no sustituye
un respaldo ante eliminaciones posteriores.

El experimento se llama `adult_income_classification` y el modelo seleccionado se registra como `adult-income-classifier`. Los runs guardan:

- Algoritmo e hiperparámetros.
- Feature set y semilla aleatoria.
- Hash SHA-256 de la versión de datos.
- Métricas de validación y prueba.
- Configuración, predicciones, matriz de confusión y curvas.
- Pipeline entrenado para inferencia.

El pipeline usa serialización `cloudpickle` porque contiene el transformador personalizado `AdultFeatureEngineer`. Solo deben cargarse modelos producidos por este proyecto y provenientes de una fuente confiable.

## 12. Resultados actuales

En la última ejecución completa disponible, HistGradientBoosting fue seleccionado por validación cruzada.

| Métrica de test | Resultado |
|---|---:|
| Precision | 0,6848 |
| Recall | 0,7868 |
| F1 | 0,7323 |
| ROC AUC | 0,9332 |
| Average Precision | 0,8407 |
| Balanced Accuracy | 0,8364 |
| Umbral seleccionado | 0,6450 |

Average Precision promedio en validación cruzada:

| Modelo | Average Precision CV |
|---|---:|
| HistGradientBoosting | 0,8264 |
| Random Forest | 0,8006 |
| Regresión logística | 0,7713 |
| Dummy baseline | 0,2394 |

Estos valores pertenecen a los artefactos actuales y pueden cambiar si se modifican datos, dependencias, configuración o código. La fuente auditable es `resultado_pipeline/modelo/manifest_modelo.json`.

## 13. Auditoría por subgrupos

La evaluación separa resultados por `sex` y reporta precision, recall, F1, tasa de falsos positivos y Average Precision. En los artefactos actuales se observan diferencias entre grupos, por lo que este componente debe conservarse y ampliarse antes de cualquier uso real.

El reporte completo se encuentra en `resultado_pipeline/modelo/metricas_subgrupos_sex.csv`.

## 14. Artefactos principales

- `resultado_pipeline/pipeline_features.joblib`: transformaciones ajustadas.
- `resultado_pipeline/reporte_calidad.csv`: resultado de los gates.
- `resultado_pipeline/manifest.json`: trazabilidad del pipeline de datos.
- `resultado_pipeline/modelo/modelo_clasificacion.joblib`: pipeline completo entrenado.
- `resultado_pipeline/modelo/comparacion_modelos_cv.csv`: comparación de candidatos.
- `resultado_pipeline/modelo/metricas_test.json`: evaluación final.
- `resultado_pipeline/modelo/predicciones_test.csv`: probabilidades y predicciones.
- `resultado_pipeline/modelo/manifest_modelo.json`: configuración, versión de datos y artefactos.

## 15. Docker

La imagen de Docker reúne la API, el modelo aprobado y todas las dependencias necesarias para hacer predicciones. Antes de construirla debe existir una versión del modelo en `resultado_pipeline/modelo/production/`.

Para crear la imagen a partir del `Dockerfile`:

```powershell
docker build -t adult-income-api .
docker run --rm -p 8011:8010 adult-income-api
```

El primer comando construye la imagen; el segundo inicia el contenedor. Cuando esté funcionando, la documentación estará en [http://127.0.0.1:8011/docs](http://127.0.0.1:8011/docs).

El mapeo `8011:8010` significa que la aplicación escucha en el puerto 8010 dentro del contenedor, pero se abre desde el puerto 8011 del equipo.


## 16. API de inferencia

La demo integra la interfaz y la API en el puerto **8010**. Para iniciarla:

```powershell
python -m uvicorn demo.app:app --host 127.0.0.1 --port 8010
```

Desde ese servidor se puede acceder a [la demo](http://127.0.0.1:8010), [Swagger](http://127.0.0.1:8010/docs), [el estado del modelo](http://127.0.0.1:8010/health) y [el monitoreo](http://127.0.0.1:8010/monitoring/system).

La demo, la API independiente y el contenedor son formas distintas de ejecutar el mismo servicio. No es necesario iniciar las tres al mismo tiempo.

La aplicación está en `src/api/main.py`. Carga una sola vez el pipeline completo y expone `GET /health`, `POST /predict` y `GET /monitoring/system`. Para ejecutarla localmente:

```powershell
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8010
```

La entrada valida campos obligatorios, tipos, campos adicionales, cadenas vacías y rangos de negocio. La respuesta contiene:

```json
{
  "prediction": 1,
  "probability": 0.873,
  "model_version": "1"
}
```

La API lee `resultado_pipeline/modelo/production/current.json` y carga el modelo
de la carpeta `version_<versión>` indicada allí. Esa copia se exportó desde Registry
después de la aprobación; conserva el pipeline completo de transformaciones.

## 17. Testing

La suite se divide por responsabilidad:

- `tests/test_data.py`: esquema, tipos, rangos, valores no finitos, target y duplicados.
- `tests/test_model.py`: carga del modelo, predicción binaria, probabilidades y determinismo.
- `tests/test_api.py`: health, predicción y rechazo HTTP 422 de entradas incorrectas.
- `tests/test_monitoring.py`: PSI, drift, métricas del modelo y métricas operativas de la API.
- `tests/test_retrain_trigger.py`: lógica de decisión de reentrenamiento (PSI + degradación de desempeño + volumen mínimo).

Para instalar y ejecutar las pruebas desde un entorno virtual activado:

```powershell
pip install -r requirements-dev.txt
pytest tests/ -v
```

Si falta `adult_clean.csv`, ejecute primero el pipeline de datos. Si falta `modelo_clasificacion.joblib`, ejecute `J_Modelo_y_Experimentacion.ipynb`.

## 18. Monitoreo y drift

El monitoreo implementado distingue:

- Monitoreo del sistema: latencia, throughput, errores y disponibilidad.
- Monitoreo de datos: PSI, Kolmogorov-Smirnov, distancia Wasserstein y categorías desconocidas.
- Monitoreo del modelo: evolución de precision, recall, F1 y AUC cuando exista ground truth.
- Simulación de drift en varios lotes de producción.

Instalación y pruebas específicas:

```powershell
pip install -r requirements-monitoring.txt
pytest tests/test_monitoring.py -v
```

Generación reproducible de seis lotes de 1.000 registros:

```powershell
python -m src.simulate_production `
    --batches 6 `
    --batch-size 1000 `
    --random-state 42
```

Generación del reporte general:

```powershell
python -m src.monitoring `
    --reference resultado_pipeline/adult_clean.csv `
    --production resultado_pipeline/monitoring/production_batch.csv `
    --output resultado_pipeline/monitoring/monitoring_report.json
```

Con la API en ejecución, las métricas operativas se consultan así:

```powershell
Invoke-RestMethod `
    -Uri "http://127.0.0.1:8010/monitoring/system" `
    -Method Get
```

El reentrenamiento no debe activarse únicamente porque cambie una distribución. Debe combinar evidencia de drift, degradación de desempeño, suficiente información nueva y aprobación operativa (Data Drift ≠ Model Degradation).

### 18.1 Lógica de reentrenamiento (`src/retrain_trigger.py`)

Esta regla queda implementada y probada en `src/retrain_trigger.py`. Para cada lote evalúa tres condiciones y solo recomienda reentrenar si las tres se cumplen a la vez:

- **Drift**: el PSI máximo entre las features numéricas y categóricas supera `psi_threshold = 0.25` (mismo corte que `monitoring._psi_level` usa para clasificar "drift_alto").
- **Degradación de desempeño**: la métrica `f1` (con ground truth disponible) cae por debajo de `minimum_performance = 0.60`.
- **Volumen mínimo**: el lote tiene al menos `minimum_production_rows = 500` filas (la mitad del tamaño de lote usado por `simulate_production.py`).

La recomendación final siempre incluye `requires_manual_approval: true`: el módulo nunca dispara un reentrenamiento automático, solo lo señala para aprobación operativa.

Ejecución sobre los seis lotes reales generados por `simulate_production.py`:

```powershell
python -m src.retrain_trigger `
    --batches-dir resultado_pipeline/monitoring `
    --output resultado_pipeline/monitoring/retrain_decision.json
```

Resultado obtenido con los lotes actuales (`resultado_pipeline/monitoring/retrain_decision.json`): los lotes 1 y 2 (drift_strength 0.0 y 0.2) quedan en `NO_RETRAIN_NEEDED`, y a partir del lote 3 (drift_strength ≥ 0.4, donde el PSI de `hours-per-week`/`age` supera el umbral y el F1 cae por debajo de 0.60) la recomendación pasa a `RETRAIN_RECOMMENDED`.

Pruebas específicas:

```powershell
pytest tests/test_retrain_trigger.py -v
```

## 19. Reproducibilidad y trazabilidad

El proyecto permite reproducir el flujo completo desde la descarga de los datos
hasta el entrenamiento, monitoreo y despliegue del modelo. El repositorio oficial
se encuentra en [GitHub](https://github.com/sebacalfa/ProyFinal_Mod6.git).

Todos los comandos siguientes deben ejecutarse desde la carpeta raíz del
repositorio.

### 19.1 Clonar el repositorio

```powershell
git clone https://github.com/sebacalfa/ProyFinal_Mod6.git
cd ProyFinal_Mod6
```

### 19.2 Crear y activar el entorno virtual

Se recomienda Python 3.11, 3.12 o 3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 19.3 Instalar las dependencias

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

`requirements.txt` contiene las dependencias de ingesta, procesamiento,
entrenamiento, visualización y MLflow. `requirements-dev.txt` agrega las
dependencias de la API, monitoreo y pruebas automatizadas.

### 19.4 Ejecutar la ingesta

```powershell
python src/ingesta.py
```

Este comando descarga el dataset Adult Income desde UCI y genera
`src/adult_raw.csv`. La descarga requiere conexión a internet.

### 19.5 Entrenar y evaluar los modelos

Con registro de experimentos en MLflow:

```powershell
python src/pipeline_ML.py --raw-path src/adult_raw.csv
```

Para ejecutar el entrenamiento sin registrar nuevos experimentos:

```powershell
python src/pipeline_ML.py --raw-path src/adult_raw.csv --no-mlflow
```

El pipeline prepara los datos, realiza feature engineering, compara modelos,
selecciona el umbral de clasificación y genera los artefactos en
`resultado_pipeline/` y `resultado_pipeline/modelo/`.

### 19.6 Ejecutar las pruebas automatizadas

Suite completa:

```powershell
python -m pytest tests/ -v
```

Pruebas por componente:

```powershell
python -m pytest tests/test_data.py -v
python -m pytest tests/test_model.py -v
python -m pytest tests/test_api.py -v
python -m pytest tests/test_monitoring.py -v
python -m pytest tests/test_retrain_trigger.py -v
```

### 19.7 Simular producción y detectar drift

```powershell
python -m src.simulate_production --batches 6 --batch-size 1000 --random-state 42
```

La simulación compara la distribución de referencia contra seis lotes de
producción y calcula PSI, Kolmogorov-Smirnov y distancia Wasserstein. Los
resultados se guardan en `resultado_pipeline/monitoring/`.

### 19.8 Simular problemas de calidad

```powershell
python -m src.quality_simulation --reference resultado_pipeline/adult_clean.csv --batch resultado_pipeline/monitoring/production_batch_1.csv --output resultado_pipeline/monitoring/quality_incident.json
```

La prueba contamina únicamente una copia en memoria con missing values, una
fila duplicada, un outlier extremo, un datatype incorrecto, una categoría
desconocida y una modificación de esquema. El sistema detecta los seis
problemas, bloquea el batch y registra el incidente sin modificar ni guardar la
copia contaminada. El resultado queda en
`resultado_pipeline/monitoring/quality_incident.json`.

### 19.9 Evaluar la estrategia de reentrenamiento

```powershell
python -m src.retrain_trigger --batches-dir resultado_pipeline/monitoring --output resultado_pipeline/monitoring/retrain_decision.json
```

La decisión combina drift, degradación del desempeño y volumen mínimo. El
resultado es una recomendación que siempre requiere aprobación manual; el
proyecto no reentrena automáticamente solo porque cambie una distribución.

### 19.10 Iniciar MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

La interfaz queda disponible en
[http://127.0.0.1:5000](http://127.0.0.1:5000).

### 19.11 Construir la imagen Docker

El entrenamiento debe ejecutarse antes de construir la imagen, porque el
`Dockerfile` incorpora el modelo y su manifiesto.

```powershell
docker build -t grupo2-mlops .
```

### 19.12 Ejecutar la API con Docker

```powershell
docker run --rm -p 8011:8010 grupo2-mlops
```

Servicios disponibles:

- API: [http://127.0.0.1:8011](http://127.0.0.1:8011).
- Documentación Swagger: [http://127.0.0.1:8011/docs](http://127.0.0.1:8011/docs).
- Estado del modelo: [http://127.0.0.1:8011/health](http://127.0.0.1:8011/health).
- Monitoreo del sistema: [http://127.0.0.1:8011/monitoring/system](http://127.0.0.1:8011/monitoring/system).

El mapeo `8011:8010` publica la API Docker en 8011 y conserva 8010 dentro del contenedor. La demo local sigue en 8010.

### 19.13 Configuración reproducible

La configuración central utiliza:

- `random_state=42`.
- División de prueba de 20 %.
- Cinco folds estratificados.
- Frecuencia mínima de categorías igual a 100.
- Versionamiento SHA-256 del dataset limpio y del código.
- Manifiestos de datos y modelo dentro de `resultado_pipeline/`.
- Registro de parámetros, métricas y artefactos mediante MLflow.

Los entornos virtuales, cachés y archivos temporales no deben versionarse. Para
trabajar sin conexión después de la primera ejecución puede reutilizarse el CSV
guardado localmente.

## 20. Flujo Git recomendado

El proyecto se encuentra versionado en el
[repositorio oficial](https://github.com/sebacalfa/ProyFinal_Mod6.git). Para
mantener un historial ordenado se recomienda trabajar mediante ramas, por
ejemplo:

```text
main
`-- develop
    |-- feature/data-ingestion
    |-- feature/data-quality
    |-- feature/model
    |-- feature/api
    `-- feature/monitoring
```

No deben versionarse entornos virtuales, cachés de Python ni archivos temporales. Los datasets grandes deben obtenerse mediante el script de ingesta o almacenamiento externo, de acuerdo con las indicaciones del proyecto.

## 21. Equipo

- Cynthia Montero Sancho.
- Sebastian Calvo.



## 22. Referencias

- Becker, B. y Kohavi, R. (1996). Adult. UCI Machine Learning Repository.
- [Documentación de scikit-learn](https://scikit-learn.org/stable/).
- [Documentación de MLflow](https://mlflow.org/docs/latest/).
