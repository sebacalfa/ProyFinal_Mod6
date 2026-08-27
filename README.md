# Adult Income - Proyecto MLOps End-to-End

Proyecto de clasificación para predecir si una persona adulta genera ingresos anuales superiores a USD 50 000 utilizando el dataset **Adult / Census Income** de UCI. El repositorio cubre actualmente la ingesta reproducible, validación de calidad, EDA, feature engineering, comparación de modelos, evaluación y seguimiento de experimentos con MLflow.

## 1. Problema de negocio

El objetivo es estimar la probabilidad de que una persona pertenezca a la clase `>50K` a partir de variables demográficas, educativas y laborales. Además del desempeño predictivo, el proyecto revisa diferencias de comportamiento del modelo entre los subgrupos `Female` y `Male`.

La variable objetivo se codifica de la siguiente forma:

- `0`: ingreso `<=50K`.
- `1`: ingreso `>50K`.

El resultado debe interpretarse como una herramienta analítica. No debe utilizarse automáticamente para decisiones sensibles sobre personas sin una evaluación adicional de equidad, legalidad y riesgo.

## 2. Dataset

- Fuente: [UCI Machine Learning Repository - Adult](https://archive.ics.uci.edu/dataset/2/adult).
- Identificador utilizado por `ucimlrepo`: `2`.
- Tipo de problema: clasificación binaria.
- Dimensión raw observada: 48 842 filas y 15 columnas.
- Dimensión después de eliminar duplicados: 48 790 filas y 15 columnas.
- Variable objetivo: `income`.
- Proporción observada de la clase positiva: aproximadamente 23,94 %.

El script de ingesta descarga los datos desde UCI, por lo que no es obligatorio depender de un CSV preparado manualmente.

## 3. Estado actual del proyecto

| Componente | Estado | Evidencia principal |
|---|---:|---|
| Ingesta reproducible | Implementado | `src/ingesta.py` y `src/pipeline_datos.py` |
| Data Quality Gates | Implementado | `src/pipeline_datos.py` y reporte CSV |
| EDA orientado a decisiones | Implementado | `H_EDA_decisiones.ipynb` |
| Feature engineering reutilizable | Implementado | `AdultFeatureEngineer` y pipeline de scikit-learn |
| Modelado y experimentación | Implementado | `src/pipeline_ML.py` y notebook J |
| MLflow Tracking | Implementado | `mlflow.db`, `mlruns/` y registro desde el pipeline |
| Model Registry | Implementado en el entrenamiento | Nombre registrado: `adult-income-classifier` |
| Docker | Pendiente | No existe `Dockerfile` actualmente |
| API de inferencia | Pendiente | No existe aplicación FastAPI actualmente |
| Pruebas automatizadas | Pendiente | No existe carpeta `tests/` actualmente |
| Monitoreo y simulación de drift | Pendiente | No existe módulo de monitoreo actualmente |

Esta tabla describe los archivos presentes en el proyecto al momento de elaborar este README; evita confundir entregables planeados con componentes ya implementados.

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

El pipeline de entrenamiento contiene también las transformaciones, por lo que el mismo objeto puede recibir datos sin transformar durante inferencia. Esto reduce el riesgo de que el feature engineering del notebook sea diferente al utilizado en producción.

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

Los archivos dentro de `resultado_pipeline/` y `mlruns/` son salidas generadas. Se pueden regenerar ejecutando los pipelines.

## 6. Instalación

### Requisitos previos

- Python 3.11, 3.12 o 3.13.
- Acceso a internet únicamente si se descargará nuevamente el dataset.
- Git y un entorno virtual son recomendados.

### Windows PowerShell

Desde la carpeta raíz del proyecto:

```powershell
py -m venv .venv
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

El pipeline principal también descarga el dataset automáticamente cuando no recibe `raw_path`. Para trabajar sin conexión se puede utilizar el CSV incluido en `resultado_pipeline/adult_raw.csv`.

## 8. Calidad, limpieza y feature engineering

Ejecute el pipeline de datos desde la raíz:

```powershell
python src/pipeline_datos.py
```

El flujo realiza, entre otras, las siguientes operaciones:

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

Las salidas quedan en `resultado_pipeline/`. El archivo `manifest.json` conserva configuración, dimensiones y lista de artefactos.

## 9. EDA

El análisis exploratorio orientado a decisiones está en `H_EDA_decisiones.ipynb`. Debe abrirse desde la raíz del proyecto:

```powershell
jupyter lab H_EDA_decisiones.ipynb
```

El notebook analiza duplicados, faltantes, desbalance, redundancia, asimetría, valores extremos, categorías raras, asociaciones y relaciones no lineales. Sus decisiones se reflejan en `src/pipeline_datos.py`.

## 10. Entrenamiento y experimentación

### Opción recomendada: notebook independiente

```powershell
jupyter lab J_Modelo_y_Experimentacion.ipynb
```

Ejecute las celdas en orden. No es necesario ejecutar antes `Proyecto_Final_Mod6.ipynb`, porque el pipeline ML llama internamente al pipeline de datos.

### Opción por terminal

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

El tracking utiliza SQLite mediante `sqlite:///mlflow.db`. Para abrir la interfaz:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Después visite [http://127.0.0.1:5000](http://127.0.0.1:5000).

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

**Estado: pendiente.** El proyecto todavía no contiene un `Dockerfile`; por lo tanto, no se incluyen comandos de construcción que aparenten ser funcionales. Para completar el entregable será necesario contenerizar la futura API junto con el modelo y sus dependencias.

La validación mínima esperada cuando se implemente será:

```text
docker build -> imagen reproducible
docker run   -> servicio disponible
POST /predict -> predicción válida
```

## 16. API de inferencia

**Estado: pendiente.** Se recomienda implementar FastAPI con un endpoint `POST /predict` que valide las variables de entrada y devuelva, como mínimo:

```json
{
  "prediction": 1,
  "probability": 0.873,
  "model_version": "1"
}
```

La API deberá cargar el pipeline completo, no recrear manualmente las transformaciones.

## 17. Testing

**Estado: pendiente.** La carpeta `tests/` deberá cubrir como mínimo:

- Esquema, tipos, rangos, faltantes y variables obligatorias.
- Predicción válida ante un input válido.
- Rechazo claro de categorías, tipos o esquemas inválidos cuando corresponda.
- Respuesta HTTP 200 y esquema de salida de la futura API.
- Respuesta controlada ante solicitudes inválidas.

## 18. Monitoreo y drift

**Estado: pendiente.** La implementación final debe distinguir:

- Monitoreo del sistema: latencia, throughput, errores y disponibilidad.
- Monitoreo de datos: comparación entre referencia y producción mediante PSI u otra prueba justificada.
- Monitoreo del modelo: evolución de precision, recall, F1 y AUC cuando exista ground truth.
- Simulación de drift en varios lotes de producción.
- Simulación de problemas de calidad sin modificar permanentemente el dataset original.

El reentrenamiento no debe activarse únicamente porque cambie una distribución. Debe combinar evidencia de drift, degradación de desempeño, suficiente información nueva y aprobación operativa.

## 19. Reproducibilidad y trazabilidad

La configuración central utiliza:

- `random_state=42`.
- División test de 20 %.
- Cinco folds estratificados.
- Frecuencia mínima de categorías igual a 100.
- Versionamiento del dataset limpio mediante SHA-256.

Para compartir el trabajo con otro integrante, envíe la carpeta completa o el repositorio. El notebook de modelado depende de `src/pipeline_datos.py`, `src/pipeline_ML.py`, `requirements.txt` y del CSV local cuando se ejecuta sin conexión.

## 20. Flujo Git recomendado

La carpeta revisada no contiene actualmente un repositorio Git detectable. Antes de la entrega se recomienda inicializar o clonar el repositorio oficial y trabajar mediante ramas, por ejemplo:

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

Agregar antes de la entrega:

- Nombre completo - responsabilidad principal.
- Nombre completo - responsabilidad principal.
- Nombre completo - responsabilidad principal.

## 22. Próximos pasos para completar el proyecto

1. Implementar y probar la API de inferencia.
2. Agregar pruebas de datos, modelo y API.
3. Crear un `Dockerfile` funcional y verificarlo en una máquina limpia.
4. Implementar monitoreo, contaminación de batches y simulación de drift.
5. Documentar la estrategia y lógica de reentrenamiento.
6. Añadir el diagrama final de arquitectura.
7. Completar integrantes, URL del repositorio y comandos definitivos de demo.

## 23. Referencias

- Becker, B. y Kohavi, R. (1996). Adult. UCI Machine Learning Repository.
- [Documentación de scikit-learn](https://scikit-learn.org/stable/).
- [Documentación de MLflow](https://mlflow.org/docs/latest/).
