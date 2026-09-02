# Demo visual · Adult Income MLOps

Esta carpeta contiene una interfaz de presentación separada del código principal. Usa los artefactos reales de `resultado_pipeline/` y carga el mismo modelo utilizado por `src/api/main.py`.

## Ejecutar

Desde la raíz del repositorio, con el entorno virtual activo:

```powershell
python -m uvicorn demo.app:app --host 127.0.0.1 --port 8000
```

Luego abre `http://127.0.0.1:8000`.

## Contenido

- Resumen ejecutivo y métricas reales del test.
- Formulario funcional de predicción con las 14 variables del API.
- Comparación de modelos y auditoría por sexo.
- Vista completa del modelo: objetivo, diseño del experimento, comparación de
  candidatos, métricas finales, umbral, hiperparámetros, trazabilidad de
  MLflow, ciclo del Model Registry y análisis por subgrupos.
- Monitoreo de drift a partir del último reporte generado.
- Vista completa de la arquitectura MLOps.
- Guion visual de la demo final: Raw Data → Validation → Training → MLflow
  Experiment → Model Registry → Docker → API → Monitoring → Drift/Quality.
- Monitoreo diferenciado de sistema, datos y modelo, con seis batches.
- Tabla de PSI, KS, p-value y Wasserstein por variable; Precision, Recall, F1
  y ROC AUC del modelo; comparación de PSI/F1 por lote.
- Evidencia visible de contaminación, bloqueo, registro y decisión de
  reentrenamiento con aprobación humana.
- Detalle de los seis problemas de calidad y de las condiciones de drift,
  desempeño y volumen usadas para recomendar reentrenamiento.
- Cada paso del recorrido abre su propia ficha, con evidencia, comando,
  resultado esperado y enlace local correspondiente; no reutiliza una pantalla
  genérica para etapas diferentes.

La demo no modifica ni duplica el modelo. La capa visual vive en `demo/assets/` y el pequeño servidor de integración está en `demo/app.py`.
