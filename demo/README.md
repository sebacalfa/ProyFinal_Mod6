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
- Monitoreo de drift a partir del último reporte generado.
- Vista completa de la arquitectura MLOps.

La demo no modifica ni duplica el modelo. La capa visual vive en `demo/assets/` y el pequeño servidor de integración está en `demo/app.py`.
