# Demo final · Adult Income

Desde la raíz del proyecto, con el entorno de Python activo:

```powershell
python -m uvicorn demo.app:app --host 127.0.0.1 --port 8010
```

Abra http://127.0.0.1:8010. Si ya está ejecutándose, no inicie otra instancia en el mismo puerto.

## Recorrido único

Use Siguiente y Anterior o el menú lateral: Datos originales → Validación → Entrenamiento → Experimentos → Registro → Docker → Predicción → Monitoreo → Alertas.

Cada evidencia se presenta una sola vez. Los detalles técnicos permanecen plegados. El guion de exposición está en GUION_EXPOSICION_15_MINUTOS.md; reserve 13:30 para el recorrido y hasta 1:30 de margen.

Experimentos comprueba los cinco runs en MLflow y sus artefactos locales. Registro muestra los criterios reales y compara la versión aprobada con la que carga la API. No vuelva a entrenar solamente para abrir la demo.

## Una sola API

Swagger en http://127.0.0.1:8010/docs muestra únicamente GET /health, POST /predict y GET /monitoring/system. Las rutas internas de la interfaz siguen funcionando, pero no aparecen duplicadas en la documentación.

## Caso o lote no visto

En Predicción se pueden editar las 14 variables o cargar/pegar un CSV de hasta 1 MB y entre 1 y 1000 filas. Use los nombres de columna del ejemplo. income es opcional y acepta 0, 1, <=50K o >50K; sin resultados reales no se calcula rendimiento supervisado.

El lote se valida con el contrato de entrada de la API. Un error de columnas, tipo o rango lo bloquea. Los lotes válidos generan predicciones y comparación con la referencia; Monitoreo y Alertas muestran el resultado de ese mismo envío. Una muestra menor de 500 filas se marca como orientativa para drift.

La entrada y el reporte se conservan en resultado_pipeline/demo_batches/<identificador>/. Esos archivos son locales y no se incluyen en Git. La pantalla conserva el último lote mientras permanezca abierta; recargarla reinicia esa selección visual, pero no borra los archivos.

Las métricas operativas pertenecen al proceso actual y se reinician al detenerlo. La simulación de calidad anterior se identifica explícitamente como ejemplo guardado.

## Docker durante la exposición

El Dockerfile empaqueta la API. La demo permanece en 8010 y Docker se publica en 8011: ambos pueden funcionar al mismo tiempo. Construya la imagen antes de exponer. Muestre docker ps y consulte /health en 8011 para comprobar el contenedor. Swagger de Docker: http://127.0.0.1:8011/docs.

```powershell
docker build -t grupo2-mlops .
docker run --rm --name adult-income-api -p 8011:8010 grupo2-mlops
# En otra terminal:
docker ps
Invoke-RestMethod http://127.0.0.1:8011/health
```
