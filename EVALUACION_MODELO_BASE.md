# Evaluación del modelo base

**Fecha:** 2026-10-25. **Modelo:** YOLO26n, SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`. **Entorno:** Intel Core i5-10210U, 8 hilos lógicos, 11,83 GB RAM, Windows, Python 3.11.6, PyTorch CPU, Ultralytics 8.4.189. Configuración: 640 px, batch 1, workers 0, clases `[1,39,56,63,67]`.

## Resultados obtenidos

| Evaluación | Imágenes / cajas | Precision | Recall | mAP50 | mAP50-95 | Inferencia media |
|---|---:|---:|---:|---:|---:|---:|
| Test retenido | 2 / 3 | 0,751 | 0,750 | 0,745 | 0,646 | 133,4 ms/imagen |
| Diagnóstico en las 21 | 21 / 70 | 0,551 | 0,491 | 0,518 | 0,370 | 122,3 ms/imagen |

La inferencia PyTorch medida equivale aproximadamente a 8,2 FPS solo para la etapa de inferencia (`1000/122,3`); el pipeline completo será más lento. En el diagnóstico, mAP50-95 por clase fue: bicycle 0,318; bottle 0,348; chair 0,266; laptop 0,780; cell phone 0,138.

## Interpretación válida

El test de dos imágenes solo contiene bicycle y cell phone, por lo que los otros mAP por clase son `null`. El diagnóstico usa todos los splits únicamente porque el modelo no fue ajustado con este piloto; sirve para descubrir debilidades, no es un test independiente. Las imágenes proceden de COCO train2017, dominio relacionado con el preentrenamiento. No se afirma precisión de producción.

Cell phone, chair, bicycle y bottle requieren más evaluación; el bajo resultado diagnóstico de cell phone (0,138) es una señal para recopilar ejemplos, no evidencia suficiente para fine-tuning. Laptop tiene solo 3 cajas, por lo que 0,780 es inestable.

## Decisión

Conservar `yolo26n.pt` como baseline del MVP y no entrenar todavía. Primero crear un test de campo propio y balanceado. Proponer fine-tuning solo si allí alguna clase incumple umbrales acordados (propuesta inicial: recall ≥0,75, mAP50-95 ≥0,50 y latencia compatible con la experiencia). La matriz de confusión y curvas están en `training/metrics/runs/`; los valores estructurados, en `training/metrics/baseline_metrics.json`.

