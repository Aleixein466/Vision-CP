# Reporte de entrenamiento VisionAR AI

**Fecha:** 25 de octubre de 2026.

## Entrenamiento ejecutado

Se completó una prueba real de entrenamiento de segmentación en CPU para validar de extremo a extremo la instalación de PyTorch, Ultralytics, carga de datos, retropropagación, validación y escritura de checkpoints.

- Dataset: COCO8-seg, ocho imágenes reales etiquetadas; 4 train y 4 val.
- Modelo base: `yolo26n-seg.pt`.
- Épocas: 1.
- Resolución: 320 px.
- Lote: 1.
- Dispositivo: CPU.
- Workers: 0.
- Resultado: `best.pt` y `last.pt`, 6.640.285 bytes cada uno.

Métricas del conjunto de validación de cuatro imágenes:

- Cajas: precision 0,895; recall 0,450; mAP50 0,75827; mAP50-95 0,53972.
- Máscaras: precision 0,680; recall 0,435; mAP50 0,64620; mAP50-95 0,45256.

Estas cifras solo prueban que la tubería funciona; el conjunto es demasiado pequeño para representar el rendimiento general. Los pesos no se desplegaron en VisionAR.

El checkpoint `best.pt` se recargó y ejecutó sobre una imagen real de validación: produjo dos detecciones, dos máscaras y las clases `umbrella` y `person`.

## Fine-tuning propio pendiente

El auditor se ejecutó nuevamente sobre `VisionAR Own Dataset v1`: 0 imágenes y 0 anotaciones en las 38 clases. Por tanto, el entrenamiento personalizado sigue bloqueado. No hay ejemplos desde los que calcular gradientes para billetera, USB, perfume, tarjetas y demás clases particulares.

Requisito configurado: al menos 30 anotaciones revisadas por clase, separadas por sesión entre train, val y test. Recomendación para las diez clases sin cobertura nativa: 100 a 200 imágenes variadas por clase.

## Estabilidad durante operación

Los registros mostraron que Qwen2-VL dejaba menos de 1,3 GB disponibles y, durante sesiones largas con YOLO Small, OpenCV y PyTorch no podían reservar bloques menores de 1 MB. El inicio normal usa ahora SmolVLM; Qwen2-VL queda disponible con `npm run dev:qwen` para consultas puntuales.
