# Auditoría inicial — Fase 1

**Ejecutada:** 2026-10-25. Resultado estructurado: `datasets/reports/phase1_audit.json`.

## Proyecto y entorno

- La estructura Fase 0 se conservó; no se reconstruyó ni eliminó contenido.
- Python 3.11.6; PyTorch 2.14.1+cpu (`cuda_available=false`); Ultralytics 8.4.189; OpenCV 5.0.0; Pillow 12.2.0; PyYAML 6.0.3; psutil 7.2.2.
- No se instaló CUDA ni software NVIDIA. No faltaban dependencias.
- Peso disponible: `training/checkpoints/yolo26n.pt`, 5.544.453 bytes, SHA-256 verificado en el manifiesto Fase 0.
- No existe ningún peso personalizado `.pt`, `.onnx` o `.engine`; por tanto no hay comparación honesta con modelo custom.

## Datos

| Conjunto | Origen | Imágenes | Uso |
|---|---|---:|---|
| visionar_pilot_v1 train/val/test | COCO128, derivado de COCO train2017 | 15 / 4 / 2 | Regresión de tubería Fase 0; no evaluación independiente |
| phase1_coco_val test | COCO val2017 oficial | 100 | Evaluación Fase 1 de cinco clases |

El test Fase 1 contiene 397 cajas: car 82, bottle 82, laptop 47, cell phone 36 y book 150. Hay al menos 25 imágenes por clase porque algunas imágenes seleccionadas para otra clase también contienen la categoría.

## Integridad y fuga

- 121 imágenes verificadas y 0 corruptas.
- 0 etiquetas ausentes, clases inesperadas o cajas YOLO inválidas.
- IDs y nombres coinciden con la ontología COCO de `yolo26n.pt`.
- 0 duplicados SHA-256 entre splits/conjuntos.
- 0 pares casi duplicados entre conjuntos con aHash/Hamming ≤4.
- 100/100 imágenes Fase 1 tienen URL, hash, nombre y URL de licencia en `datasets/manifests/phase1_coco_val.csv`.

Las licencias de las fotografías varían: 18 Attribution, 16 BY-SA, 13 BY-NC, 26 BY-NC-SA, 20 BY-NC-ND y 7 BY-ND. El uso actual es evaluación local. Este subconjunto no queda aprobado para redistribución, modificación pública o uso comercial en bloque; debe filtrarse por licencia para esos usos.

