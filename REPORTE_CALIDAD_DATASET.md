# Reporte de calidad del dataset piloto

**Ejecución:** 2026-10-25. Fuente: COCO128. Script: `scripts/audit_dataset.py`.

## Resultado obtenido

- 21 imágenes seleccionadas; división determinista 15 train / 4 val / 2 test (71,4 / 19,0 / 9,5 %).
- 70 cajas: chair 35, bottle 18, cell phone 8, bicycle 6, laptop 3.
- 21 imágenes abiertas y verificadas por Pillow.
- 0 archivos corruptos; 0 etiquetas ausentes; 0 cajas fuera de `[0,1]`; 0 clases inesperadas.
- 0 grupos de duplicados SHA-256. El constructor agrupó antes de dividir coincidencias exactas o aHash con distancia Hamming ≤4; no detectó colisiones en las imágenes seleccionadas.
- Resoluciones observadas: de 312×460 a 640×640-equivalente variable; ninguna dimensión menor de 64 px.
- Ningún archivo fue eliminado o enviado automáticamente a cuarentena.

El JSON completo y la lista auditable de cuarentena están en `datasets/reports/quality_report.json` y `datasets/reports/quarantine_candidates.csv`.

## Limitaciones

COCO128 no conserva en el paquete la metadata completa de licencia por imagen. El piloto es muy pequeño, heredado de COCO train2017 y no representa cámara móvil/AR ni las condiciones de prueba pedidas. La proximidad al dominio de preentrenamiento hace que la evaluación sea diagnóstica. `laptop` tiene solo 3 cajas y el test retenido contiene únicamente bicycle y cell phone.

## Recomendación

Mantener el piloto para regresión de tubería. Antes de fine-tuning, crear un conjunto propio de prueba con al menos 50–100 imágenes por clase y escenarios controlados (luz natural/artificial, fondos complejos, oclusión, distancia, ángulo y múltiples objetos), agrupando por sesión para evitar fuga.

