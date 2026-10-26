# Reporte de métricas — Fase 1

**Modelo:** YOLO26n oficial. **Datos:** 100 imágenes reales de COCO val2017, 397 cajas. **Equipo:** Intel Core i5-10210U, 8 procesadores lógicos, 11,83 GB RAM, CPU solamente. **Configuración:** 640 px, batch 1, workers 0.

## Métricas Ultralytics

| Clase | Cajas GT | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| car | 82 | 0,588 | 0,500 | 0,537 | 0,328 |
| bottle | 82 | 0,525 | 0,537 | 0,536 | 0,388 |
| laptop | 47 | 0,805 | 0,638 | 0,683 | 0,563 |
| cell phone | 36 | 0,650 | 0,472 | 0,525 | 0,310 |
| book | 150 | 0,316 | 0,133 | 0,153 | 0,064 |
| **Global** | **397** | **0,577** | **0,456** | **0,487** | **0,331** |

## Conteo operativo fijo

A `confidence=0,25` e IoU de correspondencia `0,50`:

| Clase | TP | FP | FN | Precision | Recall | Confianza media |
|---|---:|---:|---:|---:|---:|---:|
| car | 31 | 8 | 51 | 0,795 | 0,378 | 0,526 |
| bottle | 39 | 24 | 43 | 0,619 | 0,476 | 0,482 |
| laptop | 29 | 4 | 18 | 0,879 | 0,617 | 0,719 |
| cell phone | 12 | 4 | 24 | 0,750 | 0,333 | 0,539 |
| book | 8 | 10 | 142 | 0,444 | 0,053 | 0,384 |

Estas cifras fijas no son iguales a la precision/recall resumida de Ultralytics, que selecciona puntos sobre sus curvas. El registro por detección está en `training/metrics/phase1/errors.csv` y `predictions.csv`.

## Rendimiento CPU

- Latencia end-to-end media: 126,97 ms/imagen; mínimo 50,22 ms; máximo 2.585,83 ms (incluye calentamiento/arranque).
- Rendimiento derivado de la media: 7,88 FPS.
- Ultralytics reportó 95,7 ms de inferencia, 1,2 ms de preproceso y 4,5 ms de postproceso por imagen durante validación.
- RSS al inicio: 229,2 MB; máximo observado: 464,0 MB; incremento: 234,8 MB.
- CPU del proceso normalizada por 8 procesadores lógicos durante el pase: 67,4 %.

## Validación visual y cámara

Se generaron 100 imágenes anotadas reales. Se inspeccionaron ejemplos de las cinco clases: botella 0,90; laptop 0,95; libro 0,61; teléfono 0,91 y automóvil 0,91, además de múltiples automóviles en una escena. La carpeta es `training/metrics/phase1/annotated/`.

El prototipo encontró cámaras 0 y 1. En la ejecución registrada, la cámara 0 procesó cinco frames reales y liberó el dispositivo; el último frame alcanzó 10,46 FPS sin detectar una clase objetivo en el encuadre. Esto valida captura e inferencia, no exactitud de cámara para esas clases.

## Conclusión

`laptop` funciona mejor y es la única clase con mAP50-95 superior a 0,50 en este subconjunto. Bottle es intermedia. Car y cell phone necesitan más ejemplos difíciles. Book falla de forma marcada, especialmente en objetos pequeños, apilados u ocluidos. No debe afirmarse que el motor alcanza alta precisión global.

