# Informe final — Fase 0

**Corte:** 2026-10-25.

## Resultado

Se construyó una base reproducible, no un dashboard: investigación verificable, inventario de fuentes/licencias, piloto COCO128 filtrado, auditoría, baseline YOLO26n en CPU, catálogo de conocimiento, candidatos 3D y plan de entrenamiento.

## Obtenido versus pendiente

| Criterio | Estado | Evidencia / pendiente |
|---|---|---|
| Investigación de datasets | Completo para decisión inicial | `INVESTIGACION_DATASETS.md`, `MATRIZ_DATASETS.csv` |
| Clases preentrenadas identificadas | Completo | 18/25 exactas en COCO; matices documentados |
| Dataset piloto reproducible | Completo técnico | COCO128 → 21 imágenes, 70 cajas, scripts y manifiestos |
| Anotaciones verificadas | Completo para validación automática | 0 errores; revisión semántica humana de campo pendiente |
| Procedencia conservada | Completo para archivos presentes | ZIP original, SHA-256 y manifiesto por imagen |
| Base de conocimiento inicial | Completo para 5 clases | `CATALOGO_OBJETOS.json`; verificación básica, no de producto |
| Recursos 3D identificados | Completo como catálogo | 5 candidatos; 0 descargados, requiere inspección de archivos |
| Modelo base evaluado | Completo como piloto exploratorio | CPU: 122,3 ms inferencia media en 21 imágenes |
| Recomendación de fine-tuning | Completo | No entrenar todavía; crear test propio primero |
| Reproducibilidad | Completo para la tubería actual | README, versiones, hashes, scripts y tests |

## Decisión técnica

Usar `yolo26n.pt` cerrado para el primer MVP con bicycle, bottle, chair, laptop y cell phone. No usar YOLOE en el camino crítico de CPU: añade dependencias/peso y su rendimiento abierto necesita medición. Mantener YOLOE como herramienta de descubrimiento/pseudoetiquetado para categorías fuera de COCO.

Las métricas actuales no autorizan una afirmación de alta precisión. El test retenido tiene solo 2 imágenes; el diagnóstico de 21 imágenes alcanza precision 0,551, recall 0,491, mAP50 0,518 y mAP50-95 0,370. Cell phone es la debilidad principal observada, pero aún faltan datos representativos.

## Próxima puerta de fase

1. Capturar un test propio balanceado (mínimo recomendado 50–100 imágenes/clase) con la matriz de condiciones descrita.
2. Revisión humana y licencia/consentimiento de cada archivo; congelar el test por hash.
3. Repetir baseline. Solo si incumple objetivos, preparar fine-tuning focalizado.
4. Descargar e inspeccionar un activo 3D CC0 por clase cuando sea posible; producir GLB `web-low` ≤10 MB, texturas 1K/2K y ≤20K triángulos, conservando original y transformación.

## Reproducción

Ejecutar los comandos de `README.md`. Los tests pasan (2/2). El workspace tenía 19,42 GB libres al inicio; se evitó COCO completo (~20+ GB) y Open Images completo (~561 GB). No se inició entrenamiento largo ni se descargaron activos 3D con licencias o tamaños pendientes.

