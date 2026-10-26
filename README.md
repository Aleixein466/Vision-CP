# VisionAR AI — Fase 0

> Estado actual: [reingeniería, uso y pruebas verificables](docs/REINGENIERIA.md). El contenido de Fase 0 que sigue se conserva como documentación histórica. Dataset Studio usa SQLite; detección y segmentación tienen anotaciones y exportaciones separadas.

Base reproducible de investigación, datos y evaluación para reconocimiento de objetos. Esta fase no contiene dashboard.

La Fase 1 de evaluación sobre COCO val2017 y detección con cámara está documentada en [README_FASE_1.md](README_FASE_1.md).

Inicio directo del asistente visual local con cámara, razonamiento y voz:

```powershell
npm run dev
```

Consulte [README_ASISTENTE_VISION.md](README_ASISTENTE_VISION.md) para arquitectura, primera descarga y solución de problemas.

La integración VisionAR AI 2.0, que fusiona YOLO26 con el VLM y añade panel de modelos, chat e historial, está documentada en [README_VISIONAR_2.md](README_VISIONAR_2.md).

La recopilación del dataset propio y la segmentación se documentan en [README_DATASET_PROPIO.md](README_DATASET_PROPIO.md) y [REPORTE_FASE_2_DATASET_SEGMENTACION.md](REPORTE_FASE_2_DATASET_SEGMENTACION.md).


## Reproducción rápida (Windows / PowerShell)

```powershell
python -m pip install -r requirements.txt
python scripts/download_with_integrity.py --url https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip --output datasets/raw/coco128.zip --max-mb 20
python scripts/extract_archive.py --archive datasets/raw/coco128.zip --destination datasets/external
python scripts/build_pilot.py
python scripts/audit_dataset.py
python scripts/evaluate_baseline.py
python -m unittest discover -s tests -v
```

El descargador conserva `.part`, reanuda cuando el servidor lo permite, rechaza descargas que superen el presupuesto y registra SHA-256. La extracción nunca sobrescribe destinos existentes. El piloto mantiene las imágenes originales en `external/`, crea copias de trabajo en `processed/`, agrupa duplicados antes de dividir y no elimina sospechosos.

## Estado

- Fuente descargada/inspeccionada: COCO128, subconjunto técnico de COCO train2017.
- MVP: bicycle, bottle, chair, laptop, cell phone.
- Uso: validación de tubería y baseline; no representa el dominio final de cámara/AR.
- Licencia: requiere conservar y revisar la licencia por imagen antes de redistribución o uso comercial. Véase `LICENCIAS_Y_FUENTES.md`.

