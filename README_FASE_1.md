# VisionAR AI — ejecución Fase 1

## Preparación

```powershell
cd C:\Users\alexi\Desktop\vision\VISIONAR-AI
python -m pip install -r requirements.txt
```

El entorno validado usa Python 3.11 y PyTorch CPU. No requiere CUDA.

## Reproducir datos y auditoría

Las descargas actuales ya existen y los scripts se niegan a sobrescribirlas. Para una instalación limpia:

```powershell
python scripts/download_with_integrity.py --url http://images.cocodataset.org/annotations/annotations_trainval2017.zip --output datasets/raw/annotations_trainval2017.zip --max-mb 300
python scripts/extract_archive.py --archive datasets/raw/annotations_trainval2017.zip --destination datasets/external
python scripts/prepare_phase1_dataset.py
python scripts/audit_phase1.py
```

## Evaluación

```powershell
python scripts/evaluate_phase1.py
```

Produce métricas JSON/CSV, curvas, matriz de confusión y 100 imágenes anotadas en `training/metrics/phase1/`.

## Cámara

```powershell
npm run dev
```

Este comando permite seleccionar una cámara, reconoce las 80 clases COCO, muestra etiquetas en español y anuncia objetos nuevos mediante la voz offline de Windows. Para limitar el reconocimiento a laptop, teléfono, botella, libro y automóvil:

```powershell
npm run dev:focus
```

Comandos equivalentes y utilidades:

```powershell
python scripts/camera_detector.py --list-cameras
python scripts/camera_detector.py --camera 0
npm run speech:test
npm run demo
```

Controles: `q` o Esc cierra; `l` alterna etiquetas; `s` activa/desactiva voz; `+/-` cambia confianza; `[` y `]` cambian resolución de inferencia entre 320/480/640/800. El sistema exige tres frames consecutivos por encima de 0,45 antes de hablar, espera ocho frames de ausencia para considerar que el objeto salió y aplica ocho segundos de cooldown por clase. Así evita repetir el mismo objeto en cada frame.

Para prueba sin ventana:

```powershell
python scripts/camera_detector.py --camera 0 --smoke-frames 5
```

La cámara siempre se libera dentro de `finally`. Ajuste `frame_skip`, resolución, confianza e `imgsz` en `training/configs/phase1.yaml` para equilibrar CPU y precisión.

`npm run demo` ejecuta una validación end-to-end sobre una fotografía real: detecta un computador portátil y un ratón, genera `training/metrics/phase1/spoken_demo.jpg` y pronuncia ambos resultados. La síntesis es local mediante SAPI/pyttsx3; no envía audio ni texto a un servicio externo.

## Limitaciones

El test es un subconjunto determinista, no COCO val completo ni datos del dominio AR final. Book y cell phone requieren datos adicionales. Las licencias por imagen varían; no redistribuir el subconjunto sin filtrado y revisión legal.

