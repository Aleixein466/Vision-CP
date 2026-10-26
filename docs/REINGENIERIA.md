# VisionAR: reingeniería y evidencias

Validación local: 26 de octubre de 2026, America/Bogota. Este documento describe el código revisado y las pruebas nuevas; los informes de las fases anteriores son históricos.

## Estado del dataset real

Se reconciliaron 1.034 registros con 1.034 JPEG físicos. No se encontraron archivos faltantes/corruptos, cambios de SHA-256, duplicados exactos ni grupos dHash a distancia <=3 en este conjunto. Esto no garantiza ausencia de duplicados semánticos.

Hay **1.034 fotografías pendientes, 959 propuestas de cajas y cero anotaciones humanas aprobadas**. Las 38 clases tienen candidatos o clases previstas; una clase prevista no es una anotación. No se descargaron fotografías nuevas ni se aprobaron automáticamente las importadas. La cobertura detallada está en `evidence/dataset_audit.json`.

798 registros conservan enlaces tanto de procedencia como de licencia. Otros 236, importados del train de Open Images, solo conservan una declaración general de licencia y necesitan evidencia individual antes de redistribuirse. No se ha afirmado que esos permisos hayan sido verificados remotamente.

## Cambios implementados

- SQLite en `datasets/visionar_own_v1/studio.sqlite3`, transacciones, WAL, respaldo inicial y backups manuales. La migración conserva intactos los JSONL, reporta líneas inválidas y no descarta silenciosamente errores. SQLite es la fuente vigente; los JSONL originales dejan de actualizarse.
- Reconciliación de registros, hashes, archivos físicos, etiquetas heredadas y clases. Conteos separados de propuestas, fotos pendientes, cajas revisadas y máscaras revisadas. Una sugerencia de confianza 1.0 no acredita revisión.
- Bandeja paginada, estados, propuestas seleccionables, múltiples objetos por fotografía y lotes transaccionales. La revisión completa de cada fotografía es explícita. Se pueden retirar anotaciones para corregirlas conservando el registro histórico.
- Cajas YOLO de cinco campos para detección. Contornos independientes para segmentación, validación de coordenadas/área/intersecciones y edición por vértices. GrabCut produce propuestas sobre CPU y nunca las marca revisadas. No hay conversión automática de una caja a una máscara rectangular.
- Exportaciones de entrenamiento nuevas por experimento; los archivos originales permanecen intactos. Auditoría de fotos únicas por clase, train/val/test, revisión completa, procedencia y duplicados. Las sesiones de cámara/video y grupos de duplicados no pueden cruzar splits.
- Entrenamiento CPU configurable: prueba, parcial y completo; epochs, batch, resolución, hilos y presupuesto RAM. Se descargan de memoria los motores administrados al entrenar desde el dashboard. Un VLM externo exige cerrarlo o usar su administrador. La ejecución tiene exclusión entre procesos y vigilancia de memoria de todo el árbol Python.
- Cancelación cooperativa al completar el epoch; checkpoints de cada epoch; recuperación real mediante `resume`. Una emergencia de memoria termina el proceso y conserva los epochs previamente guardados.
- Métricas reales por clase, validación y test separados, huella del dataset y comparación de experimentos solo con el mismo dataset/tarea. Activación explícita de modelos evaluados y reversión al anterior; requiere reiniciar el servidor. Los experimentos de prueba nunca son promocionables.
- Inferencia localizada que amplía el recorte y remapea las coordenadas; seguimiento por instancia mediante IoU, suavizado y desaparición temporal. La asociación no garantiza identidad en cruces u oclusiones largas. La confianza es una puntuación del detector, no una probabilidad calibrada.
- El VLM recibe detecciones de la misma imagen/recorte que ve. Se refuerza la distinción entre evidencia, historial y detalles no observables. Esto no elimina la posibilidad de alucinaciones.
- Seis secciones funcionales del dashboard, preferencias de rendimiento, errores visibles, estados del VLM separados de YOLO, navegación móvil, foco de teclado y reducción de animaciones.
- API restringida a host/origen local, JSON obligatorio, validación de imágenes antes de decodificar, rechazo de traversal y redirecciones de importación fuera de los dominios permitidos. Nuevas capturas, SQLite, backups, experimentos y checkpoints están ignorados por Git.

## Uso

```powershell
python -m pip install -r requirements.txt
npm run dev
```

En Dataset Studio, abre una foto, revisa cada objeto y añade sus anotaciones al lote. Para segmentar, dibuja primero una caja si deseas usar GrabCut, cambia a segmentación, genera la propuesta y corrige sus vértices. Marca la revisión completa solo después de revisar todos los objetos de las clases del dataset. Guarda el lote. La exportación se genera al crear un experimento, no en las carpetas heredadas.

Entrenamiento parcial exige al menos dos fotos completas de train por clase seleccionada, una de validación y una de test. Completo exige treinta fotos de train por cada una de las 38 clases, más validación y test. Son mínimos operativos, no una garantía de calidad. Para la prueba se usan las ocho fotos COCO8 existentes: cuatro train, tres val y una test; sus métricas solo prueban la tubería y no estiman la precisión del dataset propio ni excluyen exposición del modelo base durante su preentrenamiento.

```powershell
python scripts/train_visionar.py --execute --mode smoke --epochs 1 --model training/checkpoints/yolo26n.pt
python scripts/train_visionar.py --mode partial --classes person chair
```

Después de entrenar, activa la detección continua para reanudar YOLO; una consulta reinicia el VLM administrado bajo demanda. La activación o reversión de un checkpoint se aplica al reiniciar el servidor.

Los backups SQLite protegen los registros, **no son copias de las fotografías**. Conserva también la carpeta completa del dataset. Para restaurar, cierra VisionAR, conserva una copia de la base actual y sus archivos WAL/SHM y restaura una copia del backup como `studio.sqlite3` sin reutilizar los WAL/SHM de otra base. No se implementó una restauración automática desde la interfaz.

## Evidencias y reproducción

```powershell
python -m pip install -r requirements-dev.txt
npm test
npm run tracking:test
npm run dataset:e2e
npm run verify:runtime
npm run verify:recovery
npm run verify:dashboard
npm run verify:multimodal
```

La prueba de navegador usa Edge instalado y un dataset temporal. Las aprobaciones de fixtures en pruebas no se mezclan con revisiones humanas del dataset propio. La prueba de cámara no conserva fotografías.

| Evidencia | Alcance |
|---|---|
| `evidence/dataset_audit.json` | Reconciliación física y cobertura de las 38 clases |
| `evidence/runtime_validation.json` | Inferencia real, recorte y entrenamiento CPU con evaluación |
| `evidence/recovery_validation.json` | Cancelación real y continuación desde el epoch guardado |
| `evidence/model_comparison.json` | Comparación de dos ejecuciones sobre el mismo conjunto |
| `evidence/dashboard_validation.json` | API, importación/revisión en navegador y tamaños 390/768/1440 px |
| `evidence/dashboard-desktop.png`, `dashboard-mobile.png`, `training-desktop.png` | Capturas de interfaz del fixture de prueba |
| `evidence/multimodal_validation.json` | Tres frames de cámara, máscaras reales, GrabCut y respuesta del VLM |
| `evidence/verification_summary.json` | Salida de las 27 pruebas Python, comprobaciones JavaScript, cinco experimentos y hashes del código |

También se completó un epoch real de segmentación con métricas de cajas y máscaras separadas, test independiente y pico de RAM de 584,8 MB del árbol de procesos. Los cinco experimentos son pruebas de tubería, ninguno es un modelo de producción.

## Límites pendientes

No se ejecutó entrenamiento parcial/completo sobre las 38 clases porque no existen revisiones humanas aprobadas. No se produjo ni activó un nuevo modelo de producción. Las ganancias de precisión en objetos pequeños/ocultos requieren un conjunto de evaluación revisado; no se han inventado porcentajes de mejora.

SmolVLM respondió a la foto real en inglés pese a la instrucción en español. La evidencia registra su respuesta; la prueba acredita funcionamiento, no exactitud factual ni obediencia lingüística. No se validó toda la diversidad de escenas, oclusiones o cámaras. La comprobación móvil usa un viewport de navegador, no hardware móvil físico.

Los backups conservan datos sensibles sin cifrado adicional; dependen de los permisos del usuario de Windows. Las fotografías previamente publicadas en Git siguen en el historial: `.gitignore` protege nuevas capturas, no retira las ya publicadas.
