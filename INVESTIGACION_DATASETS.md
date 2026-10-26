# Investigación de modelos y datasets

**Corte de investigación:** 2026-10-25. **Regla:** los valores no publicados se indican como tales; “candidato” no significa descargado ni aprobado.

## Hallazgos de YOLO26

La [documentación oficial de YOLO26](https://docs.ultralytics.com/models/yolo26/) publica escalas `n/s/m/l/x` para detección, segmentación de instancias, segmentación semántica, profundidad monocular, clasificación, pose y OBB. Todos figuran con train/val/predict/export. `yolo26-p2.yaml` y `yolo26-p6.yaml` son arquitecturas, no pesos publicados.

Los detectores `yolo26*.pt` están preentrenados en las 80 clases COCO. No se atribuye a estos pesos ninguna clase fuera de esa ontología. En COCO, YOLO26n reporta oficialmente 40,9 mAP50-95 a 640 px y 38,9 ± 0,7 ms de CPU ONNX; esta cifra no describe este portátil ni PyTorch. El benchmark local está en `EVALUACION_MODELO_BASE.md`.

YOLOE es [open vocabulary](https://docs.ultralytics.com/models/yoloe/): acepta prompts de texto o visuales y dispone de un peso prompt-free con 4.585 nombres. Los pesos `*-seg.pt` admiten `set_classes()`; los `*-seg-pf.pt` no. El primer prompt textual de YOLOE-26 descarga un codificador de unos 254 MB y requiere `ultralytics>=8.4.0`. Es útil para exploración y pseudoetiquetado validado, no sustituye una evaluación cerrada. El entrenamiento desde cero documentado usa alrededor de 1,4 M muestras y 8 RTX 4090, fuera del equipo disponible.

Formatos oficiales: detección usa una línea `class_id x_center y_center width height`, coordenadas normalizadas `[0,1]` e IDs desde cero. La [validación](https://docs.ultralytics.com/modes/val/) expone precision, recall, mAP50, mAP50-95, métricas por clase y matriz de confusión. El [entrenamiento](https://docs.ultralytics.com/modes/train/) admite `device=cpu`, aunque debe perfilarse localmente.

## Cobertura de las 25 categorías solicitadas

| Categoría | COCO/Yolo26 cerrado | Open Images V7 | Decisión inicial |
|---|---|---|---|
| laptop | Exacta `laptop` | `Laptop` | MVP |
| desktop | No | No exacta | Anotación propia futura |
| cell phone | Exacta | `Mobile phone` | MVP |
| keyboard | Exacta | `Computer keyboard` | Reserva |
| mouse | Exacta | `Computer mouse` | Reserva; evitar animal `Mouse` |
| monitor | Parcial `tv` | `Computer monitor` | OI futuro |
| camera | No | `Camera` | OI futuro |
| bottle | Exacta | Exacta | MVP |
| glass/cup | `cup`, `wine glass` | Parcial | Definir ontología antes |
| chair | Exacta | Exacta | MVP |
| table | `dining table` | `Table` | Semántica distinta |
| book | Exacta | Exacta | Reserva MVP |
| backpack | Exacta | Exacta | Reserva |
| watch/clock | `clock` | `Watch`, `Clock` | Separar conceptos |
| car | Exacta | Exacta | Reserva |
| motorcycle | Exacta | Exacta | Reserva |
| bicycle | Exacta | Exacta | MVP |
| bus | Exacta | Exacta | Reserva |
| truck | Exacta | Exacta | Reserva |
| tree | No | Exacta | OI futuro |
| flower | No | Exacta | OI futuro |
| plant | Solo `potted plant` | `Plant`, `Houseplant` | Definir alcance |
| fruit | Solo especies | Genérica y especies | Definir alcance |
| dog | Exacta | Exacta | Reserva |
| cat | Exacta | Exacta | Reserva |

COCO cubre 18 de 25 de forma directa; Open Images cubre 24 de 25. El detalle por fuente está en `MATRIZ_DATASETS.csv`.

## Selección del MVP

Se eligieron `bicycle`, `bottle`, `chair`, `laptop` y `cell phone`: son clases COCO exactas y ya están en los pesos; cubren escalas, contextos y familias distintas. `book` queda como sustituto razonable si la colección de campo demuestra insuficiencia de teléfonos. La selección se confirmó contra COCO128: 21 imágenes contienen al menos una clase seleccionada, con 70 cajas válidas.

## Fuentes y decisiones

- [COCO 2017](https://cocodataset.org/#download): benchmark maduro; 118.287 train + 5.000 val, 80 clases, cajas y máscaras. Descarga completa no realizada por presupuesto.
- [COCO128](https://docs.ultralytics.com/datasets/detect/coco128/): 128 imágenes, ~7 MB. Descargado e inspeccionado; aprobado solo como piloto técnico interno.
- [Open Images V7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html): ~9 M imágenes; 15,85 M cajas en 600 clases. No descargado; usar solo subconjuntos filtrados y verificar licencias por imagen.
- [LVIS v1](https://www.lvisdataset.org/): candidato de vocabulario largo; no independiente de COCO.
- Catálogos dinámicos (Roboflow, Hugging Face, Kaggle y Google Dataset Search) sirven para descubrir, no prueban licencia ni procedencia.

## Pendiente para una evaluación representativa

Recopilar y anotar fotografías propias por clase con iluminación, oclusión, distancia, ángulo, fondo y múltiples objetos. El objetivo mínimo recomendado es 50–100 imágenes de prueba por clase, agrupadas por sesión antes de dividir. No se recomienda fine-tuning hasta completar esa evaluación.

