# VisionAR Own Dataset v1

Dataset local creado para validar y mejorar las clases prioritarias del entorno real del usuario. El detector general parte de YOLO26 preentrenado sobre Objects365 (365 clases); este conjunto local cubre tanto clases ya conocidas como vacíos que requieren ajuste propio.

## Clases

El esquema contiene 38 clases: personas, gafas, camisa, cortaúñas, teléfono, perfume, planta, silla, marcador, borrador, televisor, pez, gato, perro, moto, cartuchera, billetera, tarjeta de pago, cubiertos, lapicero, mochila, bolso, cartas, cinta, destornillador, papel, mesas, sillón, gorra, buzo, tenis, armario, almohada, llaves, USB y billete.

La lista canónica y la separación entre clases y atributos está en `training/configs/unified_taxonomy.yaml`. Colores, estampados y materiales visibles se describen como atributos mediante el VLM; no se convierten en clases YOLO incompatibles entre sí.

## Captura, importación y revisión

1. Ejecute `npm run dev` y abra **Dataset Studio**.
2. Indique procedencia y licencia. Use fotografías propias, la cámara móvil, una carpeta local o una URL HTTPS directa de Hugging Face/Roboflow con licencia verificada.
3. Importe una o varias imágenes. La aplicación rechaza archivos dañados y duplicados SHA-256; también alerta sobre imágenes casi idénticas mediante hash perceptual.
4. Abra una imagen en la cola de revisión.
5. Dibuje manualmente una caja o pulse **Sugerir con YOLO** y corrija la sugerencia.
6. Seleccione la clase y el conjunto `train`, `val` o `test`.
7. Pulse **Aprobar anotación revisada**.

Las sugerencias automáticas nunca incrementan el contador de revisadas. Solo la aprobación humana exporta la caja a YOLO. La misma fotografía puede contener varias anotaciones; por eso 30 anotaciones no equivalen necesariamente a 30 fotografías.

## Estructura

```text
datasets/visionar_own_v1/
├── data.yaml
├── metadata.json
├── manifest.jsonl
├── candidates.jsonl
├── candidates/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Las etiquetas usan polígonos rectangulares YOLO: `class_id x1 y1 x2 y2 ...`, con coordenadas normalizadas. El manifiesto conserva clase, caja, estado de revisión, origen, licencia, SHA-256, hash perceptual y split.

## Plan de recopilación

- Piloto mínimo: 30 anotaciones revisadas por clase.
- Recomendado antes del primer fine-tuning: 100–200 imágenes variadas por clase.
- Variar iluminación, fondo, distancia, orientación, oclusión y cámara.
- Incluir teléfonos y perfumes visualmente parecidos como negativos difíciles.
- No extraer fotogramas casi idénticos de un mismo video entre train, val y test.
- Mantener una sesión física completa dentro de un solo split para evitar fuga de información.

El panel muestra fotografías, anotaciones revisadas, pendientes y faltantes para las 38 clases. El mínimo operativo es 1.140 anotaciones revisadas, pero no garantiza precisión; se recomiendan 100–200 fotografías variadas por clase para un modelo útil.

El entrenamiento puede iniciarse y cancelarse desde Dataset Studio o con `npm run train:own`. Siempre se vuelve a auditar antes de lanzar el proceso. Detecta GPU compatible si existe; en este equipo usa CPU, `batch=2`, `workers=0` y modelo Nano para proteger la memoria. Al finalizar copia el mejor checkpoint a `training/checkpoints/visionar_38_best.pt`; el servidor lo carga automáticamente al reiniciar. La auditoría CLI se ejecuta con `npm run train:audit`.

Prueba integral reproducible con una fotografía real del paquete COCO8-seg:

```powershell
python scripts/test_dataset_workflow.py
```

