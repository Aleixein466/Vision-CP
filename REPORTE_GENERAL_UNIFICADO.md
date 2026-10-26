# VisionAR AI - detector general unificado

**Fecha:** 25 de octubre de 2026.

## Incidente corregido

El fallo `OMP Error #137 / System error #1450` fue agotamiento de recursos de Windows. Se aplicaron cuatro defensas:

1. PyTorch, interoperabilidad y OpenCV limitados a un hilo.
2. llama.cpp limitado a dos hilos y un contexto paralelo.
3. Una sola inferencia YOLO simultanea; los fotogramas sobrantes se descartan con HTTP 429.
4. Bloqueo preventivo de modelos Medium/Large/Extra Large cuando la RAM libre no alcanza el minimo seguro.

El proceso huérfano de llama.cpp que quedó tras la caída fue cerrado de forma dirigida antes de validar.

## Modelos oficiales instalados

- `yolo26n-objv1-150.pt`: YOLO26 Nano, cajas, 365 clases Objects365.
- `yolo26n-objv1-seg.pt`: YOLO26 Nano, segmentación, 365 clases Objects365.
- `yolov8n-oiv7.pt`: respaldo de investigación con 601 clases Open Images V7; no se mantiene cargado en vivo.
- `yolo26s.pt`: modelo Small COCO solicitado, instalado y registrado.
- `yolo26s-objv1-150.pt`: variante Small de 365 clases usada por la opción Small de VisionAR.
- `yolo26s-objv1-seg.pt`: variante Small de segmentación con 365 clases.

No se descargaron Objects365 ni Open Images completos: requieren aproximadamente 712 GB y 561 GB respectivamente, mientras la unidad tenía unos 16,9 GB libres. Se usaron los pesos oficiales ya preentrenados, que ocupan pocos megabytes.

## Cobertura solicitada

De las 38 clases del dataset propio, 28 tienen equivalencia nativa en Objects365. Entre ellas: persona, gafas, teléfono, planta, silla, marcador, borrador, televisor, pez, gato, perro, moto, cartuchera, cubiertos, lapicero/lápiz, mochila, bolso, carta, cinta, destornillador, papel, mesa, sillón, gorra/sombrero, tenis y almohada.

Requieren datos propios o verificación VLM: camisa, cortaúñas, perfume, billetera, tarjeta de pago, buzo, armario, llaves, USB y billete. Colores, estampados y materiales son atributos visuales del VLM.

## Validaciones ejecutadas

- 10/10 pruebas unitarias superadas.
- Seguimiento de escena: superado.
- 20 solicitudes YOLO secuenciales: 20 respuestas HTTP 200.
- 24 solicitudes simultáneas: 2 procesadas y 22 descartadas con HTTP 429; cero respuestas 502.
- Segmentación real: 4 detecciones y 4 máscaras, 101,42 ms.
- Prueba completa YOLO + Qwen2-VL: respuesta HTTP 200 en español, con portátil, ratón y escritorio correctamente descritos.
- llama.cpp validado con `n_threads=2` y `n_slots=1`.

## Corrección de imágenes y globos

Las fotografías cargadas solo reciben una inferencia. Antes pasaban por el estabilizador temporal de cámara, que exige dos apariciones, de modo que una caja válida podía no dibujarse. Las imágenes ahora muestran las detecciones desde la primera inferencia y usan 640 px con umbral 0,18.

Sobre una fotografía real CC BY-SA 4.0 de Wikimedia Commons, YOLO26 Small Objects365 detectó tres globos con confianzas 0,667, 0,538 y 0,475. La API devolvió HTTP 200, cinco detecciones totales y 224,43 ms. La imagen se usó temporalmente y se eliminó después de la prueba.

## Entrenamiento propio

El entrenamiento se volvió a invocar y fue bloqueado por el auditor: el dataset local tiene 0 imágenes y 0 anotaciones reales. Entrenar con esa entrada no es matemáticamente posible. La base instalada ya fue entrenada por Ultralytics sobre Objects365; el siguiente fine-tuning local comenzará cuando existan al menos 30 anotaciones revisadas por clase, priorizando las diez clases sin cobertura nativa.
