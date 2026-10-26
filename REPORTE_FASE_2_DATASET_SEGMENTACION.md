# Fase 2 — Dataset propio, segmentación y selector YOLO26

**Fecha:** 25 de octubre de 2026.

## Diagnóstico

Un teléfono y un frasco de perfume pueden compartir silueta, brillo y tamaño. El detector COCO dispone de `cell phone` y `bottle`, pero no de `perfume bottle`; por eso cambiar el umbral no crea la distinción requerida. La corrección sostenible necesita ejemplos propios de ambos objetos y negativos difíciles del entorno real.

El error HTTP 502 observado podía terminar en una excepción adicional cuando el navegador cancelaba un fotograma y el servidor intentaba escribir la respuesta. El servidor ahora ignora desconexiones esperadas y registra errores reales con traceback en `training/metrics/vlm/web_errors.log`.

## Implementado

- VisionAR Own Dataset v1 con ocho clases propias, incluidas billetera, billete y tarjeta.
- Dataset Studio en el frontend.
- Captura explícita; no se guardan imágenes automáticamente.
- Separación train/val/test.
- SHA-256 por imagen y reutilización del mismo archivo para anotaciones adicionales.
- Máscara inicial GrabCut y fallback rectangular, ambas marcadas como pendientes de revisión.
- YOLO26 seleccionable en escalas nano, small, medium, large y extra large.
- Tareas seleccionables: cajas o segmentación de instancias.
- Descarga bajo demanda desde Ultralytics; no se descargaron diez modelos automáticamente.
- Liberación del modelo anterior al cambiar para controlar RAM.
- Máscaras semitransparentes dibujadas en cámara, imagen y video.

## Validación real

Se instaló `yolo26n-seg.pt`, 6.719.965 bytes, SHA-256 `361fbabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f`.

En la imagen COCO real del portátil y ratón:

- 2 detecciones.
- 2 máscaras de instancia válidas.
- 77,53 ms de inferencia HTTP después del calentamiento.
- Catálogo frontend/backend: 10 combinaciones, cinco escalas × dos tareas.
- Dataset propio: 0 imágenes y 0 anotaciones al entregar; no se inventaron datos.

## Estado de entrenamiento

El entrenamiento fue autorizado e invocado el 25 de octubre de 2026. El auditor lo bloqueó porque las ocho clases tienen 0 muestras reales. El piloto exige al menos 30 anotaciones revisadas por clase; para una primera comparación fiable se recomiendan 100–200 imágenes variadas por clase. Los modelos `m`, `l` y `x` pueden ser demasiado lentos para tiempo real en el Intel Core i5 y deben evaluarse bajo demanda.

## Referencia audiovisual

El canal compartido muestra flujos de segmentación en imágenes y video/tiempo real con Mask R-CNN. VisionAR adopta el comportamiento relevante —máscara por instancia sobre imagen, cámara y video— usando YOLO26-seg, que se integra con el motor actual y es más apropiado para la CPU objetivo.
