# Validación de detección continua, chat y video

**Fecha:** 25 de octubre de 2026.

## Implementación

- El intervalo fijo de 1,6 segundos fue sustituido por un bucle adaptativo: cada solicitud comienza después de finalizar la anterior.
- La captura continua usa un fotograma máximo de 512 px y YOLO26n con `imgsz=320`.
- No se apilan solicitudes; solo hay una inferencia activa.
- El calentamiento YOLO ocurre en segundo plano al iniciar el servidor.
- El chat comenta apariciones y desapariciones confirmadas en dos de tres muestras.
- Los comentarios pueden incorporar descripciones verificadas del catálogo local.
- El VLM no se ejecuta en cada fotograma y la detección continua se pausa durante una consulta multimodal.
- Los videos se reproducen localmente y se analizan fotograma por fotograma.

## Mediciones reales

Sobre `000000248631.jpg`, mediante el endpoint HTTP local después del calentamiento:

| Iteración | Latencia YOLO | FPS equivalente | Resultado |
|---|---:|---:|---|
| 1 | 74,70 ms | 13,39 | portátil, ratón |
| 2 | 46,21 ms | 21,64 | portátil, ratón |
| 3 | 41,62 ms | 24,02 | portátil, ratón |

La prueba técnica de video ensambló temporalmente seis fotografías reales del conjunto COCO, las leyó mediante OpenCV y procesó todos sus fotogramas. El promedio completo fue 65,69 ms por fotograma. El archivo temporal fue eliminado al terminar.

Estas cifras no garantizan el mismo FPS con cualquier cámara o video: resolución, iluminación, cantidad de objetos, navegador y carga del sistema modifican el rendimiento.

## Alcance del chat automático

Los comentarios inmediatos se basan en detecciones YOLO estabilizadas y conocimiento local. Las respuestas sobre acciones, relaciones, objetos sostenidos o una explicación extensa siguen usando Qwen2-VL al pulsar **Preguntar**. Esta separación mantiene fluidez y evita que una inferencia VLM de varios segundos bloquee cada fotograma.

## Validación visual pendiente

La infraestructura de navegador controlable no estuvo disponible para automatizar reproducción y permisos de cámara en Edge. Se validaron el HTML, JavaScript, endpoints HTTP, detección continua y recorrido de video. La reproducción visual debe comprobarse después de reiniciar VisionAR y recargar la página.
