# Reporte de validación — VisionAR AI 2.0

**Fecha:** 25 de octubre de 2026. **Equipo:** Intel Core i5-10210U, 11,83 GB RAM, Windows, CPU.

## Auditoría

Se conservaron YOLO26n, el dataset COCO de Fase 1, sus anotaciones, métricas e imágenes. Python 3.11.6, PyTorch CPU 2.14.1, Ultralytics 8.4.189 y OpenCV 5.0.0 estaban disponibles; no fue necesario instalar CUDA ni paquetes NVIDIA.

llama.cpp b11879 y dos VLM multimodales verificados estaban instalados:

- Qwen2-VL 2B Instruct Q4_K_M + mmproj Q8_0: 1.695.930.304 bytes.
- SmolVLM 500M Instruct Q8_0 + mmproj Q8_0: 545.590.272 bytes.

Los SHA-256 se encuentran en `datasets/manifests/vlm_models.json`.

## Prueba de fusión real

Imagen: `datasets/processed/phase1_coco_val/images/test/000000248631.jpg`, parte del conjunto COCO val2017 preparado en Fase 1.

| Componente | Resultado observado |
|---|---|
| YOLO26n | computador portátil, confianza 0,9508 |
| YOLO26n | ratón de computador, confianza 0,8005 |
| Latencia YOLO, arranque frío independiente | 2.859,13 ms |
| Prueba end-to-end final | interfaz 200, estado listo y fusión válida |
| Latencia YOLO dentro del flujo final | 5.005,89 ms |
| Latencia total Qwen2-VL + YOLO | 60,319 s |
| RAM del proceso llama.cpp observada | 2.402,6 MB |
| RAM total del sistema durante la prueba | 90,8 % |

Qwen2-VL describió correctamente el portátil, la pantalla visible, el ratón rojo y su cable. Las métricas corresponden a ejecuciones concretas; no son máximos ni promedios estadísticos.

## Administrador de modelos

El cambio gestionado Qwen2-VL → SmolVLM → Qwen2-VL fue ejecutado correctamente. SmolVLM quedó listo en 6,33 s y Qwen2-VL volvió a quedar listo en 9,93 s, utilizando archivos ya presentes en caché. Solo se finalizó el proceso creado por VisionAR.

## Criterios del primer hito

- Detección de modelos instalados: validada.
- Selección y cambio de VLM: validado.
- Cámara o fotografía: implementado; la cámara requiere permiso del navegador.
- Detección YOLO y cajas: validada con una imagen real.
- Consulta multimodal: validada.
- Chat sobre el fotograma actual: implementado.
- Texto, dictado compatible y reproducción de voz: implementados.
- Interfaz no bloqueante: usa solicitudes asíncronas y estado de carga.
- Métricas reales: registradas y mostradas.
- Privacidad: enlace local, límite de 8 MB, sin guardado automático de imágenes.

## Pendientes declarados

No se implementaron todavía WebXR, biblioteca 3D interactiva, análisis temporal de video, seguimiento persistente, entrenamiento ni Dataset Studio web. La detección en vivo actual realiza muestreo periódico sin solapar trabajos; no identifica todavía eventos temporales como “una persona dejó un objeto”.

La automatización visual no encontró un navegador controlable en esta sesión. La interfaz se validó por HTTP, sintaxis JavaScript y prueba end-to-end; la autorización física de cámara y la inspección visual final deben realizarse al abrir `npm run dev`.
