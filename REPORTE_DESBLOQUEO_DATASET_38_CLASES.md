# Informe final — flujo de dataset y entrenamiento de 38 clases

Fecha de validación: 25 de octubre de 2026 (America/Bogota).

## Causa del bloqueo

Dataset Studio dependía de una imagen activa y una región seleccionada en Live Vision. No había bandeja de importación, cola de revisión, editor propio, aprobación humana ni administrador de entrenamiento. Por eso `datasets/visionar_own_v1` mostraba 0 imágenes y 0 anotaciones: no había fotografías aportadas por el usuario y el sistema no debía inventarlas.

## Componentes implementados

- Importación desde archivos, carpetas, cámara móvil y el fotograma actual de cámara/imagen/video.
- Importación de una imagen por URL HTTPS directa desde dominios autorizados de Hugging Face, Roboflow o GitHub Raw, con procedencia y licencia obligatorias.
- Validación real con OpenCV, límite de tamaño, SHA-256 para duplicados exactos y dHash para casi duplicados.
- Cola persistente de candidatos con estados `pending`, `reviewed` y `rejected`.
- Editor de caja manual; sugerencias opcionales de YOLO; cambio de clase y split antes de aprobar.
- Las sugerencias no cuentan como revisadas. Solo una aprobación explícita crea la etiqueta YOLO y el registro `manual_rectangle_reviewed`.
- Protección contra la misma imagen en splits diferentes.
- Panel para las 38 clases con fotos, anotaciones, revisadas, pendientes, objetivo y faltantes.
- Auditoría de imágenes corruptas, etiquetas ausentes, índices, coordenadas y duplicados entre splits.
- Administrador real de entrenamiento con estado, log, inicio y cancelación. Rechaza el inicio si no están listas las 38 clases.
- Detección automática de dispositivo. Resultado comprobado en este equipo: CPU; CUDA no disponible.
- Exportación de `best.pt` a `training/checkpoints/visionar_38_best.pt` y carga automática por la aplicación después de reiniciar.

## Pruebas ejecutadas

- Compilación de cuatro módulos Python y validación sintáctica de JavaScript: correcta.
- Suite existente: 11/11 pruebas correctas.
- Prueba integral `scripts/test_dataset_workflow.py` con `datasets/coco8-seg/images/val/000000000036.jpg`, una fotografía real incluida en el paquete COCO8-seg:
  - importación correcta;
  - segunda importación exacta rechazada;
  - caja de persona obtenida de la anotación COCO y aprobada como revisión;
  - archivo YOLO generado;
  - contador actualizado a 1;
  - entrenamiento de 38 clases rechazado correctamente por cobertura insuficiente.
- API local comprobada en el puerto 8082: estadísticas, auditoría y estado de hardware respondieron con datos reales.
- Cámaras físicas enumeradas: cámara 0 a 640×480 y cámara 1 a 1280×720. Smoke test real sobre cámara 0: 5 fotogramas inferidos, último rendimiento 9,23 FPS a 480 px; no hubo detecciones sobre esos cinco fotogramas.
- La aplicación completa quedó activa con SmolVLM-500M y YOLO26 Nano. En la comprobación final la RAM del sistema estaba al 88,6 % (1.384 MB disponibles); por ello Small/Medium/Large/XL y entrenar mientras el asistente está abierto pueden volver a agotar memoria. Los límites OpenMP/OpenCV siguen activos y el entrenamiento CPU usa un único worker.
- La inspección visual automatizada del navegador no pudo completarse porque el auxiliar nativo de Computer Use no estaba disponible (`failed to connect native pipe`). No se declara esa prueba como superada; sí se validaron el JavaScript, el HTML servido, las rutas API y el flujo E2E del backend.
- Ya se ejecutó antes un entrenamiento smoke real de un epoch sobre ocho imágenes COCO8-seg. Sus resultados están en `REPORTE_ENTRENAMIENTO.md`; no son métricas del futuro modelo VisionAR de 38 clases.

## Estado real del dataset propio

- Fotografías persistentes de producción: 0.
- Anotaciones persistentes revisadas: 0 de 1.140.
- Clases completas: 0 de 38.
- Clases incompletas: las 38.
- Prueba temporal: 1 fotografía y 1 anotación revisada; se ejecutó en un directorio temporal y no se mezcló con el dataset propio.
- Entrenamiento personalizado completo: bloqueado correctamente.

No se copiaron las ocho imágenes de COCO8 al dataset propio porque no cubren las 38 clases ni sustituyen fotografías reales del dominio VisionAR. La prueba temporal demuestra el flujo sin inflar los contadores de producción.

## Datos que todavía debe aportar el usuario

El mínimo operativo es 30 anotaciones revisadas por clase. Conviene aportar 100–200 fotografías variadas por clase para mejorar generalización. Se necesitan imágenes reales de todas las clases: `person`, `eyeglasses`, `shirt`, `nail_clipper`, `cell_phone`, `perfume_bottle`, `potted_plant`, `chair`, `marker`, `eraser`, `tv`, `fish`, `cat`, `dog`, `motorcycle`, `pencil_case`, `wallet`, `payment_card`, `fork`, `spoon`, `knife`, `pen`, `backpack`, `handbag`, `playing_card`, `adhesive_tape`, `screwdriver`, `paper_sheet`, `table`, `sofa`, `cap`, `sweatshirt`, `sneakers`, `wardrobe`, `pillow`, `keys`, `usb_drive` y `banknote`.

Las capturas deben variar fondo, distancia, iluminación, orientación y oclusión. Para teléfono/perfume, tarjeta/billete, llave/USB y objetos pequeños en mano son especialmente importantes los negativos difíciles y las tomas cercanas. No se deben repartir fotogramas casi idénticos de una misma sesión entre train, val y test.

## Ejecución

```powershell
cd C:\Users\alexi\Desktop\vision\VISIONAR-AI
npm run dev
```

Abra `http://127.0.0.1:8080`, entre a **Dataset Studio**, importe imágenes, revise cada caja y observe el progreso. El botón **Iniciar entrenamiento** se habilita solo tras completar y auditar las 1.140 anotaciones revisadas.
