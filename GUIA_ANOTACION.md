# Guía de anotación — detección v1

## Clases

| ID COCO | Clase | Incluye | Excluye/ambigüedad |
|---:|---|---|---|
| 1 | bicycle | Bicicleta completa o parte inequívoca | Motocicleta, patinete, bicicleta estática |
| 39 | bottle | Recipiente alargado con cuello/tapa identificable | Vaso, taza, lata, garrafa ambigua sin cuello |
| 56 | chair | Asiento individual con respaldo | Sofá, banco, taburete sin respaldo si no es inequívoco |
| 63 | laptop | Computador portátil, abierto o cerrado si es inequívoco | Teclado suelto, monitor, tableta |
| 67 | cell phone | Teléfono móvil visible, incluida pantalla apagada | Control remoto, tableta, calculadora |

No crear etiquetas por inferencia contextual. Si no es inequívoco, marcar para revisión.

## Cajas

- Una caja por instancia visible; ajustada al contorno visible, sin margen artificial.
- Oclusión: anotar si la identidad es inequívoca y se ve una parte sustancial; registrar `occluded=true` en metadatos auxiliares. No estimar la extensión oculta en el TXT YOLO.
- Truncado por borde: caja hasta el borde; registrar `truncated=true`.
- Reflejos, fotografías o pantallas que muestran el objeto: no anotar como objeto físico; marcar `depiction` si se conserva como negativo difícil.
- Objetos pequeños: anotar si puede identificarse al 200 % sin adivinar. Registrar ancho/alto en píxeles para análisis.
- Varios objetos: caja separada por instancia. No encerrar un grupo completo salvo que la fuente solo ofrezca `group_of`, que debe quedar fuera del entrenamiento hasta revisión.
- Objetos anidados: anotar cada clase válida independientemente (p. ej. teléfono sobre silla).

Formato: `class_id x_center y_center width height`, todo normalizado a `[0,1]`, un objeto por línea, IDs cero-basados de COCO. Para segmentación futura: `class_id x1 y1 x2 y2 ...` con vértices normalizados y al menos tres puntos.

## Control de calidad

Primera pasada por anotador; segunda revisión independiente de una muestra mínima del 20 % y del 100 % de casos ambiguos. Las etiquetas generadas por YOLO/YOLOE deben llevar `label_origin=pseudo` y `validation_status=pending`; no entran al conjunto aprobado hasta revisión humana. No borrar sospechosos: registrarlos en `datasets/reports/quarantine_candidates.csv` y copiar/mover solo tras decisión humana documentada.

