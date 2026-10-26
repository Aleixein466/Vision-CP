# Mejoras basadas en observaciones reales

**Fecha:** 25 de octubre de 2026.

## Diagnóstico

El checkpoint `yolo26n.pt` instalado está entrenado con las 80 categorías COCO. Su tabla de clases confirma que contiene `person` y `bird`, pero no contiene `key`, `USB`, `flash drive` ni `fish`. Por ello:

- Una mano con llaves o USB termina representada únicamente por la caja `person`.
- En un acuario, el detector cerrado intenta asignar formas desconocidas a una categoría disponible; `bird` puede convertirse en un falso positivo.
- Bajar la confianza aumenta falsos positivos y no crea clases nuevas.

El lag provenía además de ejecutar inferencias YOLO periódicas mientras llama.cpp consumía CPU y RAM para la respuesta multimodal.

## Correcciones implementadas

1. Se añadió el checkpoint oficial `yoloe-26n-seg-pf.pt`, SHA-256 registrado, con vocabulario abierto de 4.585 nombres.
2. Se añadieron modos General, Manos/objetos pequeños y Acuario.
3. Los modos especializados funcionan bajo demanda; no ejecutan YOLOE continuamente.
4. El modo acuario elimina resultados terrestres y conserva términos acuáticos.
5. Se añadió selección manual por arrastre y selección automática al pulsar una detección.
6. El recorte seleccionado se amplía y se entrega al VLM con instrucciones para no adivinar.
7. La captura especializada conserva hasta 1280 px para objetos pequeños.
8. YOLO en vivo se pausa durante el análisis del VLM.
9. Solo se admite una solicitud VLM simultánea.
10. La detección en vivo exige dos apariciones en una ventana de tres muestras.
11. La respuesta se limita a 180 tokens para reducir tiempo innecesario.

## Pruebas reales

Sobre la imagen COCO del portátil y ratón:

| Prueba | Resultado |
|---|---|
| YOLO26 general | portátil 0,9508; ratón 0,8005 |
| YOLOE modo pequeño | ratón 0,8985 |
| Latencia YOLOE caliente observada | 1.074,64 ms |
| Selección y recorte | 199 × 188 px, válido |
| Flujo región + SmolVLM | identificó un ratón gris y rojo con cable |
| Tiempo total región + SmolVLM | 15,773 s |

SmolVLM volvió a responder en inglés, confirmando que Qwen2-VL debe seguir siendo el modelo principal para español.

## Validación pendiente

No había un fotograma real del acuario, llaves o USB disponible en el proyecto. Por tanto no se afirma todavía que esos tres casos estén resueltos con precisión. La infraestructura para enfocarlos y evitar el falso `bird` está implementada; la siguiente prueba debe realizarse con la cámara del usuario. Si persisten errores, deben conservarse imágenes representativas para formar un pequeño conjunto etiquetado antes de considerar fine-tuning.
