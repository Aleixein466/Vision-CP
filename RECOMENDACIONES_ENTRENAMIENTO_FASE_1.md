# Recomendaciones de entrenamiento — Fase 1

No se ejecutó entrenamiento ni existe un modelo personalizado para comparar.

## Prioridad de datos

1. **Book (alta):** 150 cajas disponibles, pero recall operativo 0,053 y mAP50-95 0,064. Recopilar 300–500 imágenes propias revisadas, con libros pequeños, apilados, abiertos, lomos, fondos domésticos y oclusión.
2. **Cell phone (alta):** 36 cajas; recall operativo 0,333. Recopilar 200–300 imágenes, distintos tamaños, pantallas apagadas, teléfonos parcialmente ocultos y negativos difíciles como controles/calculadoras.
3. **Car (media):** 82 cajas; objetos lejanos/ocultos generan FN. Añadir 200–300 imágenes del dominio real de uso, evitando depender solo de escenas urbanas COCO.
4. **Bottle (media):** 82 cajas y muchos FP. Añadir 200–300 imágenes y negativos `cup`, `wine glass`, latas y reflejos.
5. **Laptop (baja):** 47 cajas y mAP50-95 0,563. Ampliar a 150–200 imágenes antes de producción, pero conservar el baseline mientras el test propio mantenga el rendimiento.

Las cantidades son objetivos de recopilación recomendados, no datos ya disponibles. Todas las pseudoetiquetas deben revisarse humanamente.

## Configuración propuesta, no ejecutada

- Inicio: `yolo26n.pt`, 640 px, batch 2, CPU, seed 2602, `workers=0` en Windows.
- Primero una época de calibración; medir tiempo/RAM y extrapolar.
- Después, si se autoriza: 30–50 épocas, early stopping `patience=10`, test propio congelado y evaluación por clase.
- Una corrida CPU larga no tiene estimación fiable hasta la calibración. Si la proyección supera 30 minutos, solicitar autorización y usar una GPU externa; no instalar CUDA en este equipo.

Solo aceptar el fine-tune si mejora book/cell phone en un test propio sin degradar laptop y sin aumentar la latencia fuera del presupuesto del producto.

