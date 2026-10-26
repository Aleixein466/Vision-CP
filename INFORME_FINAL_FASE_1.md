# Informe final — Fase 1 ampliada

**Estado:** motor local de detección, evaluación, cámara y voz implementado y probado el 2026-10-25.

## Capacidades verificadas

- YOLO26n oficial cargado en PyTorch CPU.
- Evaluación sobre 100 imágenes COCO val2017 y 397 cajas reales.
- Captura de cámaras 0 y 1; inferencia real de cinco frames y liberación segura del dispositivo.
- Detección múltiple, cajas, confianza, conteo, umbral y resolución ajustables.
- Etiquetas y frases en español para las 80 clases COCO.
- Voz offline no bloqueante, silenciable con `s`, con estabilización y cooldown antirrepetición.
- Lanzador `npm run dev`, sin servidor web ni dashboard.
- Demostración end-to-end real: laptop 0,952 y mouse 0,685; frases “Veo un computador portátil” y “Veo un ratón de computador”.

## Ejecución

```powershell
cd C:\Users\alexi\Desktop\vision\VISIONAR-AI
npm run dev
```

Si aparecen varias cámaras, se solicita el índice. `npm run dev:focus` limita el modelo a las cinco clases evaluadas. `npm run demo` reproduce la prueba hablada sobre una imagen real.

## Alcance y limitaciones

El motor puede reconocer las 80 clases COCO, no objetos arbitrarios ni marca/modelo exactos. La voz anuncia una clase cuando aparece estable, pero no convierte una detección genérica en especificaciones de producto. Las métricas medidas siguen siendo las del reporte Fase 1: mAP50-95 global 0,331; book es la principal debilidad. No se entrenó un modelo personalizado.

El resultado cumple el motor base solicitado. La interfaz web completa, realidad aumentada y entrenamiento especializado continúan fuera de esta fase.

