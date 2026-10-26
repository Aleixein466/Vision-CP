# Plan de entrenamiento

## Puerta de entrada

No entrenar hasta: (1) licencia por archivo aceptada, (2) auditoría sin errores críticos, (3) mínimo de datos de campo definido por clase, (4) test bloqueado, (5) baseline medido y (6) autorización explícita después de estimar duración.

## Estrategia

1. Mantener YOLO26n COCO si el test propio cumple objetivos.
2. Si fallan clases concretas, ampliar solo esas clases con fotografías propias u Open Images filtrado y licencia revisada.
3. Usar pesos `yolo26n.pt`, no inicialización aleatoria. Congelar el test y agrupar por sesión/video antes del split.
4. Primera corrida: 1 época de calibración a 640 px, batch 2 o 4, CPU, para medir tiempo/RAM. Extrapolar antes de autorizar 20–50 épocas.
5. Early stopping (`patience=10`), `seed=2601`, checkpoints periódicos, métricas/curvas por clase y comparación exacta con baseline.

Comando de calibración propuesto (no ejecutado):

```powershell
yolo detect train model=training/checkpoints/yolo26n.pt data=CONFIGURACION_YOLO.yaml epochs=1 imgsz=640 batch=2 device=cpu workers=0 seed=2601 deterministic=True project=training/metrics/runs name=calibration_1epoch
```

No se estima responsablemente la duración de entrenamiento a partir de inferencia: la calibración es obligatoria. Si la proyección supera 30 minutos en este portátil, solicitar confirmación y preferir GPU. En GPU, ejecutar el mismo entorno fijado y cambiar `device=0`; registrar modelo de GPU, VRAM, CUDA, versión de PyTorch, consumo máximo, tiempo por época y costo del proveedor.

## Evaluación comparativa

Registrar precision, recall, mAP50, mAP50-95, AP por clase, matriz de confusión, latencia, FPS y RAM. Aceptar el fine-tune solo si mejora el conjunto bloqueado sin degradación material de clases, sobreajuste evidente o costo incompatible. YOLOE-26n queda limitado a exploración/pseudoetiquetado humano; su fine-tuning documentado requiere GPU.

