# Registro de errores de ejecución

## 2026-10-25 — OpenCV 5 y matriz de solo lectura

- Contexto: primera prueba del nuevo overlay español mediante `npm run camera:test`.
- Error real: `cv2.putText` rechazó la matriz devuelta por `Results.plot()` porque NumPy la marcaba `readonly`.
- Causa: la ruta de renderizado con fuente Unicode de Ultralytics puede devolver una vista no modificable.
- Corrección: `result.plot(...).copy()` antes de añadir estado y conteos con OpenCV.
- Verificación: nueva ejecución de cinco frames reales completada, cámara liberada y 11,17 FPS en el último frame.

## Advertencias de DirectShow

OpenCV emite advertencias al explorar índices sin dispositivo. No impidieron abrir los índices 0 y 1. El inventario solo conserva cámaras que entregan un frame válido.

