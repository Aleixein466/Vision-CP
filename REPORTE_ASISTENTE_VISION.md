# Reporte del asistente de visión local

**Fecha:** 2026-10-25. **Entorno:** Windows, Intel Core i5-10210U, 11,83 GB RAM, CPU.

## Implementación

Se instaló llama.cpp b11879, binario oficial Windows x64 CPU, y se creó una interfaz web local propia. El enlace de Google Drive compartido estaba truncado y no permitía identificar ni auditar un ZIP concreto, por lo que no se descargó contenido desconocido.

El modelo principal es Qwen2-VL 2B Instruct Q4_K_M con proyector Q8_0: 1.695.930.304 bytes combinados. La integridad se registró por SHA-256 en `datasets/manifests/vlm_models.json`. El servidor usa 4 hilos, contexto 4096, proyector en CPU y puertos limitados a loopback.

## Pruebas reales

| Flujo | Resultado | Tiempo |
|---|---|---:|
| llama.cpp directo, imagen COCO val2017 | “En la imagen aparecen un ordenador portátil y un ratón.” | 39,34 s |
| Interfaz HTTP completa | Página 200, estado `ok`, respuesta española correcta sobre portátil y ratón | 48,12 s |
| Lanzador final con clave efímera | Respuesta correcta; acceso directo sin clave rechazado con HTTP 401 | 39,76 s |

Durante la prueba, `llama-server` alcanzó aproximadamente 2.579 MB de working set y 2.054 MB de memoria privada. Las cifras son observaciones puntuales, no máximos garantizados.

SmolVLM 500M reconoció laptop/mouse en 7,96–9,85 s, pero respondió en inglés en tres intentos pese a instrucciones explícitas de español. Se conserva solo como fallback rápido documentado.

## Resultado funcional

- Cámara del navegador seleccionable.
- Captura redimensionada y comprimida localmente.
- Preguntas libres y tres prompts rápidos.
- Respuesta visual del VLM y lectura mediante voz del navegador.
- Estado visible del modelo y advertencia de incertidumbre.
- Backend sin dependencias web adicionales.
- API de llama.cpp protegida con clave efímera cuando la inicia VisionAR.
- Cierre coordinado de servidor web y proceso llama.cpp con `Ctrl+C`.

## Validación pendiente

La página y el endpoint se validaron por HTTP. La herramienta de automatización visual no pudo asociar con seguridad la URL del navegador Edge ya abierto, así que no se realizó una inspección automatizada de píxeles/DOM. El usuario puede verla directamente con `npm run dev`; la cámara requiere autorización explícita del navegador.

## Limitaciones

El VLM no opera en tiempo real en este CPU: se usa bajo demanda por captura. Puede equivocarse o inferir detalles. YOLO26 sigue siendo el motor rápido para cajas y conteo; Qwen2-VL aporta explicación contextual.

