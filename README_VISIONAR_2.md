# VisionAR AI 2.0 — YOLO26 + asistente visual local

VisionAR AI 2.0 combina dos motores reales en CPU:

- **YOLO26n** localiza, etiqueta y cuenta objetos.
- **Qwen2-VL 2B** interpreta la escena y responde preguntas en español mediante llama.cpp.

Las detecciones se convierten en evidencia estructurada y se añaden a la consulta del VLM. No se tratan como verdad absoluta: el prompt obliga a contrastarlas con la imagen e indicar incertidumbre.

## Ejecutar

```powershell
cd C:\Users\alexi\Desktop\vision\VISIONAR-AI
npm run dev
```

Abra `http://127.0.0.1:8080` si el navegador no se abre automáticamente. El inicio normal usa SmolVLM para preservar RAM durante la detección continua. Para solicitar mayor calidad visual con más consumo:

```powershell
npm run dev:qwen
```

### Si la cámara no aparece

1. Use exactamente `http://127.0.0.1:8080`; no abra `index.html` como archivo.
2. Pulse **Activar cámara**. El permiso se solicita en ese momento.
3. En Edge, pulse el icono de cámara situado junto a la dirección y seleccione **Permitir**.
4. Si había una versión anterior abierta, detenga VisionAR con `Ctrl+C`, ejecute otra vez `npm run dev` y recargue con `Ctrl+F5`.
5. Cierre Teams, Zoom, OBS u otras aplicaciones si Windows indica que la cámara está ocupada.

La interfaz muestra ahora sobre el visor la causa concreta: permiso rechazado, cámara ausente, dispositivo ocupado o selección inválida. Si el dispositivo elegido desaparece, VisionAR prueba automáticamente la cámara predeterminada.

SmolVLM es el predeterminado estable para el equipo de 12 GB: deja memoria disponible para YOLO y la cámara. Qwen2-VL ofrece mejor español, pero elevó el uso total de RAM hasta provocar errores de asignación durante sesiones largas; se conserva como opción puntual con `npm run dev:qwen`.

## Uso

1. Autorice la cámara o pulse **Abrir imagen**.
2. Active **Detección en vivo** si desea cajas YOLO periódicas. En CPU se evita solapar inferencias.
3. Pulse **Analizar escena** para capturar un fotograma nuevo.
4. Escriba preguntas adicionales para conversar sobre el mismo fotograma.
5. Use **Modelos IA** para cambiar entre los VLM instalados.
6. Use **Historial** para revisar la sesión actual. Las imágenes no se guardan.

## Detección general unificada

La interfaz tiene un solo modo **General**. Nano usa `yolo26n-objv1-150.pt`, preentrenado sobre Objects365 y capaz de producir cajas para 365 clases. La tarea Máscaras usa `yolo26n-objv1-seg.pt` con el mismo vocabulario. Persona, gafas, tenis, silla, escritorio, armario/estante, bolso, almohada, planta, marcador, borrador, televisor, peces, mascotas, moto, cubiertos, cartuchera, cinta y destornillador están entre las categorías cubiertas.

Para objetos diminutos o clases ausentes de Objects365 —por ejemplo cortaúñas, USB, billetera o tarjeta— pulse **Seleccionar zona** y **Analizar escena**. Qwen2-VL inspecciona el recorte en el mismo modo General. Colores, estampados de ropa y materiales también son atributos del análisis visual, no modos separados.

La inferencia en vivo nunca solapa solicitudes. Si llega otro fotograma mientras YOLO trabaja, se descarta con estado 429 en vez de crear hilos hasta agotar Windows. PyTorch y OpenCV usan un hilo; llama.cpp usa dos hilos y un solo contexto.

La narración puede activarse, detenerse, cambiar de voz y ajustar su velocidad. El dictado depende de la compatibilidad del navegador; la entrada de texto siempre permanece disponible.

## Detección continua y video

Al activar la cámara, **Detección continua** procesa un nuevo fotograma inmediatamente después de terminar el anterior; no utiliza un FPS ficticio ni acumula solicitudes. El panel muestra la latencia y el FPS equivalente reales. YOLO se calienta en segundo plano al iniciar VisionAR para evitar que el primer fotograma parezca bloqueado.

**Comentar cambios** añade al chat un mensaje cuando una categoria entra o sale de la escena. No repite el anuncio mientras siga presente: conserva su caja y ubicacion, tolera perdidas breves y solo confirma la salida tras tres ausencias y al menos 1,5 segundos. Cuando el catálogo local contiene información del objeto, añade una descripción breve. No ejecuta el VLM en cada fotograma: Qwen2-VL permanece reservado para preguntas y descripciones detalladas, evitando saturar la CPU.

Para revisar una billetera abierta, encierre la billetera con **Seleccionar zona** y pulse **Analizar escena**. El asistente debe enumerar solo contenidos visibles y no inferir valores, autenticidad ni datos privados.

Para analizar un video:

1. Pulse **Abrir video** y seleccione MP4/H.264, WebM, OGG o MOV compatible con Edge.
2. Reproduzca, pause o avance con los controles del visor.
3. YOLO dibuja detecciones continuamente sobre los fotogramas reproducidos.
4. Pause en cualquier instante y pulse **Analizar escena** o escriba una pregunta para enviar ese fotograma al VLM.

El archivo completo nunca se carga al backend ni se guarda: el navegador lo abre mediante una URL temporal local y envía únicamente fotogramas JPEG reducidos.

## Componentes

- `scripts/vlm_engine.py`: descubre llama.cpp, modelos instalados, inicia/detiene el proceso propio y cambia el VLM.
- `scripts/visual_fusion.py`: decodifica la imagen, ejecuta YOLO26 y produce cajas normalizadas, conteos y contexto verificable.
- `training/configs/vision_modes.yaml`: configuración del modo General y límites de recursos.
- `training/configs/unified_taxonomy.yaml`: clases propias y atributos visuales.
- `scripts/vision_web_server.py`: API local, cola VLM de una solicitud, historial efímero y límites de carga.
- `scripts/start_assistant.py`: ciclo de vida coordinado.
- `web/`: dashboard local responsive sin dependencias JavaScript.

## API local

| Ruta | Función |
|---|---|
| `GET /api/status` | Salud, modelo activo y consumo real |
| `GET /api/models` | Catálogo instalado y estado |
| `POST /api/models/select` | Cambia el VLM administrado |
| `POST /api/detect` | Inferencia YOLO sobre una imagen |
| `POST /api/analyze` | Fusión YOLO + VLM |
| `GET /api/history` | Metadatos efímeros de la sesión |

Los puertos 8080 y 8091 solo escuchan en `127.0.0.1`. llama.cpp recibe una clave aleatoria distinta en cada ejecución. VisionAR solo finaliza el proceso llama.cpp que haya creado; un servidor externo se reutiliza sin cerrarlo.

## Límites comprobados

- Qwen2-VL tarda decenas de segundos por consulta en el Intel Core i5-10210U; por eso no se ejecuta sobre todos los fotogramas.
- Objects365 amplía la cobertura a 365 clases, pero no sustituye la validación con fotografías reales del usuario. Diez clases prioritarias aún requieren datos propios o verificación del VLM.
- El equivalente de FPS de YOLO mostrado en el dashboard es `1000 / latencia de inferencia`; no representa necesariamente la frecuencia final de la cámara.
- La frecuencia final también incluye captura, compresión JPEG, HTTP local, dibujo y estabilización temporal. En la prueba disponible, la API caliente alcanzó 13,39–24,02 FPS equivalentes y el recorrido técnico de seis fotogramas promedió 65,69 ms por fotograma.
- Cancelar en la interfaz interrumpe la espera del navegador. La inferencia local ya iniciada termina de forma segura antes de aceptar otra solicitud.
- El historial está en memoria y se pierde al cerrar VisionAR.
- Biblioteca 3D y WebXR permanecen como fases posteriores. El análisis de video y Dataset Studio ya están disponibles en esta fase.

## Verificación

Con VisionAR activo:

```powershell
npm run fusion:test
```

Pruebas unitarias y validación de sintaxis:

```powershell
npm test
 npm run tracking:test
node --check web/app.js
python -m py_compile scripts/*.py
```

