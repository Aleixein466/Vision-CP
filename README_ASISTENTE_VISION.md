# VisionAR AI — asistente visual local

Esta ampliación reproduce el flujo del ejemplo indicado: cámara web → modelo visión-lenguaje local → respuesta → lectura por voz. No depende de una API de IA en la nube.

## Inicio

```powershell
cd C:\Users\alexi\Desktop\vision\VISIONAR-AI
npm run dev
```

Se abrirá `http://127.0.0.1:8080`. En la primera ejecución, `llama.cpp` descarga aproximadamente 1,70 GB para Qwen2-VL 2B Q4_K_M y su proyector visual. Las ejecuciones posteriores reutilizan la caché de Hugging Face.

En la página:

1. Permita acceso a la cámara.
2. Seleccione la cámara.
3. Pulse **Activar cámara**.
4. Escriba una pregunta o use una pregunta rápida.
5. Pulse **Capturar y analizar**.
6. La respuesta se muestra y, si está habilitado, el navegador la lee en español.

Detenga los servidores con `Ctrl+C` en la terminal.

## Arquitectura

- `llama-server.exe`: binario oficial Windows x64 CPU de llama.cpp b11879.
- Modelo principal: `ggml-org/Qwen2-VL-2B-Instruct-GGUF:Q4_K_M`.
- Backend: `scripts/start_assistant.py` y `scripts/vision_web_server.py`, solo biblioteca estándar de Python.
- Interfaz: `web/`, sin dependencias JavaScript ni servicios externos.
- Voz: Web Speech API del navegador para respuestas del VLM; pyttsx3/SAPI para el detector YOLO nativo.
- Detector rápido alternativo: `npm run camera` abre YOLO26n en OpenCV y anuncia clases en tiempo real.

La imagen capturada se reduce a un máximo de 960 px, se convierte a JPEG y viaja únicamente entre el navegador local, el proxy Python en el puerto 8080 y llama.cpp en el puerto 8091. Ambos escuchan exclusivamente en `127.0.0.1`. El proceso lanzador genera una clave efímera para proteger la API de llama.cpp y no la expone al navegador.

## Comandos

```powershell
npm run dev          # asistente web VLM
npm run camera       # detector YOLO en tiempo real con voz
npm run dev:focus    # YOLO limitado a cinco clases de Fase 1
npm run demo         # reconocimiento YOLO hablado sobre imagen real
npm run speech:test  # prueba de voz SAPI
npm run test         # pruebas automatizadas
```

`npm run vlm:test` requiere que llama-server ya esté activo en el puerto 8091.

## Modelos y licencias

- llama.cpp: distribución oficial desde `https://github.com/ggml-org/llama.cpp/releases`, licencia MIT del proyecto.
- Qwen2-VL 2B GGUF: `https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF`, modelo derivado de Qwen2-VL; revisar Apache-2.0 y la ficha antes de redistribuir.
- SmolVLM 500M se probó como fallback de menor consumo: reconoció los objetos, pero no respetó de forma fiable la salida en español en esta máquina.

## Limitaciones

- Qwen2-VL describe escenas, pero puede alucinar o confundir detalles. No usar sus respuestas como certeza técnica o de seguridad.
- El análisis es por captura, no por cada frame; esto mantiene el uso de CPU razonable.
- YOLO sigue siendo la fuente apropiada para cajas, conteos y seguimiento rápido. El VLM aporta descripción y preguntas abiertas.
- El modelo no conoce necesariamente marcas, referencias o especificaciones exactas de un objeto visible.
- La velocidad depende del tamaño de la imagen y de la longitud de la respuesta.

