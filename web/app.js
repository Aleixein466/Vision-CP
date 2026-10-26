const $ = (selector) => document.querySelector(selector);
const video = $('#video');
const uploadedImage = $('#uploadedImage');
const captureCanvas = $('#captureCanvas');
const overlay = $('#overlay');
const overlayContext = overlay.getContext('2d');
const chat = $('#chat');
const promptBox = $('#prompt');

let stream = null;
let imageMode = false;
let videoFileMode = false;
let videoObjectUrl = null;
let currentFrame = null;
let currentFrameId = null;
let conversation = [];
let requestController = null;
let liveBusy = false;
let liveTimer = null;
let liveGeneration = 0;
let lastDetections = [];
const sceneTracker = new SceneTracker({ confirmFrames: 2, exitFrames: 3, exitMilliseconds: 1500 });
let yoloCatalog = [];
let selectingRegion = false;
let focusRegion = null;
let dragStart = null;
let captureGroup = crypto.randomUUID();

function escapeText(value) {
  const element = document.createElement('span');
  element.textContent = value;
  return element.innerHTML;
}

function setReady(ready, text = '') {
  $('#statusDot').className = `dot ${ready ? 'ready' : 'error'}`;
  $('#statusText').textContent = text || (ready ? 'Sistemas listos' : 'Modelo no disponible');
}

function setCameraState(title, detail, isError = false) {
  $('#cameraEmpty').hidden = false;
  $('#cameraEmptyTitle').textContent = title;
  $('#cameraEmptyDetail').textContent = detail;
  $('#cameraEmpty').classList.toggle('camera-error', isError);
}

function cameraErrorMessage(error) {
  if (error?.name === 'NotAllowedError' || error?.name === 'PermissionDeniedError') {
    return 'Edge bloqueó la cámara. Pulsa el icono de cámara junto a la dirección, selecciona Permitir y vuelve a intentarlo.';
  }
  if (error?.name === 'NotFoundError' || error?.name === 'DevicesNotFoundError') {
    return 'Windows no encontró una cámara disponible. Verifica que esté conectada y que otra aplicación no la esté usando.';
  }
  if (error?.name === 'NotReadableError' || error?.name === 'TrackStartError') {
    return 'La cámara está ocupada o Windows impidió abrirla. Cierra otras aplicaciones que estén usando video.';
  }
  if (error?.name === 'OverconstrainedError') {
    return 'La cámara seleccionada ya no está disponible. Se intentará usar la cámara predeterminada.';
  }
  return error?.message || 'No se pudo iniciar la cámara.';
}

async function refreshStatus() {
  try {
    const response = await fetch('/api/status', { cache: 'no-store' });
    const data = await response.json();
    setReady(data.ready, data.training ? 'Entrenamiento activo' : data.ready ? (data.vlm_ready ? 'Sistemas listos' : 'YOLO listo · VLM inactivo') : 'YOLO en espera');
    const yoloName = data.yolo_model ? `YOLO26${data.yolo_model.scale}${data.yolo_model.task === 'segment' ? '-seg' : ''}` : 'YOLO26';
    $('#activeModel').textContent = `${yoloName} + ${data.active_model || 'VLM externo'}`;
    $('#vlmMetric').textContent = !data.vlm_ready ? 'Inactivo' : data.active_model === 'smolvlm-500m' ? 'SmolVLM 500M' : data.active_model === 'qwen2-vl-2b' ? 'Qwen2-VL 2B' : 'Servidor externo';
    if (data.metrics) {
      $('#cpuMetric').textContent = `${Number(data.metrics.cpu_percent).toFixed(0)}%`;
      $('#ramMetric').textContent = `${Number(data.metrics.ram_percent).toFixed(0)}%`;
      $('#llamaRam').textContent = data.metrics.llama_process ? `VLM ${data.metrics.llama_process.ram_mb} MB` : 'Servidor externo';
    }
  } catch (error) {
    setReady(false, 'Servidor desconectado');
  }
}

async function listCameras() {
  if (!navigator.mediaDevices?.getUserMedia) {
    $('#cameraSelect').innerHTML = '<option value="">Cámara no compatible</option>';
    setCameraState('Cámara no disponible', 'Abra VisionAR desde http://127.0.0.1:8080 en un navegador compatible.', true);
    return;
  }
  try {
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((item) => item.kind === 'videoinput');
    $('#cameraSelect').innerHTML = '';
    if (!devices.length) {
      $('#cameraSelect').innerHTML = '<option value="">Cámara predeterminada</option>';
      return;
    }
    devices.forEach((device, index) => {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.textContent = device.label || `Cámara ${index + 1}`;
      option.dataset.camera = 'true';
      $('#cameraSelect').append(option);
    });
  } catch (error) {
    $('#cameraSelect').innerHTML = '<option value="">Cámara predeterminada</option>';
    setCameraState('No se pudo consultar la cámara', cameraErrorMessage(error), true);
  }
}

async function startCamera() {
  captureGroup = crypto.randomUUID();
  stopCamera();
  clearVideoFile();
  sceneTracker.clear();
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Este navegador no ofrece acceso a la cámara.');
  const selectedOption = $('#cameraSelect').selectedOptions[0];
  const deviceId = selectedOption?.dataset.camera === 'true' ? selectedOption.value : '';
  const baseVideo = { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: { ideal: 'environment' } };
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: deviceId ? { ...baseVideo, deviceId: { exact: deviceId } } : baseVideo,
      audio: false,
    });
  } catch (error) {
    if (error.name === 'OverconstrainedError' && deviceId) {
      stream = await navigator.mediaDevices.getUserMedia({ video: baseVideo, audio: false });
    } else {
      setCameraState('No se pudo activar la cámara', cameraErrorMessage(error), true);
      throw error;
    }
  }
  video.srcObject = stream;
  await video.play();
  imageMode = false;
  videoFileMode = false;
  video.controls = false;
  uploadedImage.classList.remove('visible');
  video.classList.add('visible');
  $('#cameraEmpty').hidden = true;
  $('#cameraEmpty').classList.remove('camera-error');
  $('#capture').disabled = false;
  $('#askButton').disabled = false;
  $('#selectRegion').disabled = false;
  $('#resolutionBadge').textContent = `${video.videoWidth} × ${video.videoHeight}`;
  currentFrame = null;
  drawDetections([]);
  await listCameras();
  if ($('#liveDetection').checked) toggleLiveDetection(true);
}

function stopCamera() {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  if (video.srcObject) video.srcObject = null;
}

function clearVideoFile() {
  if (!videoFileMode && !videoObjectUrl) return;
  video.pause();
  video.removeAttribute('src');
  video.load();
  videoFileMode = false;
  video.controls = false;
  if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
  videoObjectUrl = null;
}

function loadImageFile(file) {
  if (!file || !['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) throw new Error('Use una imagen JPEG, PNG o WebP');
  if (file.size > 8 * 1024 * 1024) throw new Error('La imagen supera el límite de 8 MB');
  stopCamera();
  clearVideoFile();
  sceneTracker.clear();
  const reader = new FileReader();
  reader.onload = () => {
    uploadedImage.onload = () => {
      imageMode = true;
      uploadedImage.classList.add('visible');
      video.classList.remove('visible');
      $('#cameraEmpty').hidden = true;
      $('#capture').disabled = false;
      $('#askButton').disabled = false;
      $('#selectRegion').disabled = false;
      $('#resolutionBadge').textContent = `${uploadedImage.naturalWidth} × ${uploadedImage.naturalHeight}`;
      currentFrame = null;
      captureFrame();
      drawDetections([]);
      detectFrame();
    };
    uploadedImage.src = reader.result;
  };
  reader.readAsDataURL(file);
}

function loadVideoFile(file) {
  captureGroup = crypto.randomUUID();
  if (!file || !file.type.startsWith('video/')) throw new Error('Use un archivo de video compatible con el navegador');
  if (file.size > 1024 * 1024 * 1024) throw new Error('El video supera el límite local de 1 GB');
  stopCamera();
  clearVideoFile();
  sceneTracker.clear();
  imageMode = false;
  videoFileMode = true;
  uploadedImage.classList.remove('visible');
  videoObjectUrl = URL.createObjectURL(file);
  video.srcObject = null;
  video.src = videoObjectUrl;
  video.controls = true;
  video.muted = true;
  video.onloadedmetadata = async () => {
    video.classList.add('visible');
    $('#cameraEmpty').hidden = true;
    $('#capture').disabled = false;
    $('#askButton').disabled = false;
    $('#selectRegion').disabled = false;
    $('#resolutionBadge').textContent = `${video.videoWidth} × ${video.videoHeight}`;
    currentFrame = null;
    drawDetections([]);
    try { await video.play(); } catch { /* El usuario puede iniciar el video con sus controles. */ }
    if ($('#liveDetection').checked) toggleLiveDetection(true);
  };
  video.onerror = () => setCameraState('No se pudo abrir el video', 'Use MP4/H.264 o WebM, formatos compatibles con Edge.', true);
}

function captureFrame(maxSize = null) {
  const source = imageMode ? uploadedImage : video;
  const width = imageMode ? source.naturalWidth : source.videoWidth;
  const height = imageMode ? source.naturalHeight : source.videoHeight;
  if (!width || !height) throw new Error('No hay una imagen disponible');
  const requestedSize = maxSize || ($('#visionMode').value === 'general' ? 960 : 1280);
  const scale = Math.min(1, requestedSize / Math.max(width, height));
  captureCanvas.width = Math.round(width * scale);
  captureCanvas.height = Math.round(height * scale);
  captureCanvas.getContext('2d').drawImage(source, 0, 0, captureCanvas.width, captureCanvas.height);
  currentFrame = captureCanvas.toDataURL('image/jpeg', 0.82);
  currentFrameId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
  return currentFrame;
}

function sourceDimensions() {
  return imageMode
    ? { width: uploadedImage.naturalWidth, height: uploadedImage.naturalHeight }
    : { width: video.videoWidth, height: video.videoHeight };
}

function drawDetections(detections) {
  lastDetections = detections || [];
  const stage = $('#stage');
  const width = stage.clientWidth;
  const height = stage.clientHeight;
  const source = sourceDimensions();
  const dpr = window.devicePixelRatio || 1;
  overlay.width = width * dpr;
  overlay.height = height * dpr;
  overlay.style.width = `${width}px`;
  overlay.style.height = `${height}px`;
  overlayContext.setTransform(dpr, 0, 0, dpr, 0, 0);
  overlayContext.clearRect(0, 0, width, height);
  if (!source.width || !source.height) return;
  const scale = Math.min(width / source.width, height / source.height);
  const renderedWidth = source.width * scale;
  const renderedHeight = source.height * scale;
  const offsetX = (width - renderedWidth) / 2;
  const offsetY = (height - renderedHeight) / 2;
  overlayContext.font = '600 12px Segoe UI';
  detections.forEach((item) => {
    const x = offsetX + item.box.x * renderedWidth;
    const y = offsetY + item.box.y * renderedHeight;
    const boxWidth = item.box.w * renderedWidth;
    const boxHeight = item.box.h * renderedHeight;
    const label = `${item.label} ${Math.round(item.confidence * 100)}%${item.stale ? ' · seguimiento' : ''}`;
    if (item.polygon?.length >= 3) {
      overlayContext.beginPath();
      item.polygon.forEach(([px, py], index) => {
        const pointX = offsetX + px * renderedWidth;
        const pointY = offsetY + py * renderedHeight;
        if (index === 0) overlayContext.moveTo(pointX, pointY); else overlayContext.lineTo(pointX, pointY);
      });
      overlayContext.closePath();
      overlayContext.fillStyle = item.stale ? '#93aaa522' : '#72f2c344';
      overlayContext.strokeStyle = item.stale ? '#93aaa5' : '#c9ff69';
      overlayContext.lineWidth = 2;
      overlayContext.fill();
      overlayContext.stroke();
    }
    overlayContext.strokeStyle = item.stale ? '#93aaa5' : '#72f2c3';
    overlayContext.setLineDash(item.stale ? [6, 5] : []);
    overlayContext.lineWidth = 2;
    overlayContext.strokeRect(x, y, boxWidth, boxHeight);
    const labelWidth = overlayContext.measureText(label).width + 14;
    overlayContext.fillStyle = item.stale ? '#93aaa5' : '#72f2c3';
    overlayContext.fillRect(x, Math.max(0, y - 25), labelWidth, 25);
    overlayContext.fillStyle = '#07110f';
    overlayContext.fillText(label, x + 7, Math.max(17, y - 8));
    overlayContext.setLineDash([]);
  });
  if (focusRegion) {
    const x = offsetX + focusRegion.x * renderedWidth;
    const y = offsetY + focusRegion.y * renderedHeight;
    overlayContext.fillStyle = '#c9ff6922';
    overlayContext.strokeStyle = '#c9ff69';
    overlayContext.setLineDash([7, 5]);
    overlayContext.lineWidth = 2;
    overlayContext.fillRect(x, y, focusRegion.w * renderedWidth, focusRegion.h * renderedHeight);
    overlayContext.strokeRect(x, y, focusRegion.w * renderedWidth, focusRegion.h * renderedHeight);
    overlayContext.setLineDash([]);
  }
}

function pointToNormalized(event) {
  const rect = $('#stage').getBoundingClientRect();
  const source = sourceDimensions();
  if (!source.width || !source.height) return null;
  const scale = Math.min(rect.width / source.width, rect.height / source.height);
  const renderedWidth = source.width * scale;
  const renderedHeight = source.height * scale;
  const offsetX = (rect.width - renderedWidth) / 2;
  const offsetY = (rect.height - renderedHeight) / 2;
  const x = Math.max(0, Math.min(1, (event.clientX - rect.left - offsetX) / renderedWidth));
  const y = Math.max(0, Math.min(1, (event.clientY - rect.top - offsetY) / renderedHeight));
  return { x, y };
}

function finishRegionSelection() {
  selectingRegion = false;
  dragStart = null;
  overlay.classList.remove('selecting');
  $('#selectRegion').textContent = focusRegion ? 'Cambiar zona' : 'Seleccionar zona';
  $('#selectRegion').classList.remove('active');
  $('#clearRegion').hidden = !focusRegion;
  drawDetections(lastDetections);
}

function renderObjects(data, stabilize = false) {
  if (stabilize) {
    data = updateSceneTracks(data);
  }
  const container = $('#objectChips');
  container.innerHTML = '';
  const detections = data.detections || [];
  if (!detections.length) {
    container.innerHTML = '<span class="muted">YOLO no detectó objetos sobre el umbral; esto no implica que la escena esté vacía.</span>';
  } else {
    detections.forEach((item) => {
      const chip = document.createElement('button');
      chip.className = 'object-chip';
      chip.textContent = `${item.id} · ${item.label} ${Math.round(item.confidence * 100)}%`;
      chip.title = 'Seleccionar este objeto para preguntarle al VLM';
      chip.addEventListener('click', () => {
        focusRegion = { ...item.box };
        $('#clearRegion').hidden = false;
        $('#selectRegion').textContent = 'Cambiar zona';
        promptBox.value = `Describe con detalle el objeto seleccionado (${item.label}). ¿Qué es y qué rasgos visibles tiene?`;
        try { captureFrame(); } catch (error) { showSystemMessage(error.message); }
        drawDetections(lastDetections);
      });
      container.append(chip);
    });
  }
  drawDetections(data.detections || []);
  if (data.latency_ms || data.yolo?.latency_ms) {
    const latency = data.latency_ms ?? data.yolo.latency_ms;
    const fps = data.fps_equivalent ?? data.yolo.fps_equivalent;
    $('#yoloLatency').textContent = `${latency.toFixed(0)} ms por inferencia`;
    $('#fpsBadge').textContent = `YOLO ${fps} FPS eq.`;
    $('#yoloMetric').textContent = `${(data.detections || []).length} objetos`;
  }
  if (stabilize) announceSceneEvents(data.events || []);
}

function countPhrase(counts) {
  return Object.entries(counts).map(([label, count]) => `${count} × ${label}`).join(', ');
}

function objectLocation(box) {
  const centerX = box.x + box.w / 2;
  const centerY = box.y + box.h / 2;
  const horizontal = centerX < .34 ? 'a la izquierda' : centerX > .66 ? 'a la derecha' : 'en el centro';
  const vertical = centerY < .34 ? 'arriba' : centerY > .66 ? 'abajo' : 'a media altura';
  return `${horizontal}, ${vertical}`;
}

function updateSceneTracks(data) {
  return sceneTracker.update(data);
}

function announceSceneEvents(events) {
  if (!$('#autoComment').checked) return;
  events.forEach((event) => {
    const first = event.detections[0];
    if (!first) return;
    let message;
    if (event.type === 'enter') {
      message = `Veo ${countPhrase({ [first.label]: event.detections.length })} ${objectLocation(first.box)}.`;
      if (first.description) message += ` ${first.description}`;
    } else {
      message = `El objeto ${first.label} salió de la escena.`;
    }
    appendMessage('assistant', message, event.type === 'enter' ? 'Objeto identificado y ubicado' : 'Salida confirmada de la escena');
    speakAnswer(message);
  });
}

async function detectFrame() {
  const activeVideo = Boolean(stream || videoFileMode);
  if (liveBusy || imageMode === false && !activeVideo || activeVideo && video.readyState < 2) return false;
  if (videoFileMode && video.paused) return false;
  liveBusy = true;
  try {
    const image = captureFrame(imageMode ? 960 : 512);
    const confidence = imageMode ? 0.18 : Number(localStorage.getItem('visionar.confidence') || 0.32);
    const imgsz = imageMode ? 640 : 320;
    const response = await fetch('/api/detect', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image, confidence, imgsz, mode: $('#visionMode').value,
        scale: $('#yoloScale').value, task: $('#yoloTask').value,
        focus: focusRegion,
      }),
    });
    const data = await response.json();
    // Una foto solo produce una inferencia: no debe esperar dos fotogramas para dibujar cajas.
    if (response.ok) renderObjects(data, !imageMode);
    return response.ok;
  } finally {
    liveBusy = false;
  }
}

function toggleLiveDetection(enabled) {
  liveGeneration += 1;
  clearTimeout(liveTimer);
  liveTimer = null;
  if (!enabled) return;
  const generation = liveGeneration;
  const run = async () => {
    if (generation !== liveGeneration || !$('#liveDetection').checked) return;
    await detectFrame().catch((error) => {
      if (!String(error.message).includes('429')) console.warn('Detección continua:', error);
    });
    if (imageMode) return;
    const delay = Math.max(100,Number(localStorage.getItem('visionar.interval') || 250));
    if (generation === liveGeneration && $('#liveDetection').checked) liveTimer = setTimeout(run, delay);
  };
  run();
}

function appendMessage(role, text, meta = '') {
  const row = document.createElement('div');
  row.className = `message ${role}`;
  const avatar = document.createElement('span');
  avatar.textContent = role === 'assistant' ? 'VA' : 'TÚ';
  const body = document.createElement('div');
  const paragraph = document.createElement('p');
  paragraph.textContent = text;
  body.append(paragraph);
  if (meta) {
    const small = document.createElement('small');
    small.textContent = meta;
    body.append(small);
  }
  row.append(avatar, body);
  chat.append(row);
  chat.scrollTop = chat.scrollHeight;
  return row;
}

function showSystemMessage(text) {
  appendMessage('assistant', text, 'Aviso del sistema');
}

function speakAnswer(text) {
  if (!$('#speak').checked || !('speechSynthesis' in window)) return;
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'es-CO';
  utterance.rate = Number($('#voiceRate').value);
  const voices = speechSynthesis.getVoices();
  const selected = voices.find((voice) => voice.name === $('#voiceSelect').value);
  if (selected) utterance.voice = selected;
  speechSynthesis.speak(utterance);
}

function populateVoices() {
  if (!('speechSynthesis' in window)) return;
  const voices = speechSynthesis.getVoices();
  const spanish = voices.filter((voice) => voice.lang.toLowerCase().startsWith('es'));
  $('#voiceSelect').innerHTML = '';
  (spanish.length ? spanish : voices).forEach((voice) => {
    const option = document.createElement('option');
    option.value = voice.name;
    option.textContent = `${voice.name} (${voice.lang})`;
    $('#voiceSelect').append(option);
  });
}

async function askScene(forceCapture = false) {
  const prompt = promptBox.value.trim();
  if (!prompt) return;
  try {
    if (forceCapture || !currentFrame) captureFrame();
  } catch (error) {
    showSystemMessage(error.message);
    return;
  }
  const mode = $('#visionMode').value;
  appendMessage('user', prompt, `Fotograma ${currentFrameId.slice(0, 8)} · modo ${mode}${focusRegion ? ' · zona seleccionada' : ''}`);
  conversation.push({ role: 'user', content: prompt });
  const loading = appendMessage('assistant', 'Analizando con YOLO26 y el modelo visual…', 'Inferencia local en CPU');
  $('#askButton').disabled = true;
  $('#capture').disabled = true;
  $('#cancelButton').hidden = false;
  requestController = new AbortController();
  const liveWasEnabled = $('#liveDetection').checked;
  if (liveWasEnabled) toggleLiveDetection(false);
  const started = performance.now();
  try {
    const response = await fetch('/api/analyze', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: requestController.signal,
      body: JSON.stringify({
        image: currentFrame, prompt, frame_id: currentFrameId,
        history: conversation.slice(-7, -1), confidence: imageMode ? 0.18 : 0.30,
        imgsz: imageMode || focusRegion ? 640 : 480, mode, focus: focusRegion,
        scale: $('#yoloScale').value, task: $('#yoloTask').value,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'No se pudo analizar la escena');
    loading.remove();
    conversation.push({ role: 'assistant', content: data.answer });
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    appendMessage('assistant', data.answer, `${seconds} s · ${(data.detections || []).length} evidencias YOLO · ${data.usage?.completion_tokens ?? '—'} tokens`);
    renderObjects(data);
    $('#vlmLatency').textContent = `${seconds} s última respuesta`;
    speakAnswer(data.answer);
  } catch (error) {
    loading.remove();
    showSystemMessage(error.name === 'AbortError' ? 'La espera se canceló en esta interfaz. El servidor finalizará de forma segura la inferencia en curso.' : error.message);
  } finally {
    requestController = null;
    $('#askButton').disabled = false;
    $('#capture').disabled = false;
    $('#cancelButton').hidden = true;
    if (liveWasEnabled && (stream || videoFileMode)) toggleLiveDetection(true);
  }
}

async function loadModels() {
  const response = await fetch('/api/models');
  const data = await response.json();
  const grid = $('#modelGrid');
  grid.innerHTML = '';
  data.models.forEach((model) => {
    const card = document.createElement('article');
    card.className = `model-card panel ${model.active ? 'selected' : ''}`;
    const size = model.size_bytes ? `${(model.size_bytes / 1024 / 1024 / 1024).toFixed(2)} GB` : 'Tamaño no registrado';
    card.innerHTML = `<div class="model-icon">◇</div><div class="model-copy"><div><span class="model-state">${model.active ? 'ACTIVO' : model.installed ? 'INSTALADO' : 'NO DESCARGADO'}</span><h3>${escapeText(model.name)}</h3></div><p>${escapeText(model.note)}</p><small>${escapeText(model.repository)} · ${size} · entrada visual verificada</small></div>`;
    const button = document.createElement('button');
    button.className = model.active ? 'button secondary' : 'button primary';
    button.textContent = model.active ? 'En uso' : model.installed ? 'Usar modelo' : 'No disponible';
    button.disabled = model.active || !model.installed;
    button.addEventListener('click', () => switchModel(model.id, button));
    card.append(button);
    grid.append(card);
  });
}

async function loadYoloModels() {
  const response = await fetch('/api/yolo/models', { cache: 'no-store' });
  const data = await response.json();
  yoloCatalog = data.models || [];
  $('#yoloScale').value = data.selected?.scale || 'n';
  $('#yoloTask').value = data.selected?.task || 'detect';
  document.querySelectorAll('#yoloScale option').forEach((option) => {
    const entries = yoloCatalog.filter((item) => item.scale === option.value && item.task === $('#yoloTask').value);
    const installed = entries.some((item) => item.installed);
    const vocabulary = entries[0]?.vocabulary || '';
    const classCount = vocabulary.includes('365') ? '365 clases' : '80 clases';
    option.textContent = `${option.value === 'n' ? 'Nano' : option.value === 's' ? 'Small' : option.value === 'm' ? 'Medium' : option.value === 'l' ? 'Large' : 'Extra Large'} · ${classCount}${installed ? ' · instalado' : ' · descargar'}`;
    option.title = vocabulary;
  });
}

async function selectYoloModel() {
  const scale = $('#yoloScale').value;
  const task = $('#yoloTask').value;
  let entry = yoloCatalog.find((item) => item.scale === scale && item.task === task);
  if (!entry) return;
  let download = false;
  if (!entry.installed) {
    download = window.confirm(`${entry.filename} no está instalado. ¿Deseas descargarlo ahora desde Ultralytics?\n\n${entry.cpu_guidance}`);
    if (!download) {
      await loadYoloModels();
      return;
    }
  }
  const liveWasEnabled = $('#liveDetection').checked;
  toggleLiveDetection(false);
  $('#yoloScale').disabled = true;
  $('#yoloTask').disabled = true;
  setReady(true, `Cargando ${entry.filename}`);
  try {
    const response = await fetch('/api/yolo/select', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scale, task, download }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'No se pudo cambiar el modelo YOLO26');
    yoloCatalog = data.models;
    sceneTracker.clear();
    lastDetections = [];
    drawDetections([]);
    showSystemMessage(`Modelo activo: ${data.selected.filename}. ${data.selected.cpu_guidance}`);
    await refreshStatus();
  } catch (error) {
    showSystemMessage(error.message);
  } finally {
    $('#yoloScale').disabled = false;
    $('#yoloTask').disabled = false;
    await loadYoloModels().catch(() => {});
    if (liveWasEnabled && (stream || videoFileMode)) toggleLiveDetection(true);
  }
}

async function switchModel(model, button) {
  button.disabled = true;
  button.textContent = 'Cargando…';
  try {
    const response = await fetch('/api/models/select', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'No se pudo cambiar el modelo');
    await loadModels();
    await refreshStatus();
  } catch (error) {
    showSystemMessage(error.message);
    button.disabled = false;
    button.textContent = 'Reintentar';
  }
}

async function loadHistory() {
  const response = await fetch('/api/history');
  const data = await response.json();
  const list = $('#historyList');
  list.innerHTML = '';
  if (!data.items.length) {
    list.innerHTML = '<div class="panel empty-list">Todavía no hay análisis en esta sesión.</div>';
    return;
  }
  data.items.filter((item) => item.status !== 'rejected').forEach((item) => {
    const card = document.createElement('article');
    card.className = 'history-card panel';
    const date = new Date(item.timestamp).toLocaleTimeString('es-CO', { hour: '2-digit', minute: '2-digit' });
    card.innerHTML = `<div><span>${date} · ${item.elapsed_seconds} s</span><strong>${escapeText(item.prompt)}</strong><p>${escapeText(item.answer)}</p></div><small>Fotograma ${escapeText(item.frame_id.slice(0, 8))}</small>`;
    list.append(card);
  });
}

let activeCandidate = null;
let candidateBox = null;
let candidateDrag = null;

async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.audit ? `Dataset no apto: ${JSON.stringify(data.audit.issues)}` : data.error || data.reason || `Error HTTP ${response.status}`);
  return data;
}

function fileToDatasetImage(file) {
  return new Promise((resolve, reject) => {
    if (!file?.type?.startsWith('image/')) return reject(new Error(`${file?.name || 'archivo'} no es una imagen`));
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(`No se pudo leer ${file.name}`));
    reader.onload = () => {
      const image = new Image();
      image.onerror = () => reject(new Error(`${file.name} está corrupta o usa un formato no compatible`));
      image.onload = () => {
        const scale = Math.min(1, 1600 / Math.max(image.naturalWidth, image.naturalHeight));
        const canvas = document.createElement('canvas');
        canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
        canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
        canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL('image/jpeg', .94));
      };
      image.src = reader.result;
    };
    reader.readAsDataURL(file);
  });
}

async function importDatasetFiles(files) {
  const progress = $('#datasetImportProgress');
  const source = $('#datasetSource').value.trim();
  const license = $('#datasetLicense').value.trim();
  if (!source || !license) throw new Error('Procedencia y licencia son obligatorias');
  let imported = 0; let duplicates = 0; let failed = 0;
  const images = [...files].filter((file) => file.type.startsWith('image/'));
  for (let index = 0; index < images.length; index += 1) {
    progress.textContent = `Importando ${index + 1}/${images.length}: ${images[index].name}`;
    try {
      const image = await fileToDatasetImage(images[index]);
      const data = await apiJson('/api/dataset/import', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image, filename: images[index].webkitRelativePath || images[index].name, source, license }) });
      if (data.result.imported) imported += 1; else duplicates += 1;
    } catch (error) { failed += 1; progress.textContent = error.message; }
  }
  progress.textContent = `${imported} importadas · ${duplicates} duplicadas omitidas · ${failed} con error`;
  progress.className = `dataset-message ${failed ? 'error' : 'success'}`;
  await Promise.all([loadDatasetStats(), loadCandidateQueue()]);
}

async function importCurrentFrame() {
  if (stream || videoFileMode || imageMode) captureFrame(1600);
  if (!currentFrame) throw new Error('No hay una escena activa; abre la cámara, una imagen o un video');
  const source = stream ? 'web_camera' : videoFileMode ? 'local_video_frame' : 'local_image';
  $('#datasetSource').value = source;
  const data = await apiJson('/api/dataset/import', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: currentFrame, filename: `${source}-${Date.now()}.jpg`, source, license: $('#datasetLicense').value.trim(), group_id: stream || videoFileMode ? captureGroup : '' }) });
  $('#datasetImportProgress').textContent = data.result.imported ? 'Fotograma añadido a revisión.' : 'El fotograma ya existía y no se duplicó.';
  await Promise.all([loadDatasetStats(), loadCandidateQueue()]);
}

async function importPublicUrl() {
  const url = $('#datasetPublicUrl').value.trim();
  const license = $('#datasetLicense').value.trim();
  if (!url || !license) throw new Error('Indica la URL directa y su licencia verificable');
  const data = await apiJson('/api/dataset/import-url', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, source: url, license }) });
  $('#datasetImportProgress').textContent = data.result.imported ? 'Imagen pública añadida a revisión.' : 'La imagen pública ya existía.';
  await Promise.all([loadDatasetStats(), loadCandidateQueue()]);
}

async function loadCandidateQueue() {
  const data = await apiJson('/api/dataset/candidates', { cache: 'no-store' });
  const list = $('#candidateList'); const selected = list.value;
  list.innerHTML = '<option value="">Selecciona un candidato pendiente</option>';
  data.items.forEach((item) => {
    const option = document.createElement('option'); option.value = item.id;
    const targets = (item.intended_classes || []).join(', ');
    option.textContent = `${item.status === 'reviewed' ? '✓' : '•'} ${item.filename}${targets ? ` · ${targets}` : ''}${item.near_duplicate_of ? ' · posible duplicado' : ''}`; list.append(option);
  });
  if (data.items.some((item) => item.id === selected && item.status !== 'rejected')) list.value = selected;
}

function candidateImageRect() {
  const image = $('#candidateImage'); const canvas = $('#candidateCanvas');
  const scale = Math.min(canvas.clientWidth / image.naturalWidth, canvas.clientHeight / image.naturalHeight);
  const width = image.naturalWidth * scale; const height = image.naturalHeight * scale;
  return { x: (canvas.clientWidth - width) / 2, y: (canvas.clientHeight - height) / 2, width, height };
}

function drawCandidateBox() {
  const canvas = $('#candidateCanvas'); const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth * ratio); canvas.height = Math.round(canvas.clientHeight * ratio);
  const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio);
  if (!candidateBox || !activeCandidate) return;
  const rect = candidateImageRect();
  ctx.strokeStyle = '#72f2c3'; ctx.lineWidth = 3; ctx.fillStyle = '#72f2c322';
  const x = rect.x + candidateBox.x * rect.width; const y = rect.y + candidateBox.y * rect.height;
  const w = candidateBox.w * rect.width; const h = candidateBox.h * rect.height;
  ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);
}

function candidatePoint(event) {
  const canvas = $('#candidateCanvas'); const bounds = canvas.getBoundingClientRect(); const rect = candidateImageRect();
  const x = (event.clientX - bounds.left - rect.x) / rect.width; const y = (event.clientY - bounds.top - rect.y) / rect.height;
  return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
}

function renderSuggestions(suggestions = []) {
  const container = $('#suggestionList'); container.innerHTML = '';
  suggestions.forEach((item) => {
    const button = document.createElement('button');
    button.textContent = `${item.class_name} ${Math.round(item.confidence * 100)}%`;
    button.addEventListener('click', () => { candidateBox = item.box; drawCandidateBox(); });
    container.append(button);
  });
}

async function openCandidate() {
  const id = $('#candidateList').value; if (!id) throw new Error('Selecciona un candidato');
  activeCandidate = await apiJson(`/api/dataset/candidate?id=${encodeURIComponent(id)}`, { cache: 'no-store' });
  candidateBox = null; const image = $('#candidateImage'); image.src = activeCandidate.image;
  if (activeCandidate.intended_classes?.length && [...$('#datasetClass').options].some((option) => option.value === activeCandidate.intended_classes[0])) $('#datasetClass').value = activeCandidate.intended_classes[0];
  image.onload = () => { $('#candidateEmpty').hidden = true; drawCandidateBox(); };
  renderSuggestions(activeCandidate.suggestions);
}

async function suggestCandidate() {
  if (!activeCandidate) throw new Error('Abre primero un candidato');
  const data = await apiJson('/api/dataset/suggest', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ candidate_id: activeCandidate.id, confidence: .15, imgsz: 640, scale: $('#yoloScale').value, task: 'detect' }) });
  activeCandidate.suggestions = data.candidate.suggestions; renderSuggestions(activeCandidate.suggestions);
  $('#datasetMessage').textContent = `${data.detector.detections} sugerencias en ${data.detector.latency_ms} ms; aún no cuentan como revisadas.`;
}

async function approveCandidate() {
  if (!activeCandidate || !candidateBox) throw new Error('Abre una imagen y dibuja o elige una caja');
  const data = await apiJson('/api/dataset/approve', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ candidate_id: activeCandidate.id, class_name: $('#datasetClass').value,
      split: $('#datasetSplit').value, box: candidateBox }) });
  $('#datasetMessage').textContent = `Anotación ${data.annotation.annotation_id} aprobada y exportada a YOLO.`;
  $('#datasetMessage').className = 'dataset-message success'; activeCandidate = null; candidateBox = null;
  $('#candidateImage').removeAttribute('src'); $('#candidateEmpty').hidden = false; renderSuggestions([]); drawCandidateBox();
  await Promise.all([loadDatasetStats(), loadCandidateQueue()]);
}

async function rejectCandidate() {
  if (!activeCandidate) throw new Error('Abre primero un candidato');
  await apiJson('/api/dataset/reject', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ candidate_id: activeCandidate.id }) });
  activeCandidate = null; candidateBox = null; $('#candidateImage').removeAttribute('src'); $('#candidateEmpty').hidden = false;
  await Promise.all([loadDatasetStats(), loadCandidateQueue()]);
}

async function loadDatasetStats() {
  const data = await apiJson('/api/dataset/stats', { cache: 'no-store' });
  const container = $('#datasetStats'); container.innerHTML = '';
  const classSelect = $('#datasetClass'); const previousClass = classSelect.value; classSelect.innerHTML = '';
  data.classes.forEach((name) => { const option = document.createElement('option'); option.value = name; option.textContent = name.replaceAll('_', ' '); classSelect.append(option); });
  if (data.classes.includes(previousClass)) classSelect.value = previousClass;
  const summary = document.createElement('div'); summary.className = 'dataset-summary';
  summary.innerHTML = `<strong>${data.reviewed_annotations}/${data.required_annotations}</strong><span>anotaciones revisadas necesarias</span><small>${data.total_images} imágenes YOLO · ${data.pending_candidates} candidatos pendientes</small>`;
  container.append(summary);
  Object.entries(data.per_class).forEach(([name, values]) => {
    const row = document.createElement('div'); row.className = 'dataset-class-row';
    const progress = Math.min(100, (values.reviewed / values.target) * 100);
    row.innerHTML = `<div><span>${escapeText(name)} · ${values.photos} fotos aprobadas</span><b>${values.reviewed}/${values.target}${values.pending ? ` · ${values.pending} fotos pendientes` : ''}</b></div><i><em style="width:${progress}%"></em></i>`;
    container.append(row);
  });
  $('#startTraining').disabled = false;
}

async function auditDataset() {
  const data = await apiJson('/api/dataset/audit', { cache: 'no-store' });
  $('#trainingStatus').textContent = data.ready ? 'AUDITORÍA APROBADA\nDataset listo para entrenar.' : `ENTRENAMIENTO BLOQUEADO\n${data.stats.reviewed_annotations}/${data.stats.required_annotations} revisadas\n${data.blockers} problemas técnicos\nFaltan ${data.stats.required_annotations - data.stats.reviewed_annotations} anotaciones revisadas.`;
  return data;
}

async function trainingStatus() {
  const data = await apiJson('/api/training/status', { cache: 'no-store' });
  $('#trainingStatus').textContent = `${data.running ? 'ENTRENANDO' : 'DETENIDO'} · ${data.hardware.device}\n${data.log_tail || 'Sin registro de entrenamiento.'}`;
}

async function startTraining() {
  const data = await apiJson('/api/training/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ epochs: 40 }) });
  $('#trainingStatus').textContent = `Entrenamiento iniciado. PID ${data.pid}`;
}

async function cancelTraining() {
  const data = await apiJson('/api/training/cancel', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  $('#trainingStatus').textContent = data.cancelled ? 'Entrenamiento cancelado de forma segura.' : 'No había entrenamiento activo.';
}

function showSection(name) {
  document.querySelectorAll('.section').forEach((section) => section.classList.remove('active-section'));
  document.querySelectorAll('.nav-item').forEach((button) => button.classList.toggle('active', button.dataset.section === name));
  $(`#${name}Section`).classList.add('active-section');
  $('#pageTitle').textContent = ({live:'Live Vision',models:'Modelos IA',dataset:'Dataset Studio',history:'Historial y resultados',training:'Entrenamiento',settings:'Configuración y rendimiento'})[name];
  if (name === 'training') { loadDatasetStats().then(() => loadExperiments()).catch(showSystemMessage); }
  if (name === 'models') loadModels();
  if (name === 'history') loadHistory();
  if (name === 'dataset') Promise.all([loadDatasetStats(), loadCandidateQueue(), trainingStatus()]).catch((error) => { $('#datasetMessage').textContent = error.message; });
  $('#sidebar').classList.remove('open');
}

function setupSpeechRecognition() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    $('#micButton').disabled = true;
    $('#micButton').title = 'Reconocimiento de voz no disponible en este navegador';
    return;
  }
  const recognition = new Recognition();
  recognition.lang = 'es-CO';
  recognition.interimResults = false;
  recognition.onresult = (event) => { promptBox.value = event.results[0][0].transcript; };
  recognition.onerror = (event) => showSystemMessage(`No se pudo reconocer la voz: ${event.error}`);
  $('#micButton').addEventListener('click', () => recognition.start());
}

$('#startCamera').addEventListener('click', () => startCamera().catch((error) => {
  setCameraState('No se pudo activar la cámara', cameraErrorMessage(error), true);
  showSystemMessage(cameraErrorMessage(error));
}));
$('#imageFile').addEventListener('change', (event) => { try { loadImageFile(event.target.files[0]); } catch (error) { showSystemMessage(error.message); } });
$('#videoFile').addEventListener('change', (event) => { try { loadVideoFile(event.target.files[0]); } catch (error) { showSystemMessage(error.message); } });
$('#capture').addEventListener('click', () => askScene(true));
$('#askButton').addEventListener('click', () => askScene(false));
$('#cancelButton').addEventListener('click', () => requestController?.abort());
$('#liveDetection').addEventListener('change', (event) => toggleLiveDetection(event.target.checked));
$('#yoloScale').addEventListener('change', selectYoloModel);
$('#yoloTask').addEventListener('change', selectYoloModel);
video.addEventListener('play', () => {
  if (videoFileMode && $('#liveDetection').checked) toggleLiveDetection(true);
});
video.addEventListener('pause', () => {
  if (videoFileMode) {
    liveGeneration += 1;
    clearTimeout(liveTimer);
  }
});
video.addEventListener('ended', () => {
  liveGeneration += 1;
  clearTimeout(liveTimer);
  if (videoFileMode) appendMessage('assistant', 'El video llegó al final. Puedes retrocederlo o hacer una pregunta sobre el último fotograma.', 'Análisis de video');
});
$('#selectRegion').addEventListener('click', () => {
  selectingRegion = true;
  focusRegion = null;
  overlay.classList.add('selecting');
  $('#selectRegion').classList.add('active');
  $('#selectRegion').textContent = 'Arrastra sobre el objeto';
  drawDetections(lastDetections);
});
$('#clearRegion').addEventListener('click', () => {
  focusRegion = null;
  finishRegionSelection();
});
overlay.addEventListener('pointerdown', (event) => {
  if (!selectingRegion) return;
  dragStart = pointToNormalized(event);
  overlay.setPointerCapture(event.pointerId);
});
overlay.addEventListener('pointermove', (event) => {
  if (!selectingRegion || !dragStart) return;
  const point = pointToNormalized(event);
  if (!point) return;
  focusRegion = {
    x: Math.min(dragStart.x, point.x), y: Math.min(dragStart.y, point.y),
    w: Math.abs(point.x - dragStart.x), h: Math.abs(point.y - dragStart.y),
  };
  drawDetections(lastDetections);
});
overlay.addEventListener('pointerup', () => {
  if (!selectingRegion) return;
  if (!focusRegion || focusRegion.w < 0.03 || focusRegion.h < 0.03) {
    focusRegion = null;
    showSystemMessage('La zona era demasiado pequeña. Intenta encerrar la mano y el objeto juntos.');
  } else {
    try { captureFrame(); } catch (error) { showSystemMessage(error.message); }
  }
  finishRegionSelection();
});
$('#stopSpeech').addEventListener('click', () => speechSynthesis?.cancel());
const datasetAction = (action) => Promise.resolve().then(action).catch((error) => { $('#datasetMessage').textContent = error.message; $('#datasetMessage').className = 'dataset-message error'; if ($('#trainingSection').classList.contains('active-section')) $('#trainingStatus').textContent = error.message; });
$('#datasetFiles').addEventListener('change', (event) => datasetAction(() => importDatasetFiles(event.target.files)));
$('#datasetMobile').addEventListener('change', (event) => datasetAction(() => importDatasetFiles(event.target.files)));
$('#datasetFolder').addEventListener('change', (event) => datasetAction(() => importDatasetFiles(event.target.files)));
$('#importCurrentFrame').addEventListener('click', () => datasetAction(importCurrentFrame));
$('#importPublicUrl').addEventListener('click', () => datasetAction(importPublicUrl));
$('#loadCandidate').addEventListener('click', () => datasetAction(openCandidate));
$('#suggestCandidate').addEventListener('click', () => datasetAction(suggestCandidate));
$('#approveCandidate').addEventListener('click', () => datasetAction(approveCandidate));
$('#rejectCandidate').addEventListener('click', () => datasetAction(rejectCandidate));
$('#auditDataset').addEventListener('click', () => datasetAction(auditDataset));
$('#startTraining').addEventListener('click', () => datasetAction(startTraining));
$('#cancelTraining').addEventListener('click', () => datasetAction(cancelTraining));
$('#candidateCanvas').addEventListener('pointerdown', (event) => { if (!activeCandidate || $('#annotationTask').value === 'segment') return; candidateDrag = candidatePoint(event); event.target.setPointerCapture(event.pointerId); });
$('#candidateCanvas').addEventListener('pointermove', (event) => {
  if (!candidateDrag) return; const point = candidatePoint(event);
  candidateBox = { x: Math.min(candidateDrag.x, point.x), y: Math.min(candidateDrag.y, point.y), w: Math.abs(point.x - candidateDrag.x), h: Math.abs(point.y - candidateDrag.y) };
  drawCandidateBox();
});
$('#candidateCanvas').addEventListener('pointerup', () => { candidateDrag = null; });
$('#refreshDataset').addEventListener('click', loadDatasetStats);
$('#clearChat').addEventListener('click', () => {
  conversation = [];
  chat.innerHTML = '';
  appendMessage('assistant', 'Conversación reiniciada. La escena actual sigue disponible.', 'Contexto de chat borrado');
});
document.querySelectorAll('[data-prompt]').forEach((button) => button.addEventListener('click', () => { promptBox.value = button.dataset.prompt; }));
document.querySelectorAll('.nav-item[data-section]').forEach((button) => button.addEventListener('click', () => showSection(button.dataset.section)));
$('#menuButton').addEventListener('click', () => $('#sidebar').classList.toggle('open'));
window.addEventListener('resize', () => { drawDetections(lastDetections); drawCandidateBox(); });
window.addEventListener('beforeunload', () => { stopCamera(); clearVideoFile(); clearTimeout(liveTimer); speechSynthesis?.cancel(); });

if ('speechSynthesis' in window) speechSynthesis.onvoiceschanged = populateVoices;
populateVoices();
setupSpeechRecognition();
refreshStatus();
loadYoloModels().catch((error) => showSystemMessage(`No se pudo consultar YOLO26: ${error.message}`));
listCameras();
setInterval(refreshStatus, 5000);
