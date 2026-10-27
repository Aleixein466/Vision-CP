/* Workflow UI uses the existing live camera and API; no fabricated metrics. */
let queueOffset=0, queueTotal=0, candidatePolygon=[];
const reviewDrafts=new Map();
const post=(url,body={})=>apiJson(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
const notifyError=error=>{ $('#datasetMessage').textContent=error.message; $('#datasetMessage').className='dataset-message error'; };
const originalOpenCandidate=openCandidate;
openCandidate=async function(){
  await originalOpenCandidate();
  candidatePolygon=[]; $('#polygonPoints').value='[]'; $('#completeReview').checked=false;
  $('#datasetMessage').textContent=`${activeCandidate.source} · ${activeCandidate.license}`;
  let saved=$('#savedAnnotations');
  if(!saved){saved=document.createElement('div');saved.id='savedAnnotations';$('#suggestionList').after(saved);}
  saved.replaceChildren();
  for(const annotation of activeCandidate.annotations||[]){
    if(annotation.withdrawn_at)continue;
    const row=document.createElement('p');row.textContent=`Guardada: ${datasetClassLabel(annotation.class_name)} · ${annotation.task||'detect'} `;
    const button=document.createElement('button');button.className='text-button';button.textContent='Retirar para corregir';
    button.onclick=()=>datasetAction(async()=>{await post('/api/dataset/withdraw',{annotation_id:annotation.annotation_id});await openCandidate();await loadDatasetStats();});
    row.append(button);saved.append(row);
  }
};
loadCandidateQueue=async function(){
  const data=await apiJson(`/api/dataset/candidates?limit=12&offset=${queueOffset}&status=${encodeURIComponent($('#reviewFilter').value)}`);
  queueTotal=data.total;
  $('#queuePosition').textContent=`${data.total ? queueOffset+1 : 0}–${Math.min(queueOffset+12,data.total)} de ${data.total}`;
  $('#previousBatch').disabled=queueOffset===0;
  $('#nextBatch').disabled=queueOffset+12>=data.total;
  $('#candidateList').replaceChildren(); $('#reviewGrid').replaceChildren();
  for(const c of data.items){
    const spanishClasses=(c.intended_classes||[]).map(datasetClassLabel).join(', ');
    const option=new Option(`${c.filename}${spanishClasses ? ` · ${spanishClasses}` : ''} · ${c.status}`,c.id); $('#candidateList').add(option);
    const button=document.createElement('button'); button.className='review-card';
    const img=document.createElement('img'); img.alt=c.filename; img.loading='lazy';
    const label=document.createElement('span'); label.textContent=(c.intended_classes||[]).join(', ') || c.filename;
    label.textContent=spanishClasses || c.filename;
    button.append(img,label); button.title=`${c.source} · ${c.license}`;
    button.onclick=()=>{ $('#candidateList').value=c.id; openCandidate().catch(notifyError); };
    $('#reviewGrid').append(button);
    // Sequential bounded page keeps base64 images out of the list response.
    try { const detail=await apiJson(`/api/dataset/candidate?id=${encodeURIComponent(c.id)}`); img.src=detail.image; }
    catch(error){label.textContent=`Archivo no disponible: ${c.filename}`;}
  }
};
renderSuggestions=function(suggestions=[]){
  $('#suggestionList').replaceChildren();
  const aliases={'cell phone':'cell_phone','potted plant':'potted_plant','dining table':'table','couch':'sofa'};
  for(const item of suggestions){
    const button=document.createElement('button'); button.textContent=`Propuesta: ${item.class_name}`;
    button.textContent=`Propuesta: ${datasetClassLabel(item.class_name)}`;
    button.onclick=()=>{
      candidateBox={...item.box};
      const name=aliases[item.class_name]||item.class_name.replaceAll(' ','_');
      if([...$('#datasetClass').options].some(o=>o.value===name)) $('#datasetClass').value=name;
      drawCandidateBox();
    };
    $('#suggestionList').append(button);
  }
};
const originalDraw=drawCandidateBox;
drawCandidateBox=function(){
  originalDraw();
  if(!activeCandidate || !candidatePolygon.length) return;
  const ctx=$('#candidateCanvas').getContext('2d'), rect=candidateImageRect();
  ctx.beginPath();
  candidatePolygon.forEach(([x,y],i)=>{ const px=rect.x+x*rect.width,py=rect.y+y*rect.height; if(i)ctx.lineTo(px,py);else ctx.moveTo(px,py); });
  ctx.closePath();ctx.fillStyle='#6ea8fe44';ctx.strokeStyle='#a7c9ff';ctx.lineWidth=2;ctx.fill();ctx.stroke();
};
function updatePolygon(){ $('#polygonPoints').value=JSON.stringify(candidatePolygon);drawCandidateBox(); }
function addReviewObject(){
  if(!activeCandidate) throw new Error('Abre una fotografía');
  const task=$('#annotationTask').value;
  if(task==='detect' && !candidateBox) throw new Error('Dibuja o selecciona una caja');
  if(task==='segment' && candidatePolygon.length<3) throw new Error('Revisa un contorno con al menos tres vértices');
  const id=activeCandidate.id;
  const draft=reviewDrafts.get(id)||{candidate_id:id,objects:[],split:$('#datasetSplit').value,complete:false};
  if(draft.split!==$('#datasetSplit').value) throw new Error('Todos los objetos de una fotografía deben pertenecer al mismo split');
  draft.objects.push({class_name:$('#datasetClass').value,task,box:candidateBox?{...candidateBox}:null,polygon:task==='segment'?candidatePolygon.map(p=>[...p]):undefined});
  draft.complete=$('#completeReview').checked;
  reviewDrafts.set(id,draft);
  candidateBox=null;candidatePolygon=[];updatePolygon();renderDrafts();
}
function renderDrafts(){
  $('#reviewDrafts').replaceChildren();
  for(const [id,draft] of reviewDrafts){
    const row=document.createElement('div');
    row.textContent=`${id}: ${draft.objects.map(o=>datasetClassLabel(o.class_name)).join(', ')} · ${draft.complete?'revisión completa':'revisión parcial'} `;
    const remove=document.createElement('button');remove.className='text-button';remove.textContent='Quitar del lote';
    remove.onclick=()=>{reviewDrafts.delete(id);renderDrafts();};row.append(remove);$('#reviewDrafts').append(row);
  }
}
async function saveReviewBatch(){
  if(!reviewDrafts.size) throw new Error('Añade objetos revisados al lote');
  const result=await post('/api/dataset/approve-batch',{items:[...reviewDrafts.values()]});
  reviewDrafts.clear();renderDrafts();
  $('#datasetMessage').textContent=`${result.items.length} fotografías guardadas. Anotaciones separadas por tarea.`;
  await Promise.all([loadDatasetStats(),loadCandidateQueue()]);
}
approveCandidate=async function(){addReviewObject();await saveReviewBatch();};
$('#completeReview').onchange=()=>{if(activeCandidate && reviewDrafts.has(activeCandidate.id)){reviewDrafts.get(activeCandidate.id).complete=$('#completeReview').checked;renderDrafts();}};
$('#annotationTask').onchange=()=>{$('#polygonTools').hidden=$('#annotationTask').value!=='segment';candidateDrag=null;};
$('#candidateCanvas').addEventListener('pointerdown',event=>{
  if(activeCandidate && $('#annotationTask').value==='segment'){ const p=candidatePoint(event);candidatePolygon.push([p.x,p.y]);updatePolygon(); }
});
$('#polygonPoints').onchange=()=>{try{const points=JSON.parse($('#polygonPoints').value);if(!Array.isArray(points)||!points.every(p=>Array.isArray(p)&&p.length===2&&p.every(v=>Number.isFinite(v)&&v>=0&&v<=1)))throw new Error('Vértices inválidos');candidatePolygon=points;drawCandidateBox();}catch(e){notifyError(e);}};
$('#undoVertex').onclick=()=>{candidatePolygon.pop();updatePolygon();};
$('#presegment').onclick=()=>datasetAction(async()=>{
  if(!activeCandidate||!candidateBox) throw new Error('Primero dibuja una caja en modo detección y luego selecciona segmentación');
  const result=await post('/api/dataset/presegment',{candidate_id:activeCandidate.id,box:candidateBox});
  candidatePolygon=result.polygon;updatePolygon();$('#datasetMessage').textContent=result.warning;
});
$('#addReviewObject').onclick=()=>datasetAction(async()=>addReviewObject());
$('#saveReviewBatch').onclick=()=>datasetAction(saveReviewBatch);
$('#previousBatch').onclick=()=>{queueOffset=Math.max(0,queueOffset-12);loadCandidateQueue().catch(notifyError);};
$('#nextBatch').onclick=()=>{queueOffset+=12;loadCandidateQueue().catch(notifyError);};
$('#reviewFilter').onchange=()=>{queueOffset=0;loadCandidateQueue().catch(notifyError);};
$('#trainingActions').append($('.training-controls'));
startTraining=async function(){
  const task=$('#trainTask').value;
  const result=await post('/api/training/start',{
    mode:$('#trainMode').value,task,epochs:Number($('#trainEpochs').value),batch:Number($('#trainBatch').value),
    imgsz:Number($('#trainImgsz').value),threads:Number($('#trainThreads').value),ram_limit_mb:Number($('#trainRam').value),
    classes:[...$('#trainClasses').selectedOptions].map(o=>o.value),
    model:`training/checkpoints/yolo26n${task==='segment'?'-seg':''}.pt`
  });
  toggleLiveDetection(false);
  $('#liveDetection').checked=false;
  $('#trainingStatus').textContent=`Iniciado ${result.experiment}. Motores visuales pausados.`;
  await loadExperiments();
};
cancelTraining=async function(){const result=await post('/api/training/cancel');$('#trainingStatus').textContent=result.detail||result.reason;};
async function loadExperiments(){
  if(!$('#trainClasses').options.length){ const data=await apiJson('/api/dataset/stats');data.classes.forEach(c=>$('#trainClasses').add(new Option(datasetClassLabel(c),c))); }
  const data=await apiJson('/api/training/experiments');$('#experimentsList').replaceChildren();
  for(const id of ['compareLeft','compareRight']){
    const selected=$('#'+id).value;$('#'+id).replaceChildren();
    data.items.forEach(item=>$('#'+id).add(new Option(item.id,item.id)));
    if(data.items.some(item=>item.id===selected))$('#'+id).value=selected;
  }
  if(!data.items.length) $('#experimentsList').textContent='Aún no hay experimentos. Ejecuta una prueba para comprobar la tubería.';
  for(const item of data.items){
    const card=document.createElement('article');card.className='panel experiment-card';
    const heading=document.createElement('h3');heading.textContent=`${item.id} · ${item.result.status||'preparando'}`;
    const config=document.createElement('p');config.textContent=`${item.config.mode} · ${item.config.task} · ${item.config.epochs} epochs · ${item.config.imgsz}px · batch ${item.config.batch}`;
    const metrics=document.createElement('pre');metrics.textContent=JSON.stringify({validation:item.result.validation||'Sin medición',test:item.result.independent_test||'Sin evaluación independiente'},null,2);
    card.append(heading,config,metrics);
    for(const [action,label,enabled] of [['resume','Recuperar checkpoint',item.recoverable&&['cancelled','failed','interrupted'].includes(item.result.status)],['promote','Activar al reiniciar',item.result.promotable]]){
      if(!enabled)continue;
      const button=document.createElement('button');button.className='button secondary';button.textContent=label;
      button.onclick=async()=>{try{const result=await post(`/api/training/${action}`,{experiment:item.id});$('#trainingStatus').textContent=JSON.stringify(result,null,2);await loadExperiments();}catch(e){$('#trainingStatus').textContent=e.message;}};
      card.append(button);
    }
    $('#experimentsList').append(card);
  }
}
$('#refreshExperiments').onclick=()=>loadExperiments().catch(e=>{$('#trainingStatus').textContent=e.message;});
$('#compareModels').onclick=async()=>{try{$('#comparisonResult').textContent=JSON.stringify(await apiJson(`/api/training/compare?left=${encodeURIComponent($('#compareLeft').value)}&right=${encodeURIComponent($('#compareRight').value)}`),null,2);}catch(e){$('#comparisonResult').textContent=e.message;}};
$('#trainMode').onchange=()=>{$('#trainClasses').disabled=$('#trainMode').value!=='partial';};
$('#trainClasses').disabled=true;
$('#rollbackModel').onclick=async()=>{try{$('#trainingStatus').textContent=JSON.stringify(await post('/api/training/rollback'));}catch(e){$('#trainingStatus').textContent=e.message;}};
for(const action of ['startTraining','cancelTraining','auditDataset']){
  // Existing handlers report errors in Dataset Studio; also surface them in Training.
  $('#'+action).addEventListener('click',()=>setTimeout(()=>{if($('#datasetMessage').classList.contains('error'))$('#trainingStatus').textContent=$('#datasetMessage').textContent;},500));
}
$('#frameInterval').value=localStorage.getItem('visionar.interval')||250;
$('#confidenceSetting').value=localStorage.getItem('visionar.confidence')||.32;
$('#confidenceSetting').oninput=()=>{$('#confidenceOutput').textContent=$('#confidenceSetting').value;};
$('#saveSettings').onclick=()=>{
  const interval=Math.max(100,Math.min(5000,Number($('#frameInterval').value)||250));
  localStorage.setItem('visionar.interval',interval);localStorage.setItem('visionar.confidence',$('#confidenceSetting').value);
  $('#settingsResult').textContent='Preferencias guardadas en este navegador.';
};
for(const [id,route] of [['backupDataset','backup'],['reconcileDataset','reconcile']]) $('#'+id).onclick=async()=>{
  $('#'+id).disabled=true;$('#settingsResult').textContent='Procesando…';
  try{$('#settingsResult').textContent=JSON.stringify(await post(`/api/dataset/${route}`),null,2);}
  catch(e){$('#settingsResult').textContent=e.message;}finally{$('#'+id).disabled=false;}
};
setInterval(()=>{if($('#trainingSection').classList.contains('active-section'))trainingStatus().catch(()=>{});},3000);
document.querySelectorAll('textarea,input,select').forEach(el=>{if(!el.getAttribute('aria-label')&&!el.closest('label'))el.setAttribute('aria-label',el.id||'Entrada');});
$('#datasetMessage').setAttribute('role','status');
$('#trainingStatus').setAttribute('role','status');
document.addEventListener('keydown',e=>{if(e.key==='Escape'){$('#sidebar').classList.remove('open');$('#menuButton').setAttribute('aria-expanded','false');}});
$('#menuButton').addEventListener('click',()=>$('#menuButton').setAttribute('aria-expanded',$('#sidebar').classList.contains('open')));
