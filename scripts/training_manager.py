"""Experiment lifecycle with resource limits, recovery and explicit promotion."""
import json
import os
import subprocess
import sys
import threading
import time
import uuid
import hashlib
from pathlib import Path
import psutil
import yaml
try:
    from dataset_studio import DatasetStudio, CLASSES
    from train_visionar import write_json
except ModuleNotFoundError:
    from scripts.dataset_studio import DatasetStudio, CLASSES
    from scripts.train_visionar import write_json

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENTS=ROOT/'training'/'experiments'
REGISTRY=ROOT/'training'/'model_registry.json'


def dataset_fingerprint(data):
    config=yaml.safe_load(Path(data).read_text(encoding='utf-8'))
    root=Path(config['path'])
    digest=hashlib.sha256(json.dumps(config['names'],sort_keys=True).encode())
    for path in sorted(root.glob('**/*')):
        if path.is_file() and path.suffix in {'.jpg','.txt'}:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def active_checkpoint():
    if REGISTRY.exists():
        registry=json.loads(REGISTRY.read_text(encoding='utf-8'))
        if registry.get('active'):
            path=Path(registry['active'])
            if path.is_file() and path.resolve().is_relative_to(EXPERIMENTS.resolve()):
                return path
    return None


class TrainingManager:
    def __init__(self,dataset:DatasetStudio,root=None,pause=None):
        self.dataset=dataset
        self.root=Path(root or EXPERIMENTS)
        self.process=None
        self.lock=threading.RLock()
        self.current=None
        self.pause=pause
        self.monitor=None

    def experiments(self):
        items=[]
        for folder in sorted(self.root.glob('*'),reverse=True):
            if not (folder/'config.json').exists():
                continue
            config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
            result=json.loads((folder/'result.json').read_text(encoding='utf-8')) if (folder/'result.json').exists() else {}
            if result.get('status')=='running' and not (self.current==folder and self.process and self.process.poll() is None):
                result['status']='interrupted'
            items.append({'id':folder.name,'config':config,'result':result,
                          'recoverable':bool(list((folder/'run'/'weights').glob('epoch*.pt')))})
        return items

    def status(self):
        with self.lock:
            running=self.process is not None and self.process.poll() is None
            if not running and self.monitor and self.monitor.is_alive():
                self.monitor.join(timeout=2)
            saved={}
            if self.current and (self.current/'result.json').exists():
                saved=json.loads((self.current/'result.json').read_text(encoding='utf-8'))
            if self.current and (self.current/'resources.json').exists():
                saved['resources']=json.loads((self.current/'resources.json').read_text())
            if not running and saved.get('status')=='running':
                saved['status']='interrupted'
            if self.current and (self.current/'train.log').exists():
                with (self.current/'train.log').open('rb') as handle:
                    handle.seek(0,2)
                    handle.seek(max(0,handle.tell()-6000))
                    saved['log_tail']=handle.read().decode('utf-8',errors='replace')
            return {**saved,'running':running,'pid':self.process.pid if running else None,
                    'experiment':self.current.name if self.current else None,
                    'hardware':{'device':'cpu','ram_available_mb':round(psutil.virtual_memory().available/2**20)}}

    def compare(self,left,right):
        experiments={item['id']:item for item in self.experiments()}
        if left not in experiments or right not in experiments:
            raise ValueError('Selecciona dos experimentos existentes')
        a,b=experiments[left],experiments[right]
        if a['config']['task']!=b['config']['task'] or dataset_fingerprint(a['config']['data'])!=dataset_fingerprint(b['config']['data']):
            raise ValueError('Los experimentos usan tareas o datasets distintos; sus métricas no son comparables directamente')
        am=(a['result'].get('independent_test') or {}).get('aggregate')
        bm=(b['result'].get('independent_test') or {}).get('aggregate')
        if not am or not bm:
            raise ValueError('Falta evaluación independiente en uno de los experimentos')
        return {'left':left,'right':right,'same_dataset':True,'metrics':{
            k:{'left':v,'right':bm[k],'delta':bm[k]-v} for k,v in am.items() if k in bm and v is not None and bm[k] is not None}}

    def _launch(self,folder,cfg):
        self.root.mkdir(parents=True,exist_ok=True)
        lease=(self.root/'worker.lock').open('a+b')
        lease.seek(0);lease.write(b'0');lease.flush();lease.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(lease.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            owner_path=self.root/'worker_owner.json'
            if owner_path.exists():
                owner=json.loads(owner_path.read_text())
                try:
                    prior=psutil.Process(owner['pid'])
                    if abs(prior.create_time()-owner['created'])<.01 and prior.is_running():
                        raise ValueError('Existe otro entrenamiento activo; espera a que termine')
                except psutil.NoSuchProcess:
                    pass
            if self.pause:
                self.pause()
            else:
                import urllib.request
                external_ready=False
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8091/health',timeout=1) as response:
                        external_ready=response.status==200
                except OSError:
                    pass
                if external_ready:
                    raise ValueError('El VLM está activo. Inicia el entrenamiento desde su dashboard para pausarlo')
        except Exception:
            lease.close()
            raise
        self.current=folder
        log_handle=(folder/'train.log').open('a',encoding='utf-8')
        try:
            self.process=subprocess.Popen([sys.executable,str(ROOT/'scripts'/'train_visionar.py'),'--experiment',str(folder)],
                cwd=ROOT,stdout=log_handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except Exception:
            log_handle.close()
            lease.close()
            raise
        process=self.process
        write_json(owner_path,{'pid':process.pid,'created':psutil.Process(process.pid).create_time()})
        def monitor():
            peak=0
            while process.poll() is None:
                try:
                    parent=psutil.Process(process.pid)
                    owned=[parent]+parent.children(recursive=True)
                    memory=sum(p.memory_info().rss for p in owned if p.is_running())/2**20
                    peak=max(peak,memory)
                    if memory>cfg['ram_limit_mb'] or psutil.virtual_memory().available<350*2**20:
                        (folder/'cancel.request').touch()
                        for child in reversed(owned):
                            try:
                                child.terminate()
                            except psutil.NoSuchProcess:
                                pass
                        write_json(folder/'resource_stop.json',{'reason':'memory_limit','rss_mb':memory})
                except psutil.Error:
                    break
                time.sleep(.5)
            process.wait()
            log_handle.close()
            write_json(folder/'resources.json',{'peak_worker_rss_mb':round(peak,1),'returncode':process.returncode})
            lease.close()
        self.monitor=threading.Thread(target=monitor,daemon=True)
        self.monitor.start()
        return {'started':True,'pid':process.pid,'experiment':folder.name,'config':cfg}

    def start(self,epochs=40,model='training/checkpoints/yolo26n-objv1-150.pt',mode='full',classes=None,
              task='detect',batch=1,imgsz=320,threads=1,ram_limit_mb=2400):
        with self.lock:
            if self.process and self.process.poll() is None:
                return {'started':False,'reason':'already_running'}
            if mode not in {'smoke','partial','full'} or task not in {'detect','segment'}:
                raise ValueError('Modo o tarea inválida')
            epochs,batch,imgsz,threads,ram_limit_mb=map(int,(epochs,batch,imgsz,threads,ram_limit_mb))
            if not (1<=epochs<=300 and 1<=batch<=4 and 320<=imgsz<=960 and imgsz%32==0 and 1<=threads<=min(4,os.cpu_count() or 1) and 800<=ram_limit_mb<=8192):
                raise ValueError('Configuración fuera de los límites CPU: batch 1–4, imgsz 320–960 múltiplo de 32, threads 1–4')
            classes=CLASSES if mode=='full' else classes
            if mode=='partial' and not classes:
                raise ValueError('Selecciona al menos una clase')
            if mode!='smoke':
                audit=self.dataset.training_audit(classes,task,30 if mode=='full' else 2)
                if not audit['ready']:
                    return {'started':False,'reason':'dataset_not_ready','audit':audit}
            path=(ROOT/model).resolve()
            if not path.is_relative_to((ROOT/'training'/'checkpoints').resolve()) or not path.is_file() or path.suffix!='.pt':
                raise ValueError('Usa un checkpoint local de training/checkpoints')
            if psutil.virtual_memory().available < 1000*2**20:
                raise ValueError('Se requieren al menos 1000 MB disponibles antes del entrenamiento')
            folder=self.root/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
            folder.mkdir(parents=True)
            if mode=='smoke':
                # External fixture only, never counted as reviewed production data.
                from ultralytics import YOLO
                import shutil
                data_root=ROOT/'datasets'/'coco8-seg'
                names=YOLO(str(ROOT/'training'/'checkpoints'/'yolo26n-seg.pt')).names
                snapshot=folder/'dataset'
                for split in ('train','val','test'):
                    (snapshot/'images'/split).mkdir(parents=True)
                    (snapshot/'labels'/split).mkdir(parents=True)
                    source_split='val' if split=='test' else split
                    images=sorted((data_root/'images'/source_split).glob('*.jpg'))
                    if split=='val':
                        images=images[:-1]
                    if split=='test':
                        images=images[-1:]
                    for image in images:
                        shutil.copy2(image,snapshot/'images'/split/image.name)
                        rows=[]
                        for line in (data_root/'labels'/source_split/(image.stem+'.txt')).read_text().splitlines():
                            parts=line.split(); values=list(map(float,parts[1:])); xs,ys=values[::2],values[1::2]
                            rows.append(line if task=='segment' else parts[0]+' '+' '.join(map(str,[(min(xs)+max(xs))/2,(min(ys)+max(ys))/2,max(xs)-min(xs),max(ys)-min(ys)])))
                        (snapshot/'labels'/split/(image.stem+'.txt')).write_text('\n'.join(rows)+'\n',encoding='utf-8')
                data=snapshot/'data.yaml'
                data.write_text(yaml.safe_dump({'path':snapshot.resolve().as_posix(),'train':'images/train','val':'images/val','test':'images/test','names':names}),encoding='utf-8')
                epochs=min(epochs,3)
            else:
                data=self.dataset.export_snapshot(folder/'dataset',classes,task,30 if mode=='full' else 2)
            cfg={'mode':mode,'task':task,'classes':classes,'model':str(path),'data':str(data.resolve()),
                 'epochs':epochs,'batch':batch,'imgsz':imgsz,'threads':threads,'ram_limit_mb':ram_limit_mb,
                 'purpose':'pipeline_validation_only' if mode=='smoke' else 'candidate_model'}
            cfg['dataset_fingerprint']=dataset_fingerprint(data)
            write_json(folder/'config.json',cfg)
            return self._launch(folder,cfg)

    def cancel(self):
        with self.lock:
            if not self.process or self.process.poll() is not None:
                return {'cancelled':False,'reason':'not_running'}
            (self.current/'cancel.request').touch()
            return {'cancelled':True,'status':'cancelling','detail':'Se detendrá después de guardar el epoch actual'}

    def resume(self,experiment):
        with self.lock:
            if self.process and self.process.poll() is None:
                raise ValueError('Ya hay entrenamiento activo')
            folder=(self.root/experiment).resolve()
            if folder.parent!=self.root.resolve():
                raise ValueError('Experimento inválido')
            cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
            weights=sorted((folder/'run'/'weights').glob('epoch*.pt'),key=lambda p:int(p.stem[5:]))
            if not weights:
                raise ValueError('No hay un epoch completo recuperable; inicia otro experimento')
            cfg['resume']=str(weights[-1])
            write_json(folder/'config.json',cfg)
            if (folder/'cancel.request').exists():
                (folder/'cancel.request').rename(folder/('cancel-'+uuid.uuid4().hex+'.request'))
            return self._launch(folder,cfg)

    def promote(self,experiment):
        with self.lock:
            item=next((e for e in self.experiments() if e['id']==experiment),None)
            if not item or not item['result'].get('promotable') or item['result'].get('status')!='completed':
                raise ValueError('Solo puedes activar experimentos completados con evaluación independiente')
            checkpoint=Path(item['result']['best_checkpoint']).resolve()
            if not checkpoint.is_relative_to((self.root/experiment).resolve()) or not checkpoint.is_file():
                raise ValueError('El checkpoint evaluado no está disponible en el experimento')
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=item['result'].get('checkpoint_sha256'):
                raise ValueError('El checkpoint no coincide con el que fue evaluado')
            if dataset_fingerprint(item['config']['data'])!=item['config'].get('dataset_fingerprint'):
                raise ValueError('El dataset fue modificado después de crear el experimento')
            registry=json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
            write_json(REGISTRY,{'active':item['result']['best_checkpoint'],'previous':registry.get('active'),'experiment':experiment})
            return {'activated':True,'restart_required':True}

    def rollback(self):
        with self.lock:
            if not REGISTRY.exists():
                raise ValueError('No hay activaciones anteriores')
            registry=json.loads(REGISTRY.read_text())
            write_json(REGISTRY,{'active':registry.get('previous'),'previous':registry.get('active')})
            return {'rolled_back':True,'restart_required':True}
