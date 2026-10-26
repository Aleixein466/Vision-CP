"""Isolated training worker; results never replace the active model."""
import argparse
import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def run(experiment):
    experiment = Path(experiment).resolve()
    cfg = json.loads((experiment/'config.json').read_text(encoding='utf-8'))
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(cfg['threads'])
    import torch
    import yaml
    from ultralytics import YOLO
    torch.set_num_threads(cfg['threads'])
    state = {'status':'running','started_at':datetime.now(timezone.utc).isoformat(), 'production_deployed':False}
    def persist(**changes):
        state.update(changes)
        write_json(experiment/'result.json',state)
    def checkpoint(trainer):
        persist(epoch=min(trainer.epoch+1,trainer.epochs), last_checkpoint=str(trainer.last), best_checkpoint=str(trainer.best))
        if (experiment/'cancel.request').exists():
            trainer.stop = True
    try:
        persist()
        model = YOLO(cfg.get('resume') or cfg['model'])
        if model.task != cfg['task']:
            raise ValueError('La tarea del checkpoint no coincide con las anotaciones')
        model.add_callback('on_fit_epoch_end',checkpoint)
        model.train(data=cfg['data'], epochs=cfg['epochs'], imgsz=cfg['imgsz'], batch=cfg['batch'],
                    device='cpu', workers=0, cache=False, plots=False, patience=10,
                    save=True, save_period=1, seed=42, deterministic=True,
                    project=str(experiment), name='run', exist_ok=True, resume=bool(cfg.get('resume')))
        if (experiment/'cancel.request').exists():
            persist(status='cancelled', recovery='Resume from the last complete epoch checkpoint')
            return
        best = experiment/'run'/'weights'/'best.pt'
        evaluator=YOLO(str(best))
        def evaluate(split):
            return evaluator.val(data=cfg['data'], split=split, imgsz=cfg['imgsz'], batch=1,
                                 device='cpu',workers=0,plots=False,project=str(experiment),name=split)
        validation=evaluate('val')
        config=yaml.safe_load(Path(cfg['data']).read_text(encoding='utf-8'))
        independent=evaluate('test') if config.get('test') else None
        def metrics(result):
            import math
            def finite(v):
                value=float(v)
                return value if math.isfinite(value) else None
            per_class=[]
            for i, class_id in enumerate(result.box.ap_class_index):
                p,r,ap50,ap=result.box.class_result(i)
                per_class.append({'class_id':int(class_id),'name':evaluator.names[int(class_id)],
                                  'precision':finite(p),'recall':finite(r),'mAP50':finite(ap50),'mAP50_95':finite(ap)})
                if cfg['task']=='segment':
                    mp,mr,ma50,ma=result.seg.class_result(i)
                    per_class[-1]['mask']={'precision':finite(mp),'recall':finite(mr),'mAP50':finite(ma50),'mAP50_95':finite(ma)}
            return {'aggregate':{k:finite(v) for k,v in result.results_dict.items()},'per_class':per_class}
        persist(status='completed', best_checkpoint=str(best), validation=metrics(validation),
                checkpoint_sha256=hashlib.sha256(best.read_bytes()).hexdigest(),
                independent_test=metrics(independent) if independent is not None else None,
                promotable=cfg['mode']!='smoke' and independent is not None,
                finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        persist(status='failed',error=str(exc),finished_at=datetime.now(timezone.utc).isoformat())
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--mode',choices=['smoke','partial','full'],default='full')
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--task',choices=['detect','segment'],default='detect')
    parser.add_argument('--classes',nargs='+')
    parser.add_argument('--model',default='training/checkpoints/yolo26n.pt')
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--imgsz',type=int,default=320)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--ram-limit-mb',type=int,default=2400)
    args=parser.parse_args()
    if args.experiment:
        run(args.experiment)
    else:
        from dataset_studio import DatasetStudio
        studio=DatasetStudio()
        if not args.execute:
            print(json.dumps(studio.training_audit(args.classes,args.task,2 if args.mode=='partial' else 30),ensure_ascii=False,indent=2))
        else:
            from training_manager import TrainingManager
            import time
            manager=TrainingManager(studio)
            result=manager.start(epochs=args.epochs,model=args.model,mode=args.mode,classes=args.classes,task=args.task,
                                 batch=args.batch,imgsz=args.imgsz,threads=args.threads,ram_limit_mb=args.ram_limit_mb)
            print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
            if not result.get('started'):
                raise SystemExit(2)
            while manager.status()['running']:
                time.sleep(1)
            print(json.dumps(manager.status(),ensure_ascii=False,indent=2))
            if manager.status().get('status')!='completed':
                raise SystemExit(1)


if __name__=='__main__':
    main()
