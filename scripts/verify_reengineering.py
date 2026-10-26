"""Real local evidence. Synthetic contract tests live separately in tests/."""
import argparse
import base64
import json
import time
from pathlib import Path
from dataset_studio import DatasetStudio
from visual_fusion import VisualFusionEngine
from training_manager import TrainingManager
from train_visionar import write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',action='store_true')
    args=parser.parse_args()
    studio=DatasetStudio()
    report={'dataset':studio.reconcile()}
    write_json(ROOT/'docs'/'evidence'/'dataset_audit.json',report['dataset'])
    engine=VisualFusionEngine()
    photo=ROOT/'datasets'/'coco8-seg'/'images'/'val'/'000000000036.jpg'
    encoded='data:image/jpeg;base64,'+base64.b64encode(photo.read_bytes()).decode()
    report['inference']=engine.detect(encoded,.25,320)
    report['roi_inference']=engine.detect(encoded,.25,320,focus={'x':.1,'y':.1,'w':.7,'h':.7})
    engine.unload()
    if args.train:
        manager=TrainingManager(studio)
        started=manager.start(mode='smoke',epochs=1,model='training/checkpoints/yolo26n.pt',ram_limit_mb=2400)
        print(json.dumps(started),flush=True)
        if not started['started']:
            raise RuntimeError(started)
        while manager.status()['running']:
            time.sleep(2)
        report['training']=manager.status()
        report['experiments']=manager.experiments()
    report['limitations']=['COCO8 smoke validates the pipeline, not 38-class performance',
                          'No pending production annotation was approved automatically']
    write_json(ROOT/'docs'/'evidence'/'runtime_validation.json',report)
    print(json.dumps({'inference_detections':len(report['inference']['detections']),
                      'training_status':report.get('training',{}).get('status'),'evidence':'docs/evidence/runtime_validation.json'}),flush=True)


if __name__=='__main__':
    main()
