"""Camera, genuine masks and a real VLM response; no camera photographs are saved."""
import base64
import json
import time
from pathlib import Path
import cv2
import psutil
from visual_fusion import VisualFusionEngine
from vlm_engine import VLMEngine
from vision_web_server import llama_request
from train_visionar import write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    report={}
    engine=VisualFusionEngine()
    capture=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    frames=[]
    try:
        for _ in range(3):
            ok,frame=capture.read()
            if not ok:
                break
            encoded='data:image/jpeg;base64,'+base64.b64encode(cv2.imencode('.jpg',frame)[1]).decode()
            result=engine.detect(encoded,.3,320)
            frames.append({'width':frame.shape[1],'height':frame.shape[0],'detections':len(result['detections']),'latency_ms':result['latency_ms']})
    finally:
        capture.release()
    report['camera']={'status':'passed' if frames else 'unavailable','frames':frames,'photos_saved':False}
    photo=ROOT/'datasets'/'coco8-seg'/'images'/'val'/'000000000036.jpg'
    encoded='data:image/jpeg;base64,'+base64.b64encode(photo.read_bytes()).decode()
    result=engine.detect(encoded,.25,320,task='segment')
    assert any(d.get('polygon') for d in result['detections'])
    report['segmentation']={'masks':len([d for d in result['detections'] if d.get('polygon')]),'model':result['model']}
    proposal=engine.presegment(encoded,{'x':.2,'y':.1,'w':.7,'h':.8})
    assert proposal['reviewed'] is False and len(proposal['polygon'])>=3
    report['presegmentation']={'method':proposal['method'],'points':len(proposal['polygon']),'reviewed':False}
    engine.unload()
    vlm=VLMEngine(port=18091,selected='smolvlm-500m')
    try:
        started=time.perf_counter()
        vlm.start(wait_seconds=90)
        report['vlm']=llama_request(vlm.url,encoded,'Describe únicamente lo visible. Si no puedes confirmar algo, dilo.',vlm.api_key,engine.prompt_context(result),model='smolvlm-500m')
        report['vlm']['wall_seconds']=round(time.perf_counter()-started,2)
        report['vlm']['automatically_fact_checked']=False
    except Exception as exc:
        report['vlm']={'status':'unavailable','error':str(exc)}
    finally:
        vlm.stop()
    report['ram_available_mb']=round(psutil.virtual_memory().available/2**20)
    write_json(ROOT/'docs'/'evidence'/'multimodal_validation.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
