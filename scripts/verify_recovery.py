"""Real cancellation + checkpoint resumption, isolated from production records."""
import json
import time
from pathlib import Path
from dataset_studio import DatasetStudio
from training_manager import TrainingManager
from train_visionar import write_json


def main():
    manager=TrainingManager(DatasetStudio())
    started=manager.start(mode='smoke',epochs=3,model='training/checkpoints/yolo26n.pt',ram_limit_mb=3000)
    assert started['started'],started
    cancelled=manager.cancel()
    deadline=time.monotonic()+180
    while manager.status()['running'] and time.monotonic()<deadline:
        time.sleep(1)
    first=manager.status()
    assert first.get('status')=='cancelled',first
    resumed=manager.resume(started['experiment'])
    assert resumed['started']
    deadline=time.monotonic()+180
    while manager.status()['running'] and time.monotonic()<deadline:
        time.sleep(1)
    final=manager.status()
    assert final.get('status')=='completed',final
    report={'cancel_request':cancelled,'cancelled':first,'resumed':resumed,'final':final}
    write_json(Path(__file__).resolve().parents[1]/'docs'/'evidence'/'recovery_validation.json',report)
    print(json.dumps({'cancelled':first['status'],'resumed':final['status'],'experiment':started['experiment']}),flush=True)


if __name__=='__main__':
    main()
