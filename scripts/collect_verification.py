"""Capture final test results and existing real experiment artifacts."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from train_visionar import write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    commands=[[sys.executable,'-m','unittest','discover','-s','tests','-v'],
              ['node','scripts/test_scene_tracking.js'],['node','--check','web/app.js'],['node','--check','web/dashboard.js']]
    checks=[]
    for command in commands:
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
        checks.append({'command':command,'exit_code':result.returncode,'output':result.stdout+result.stderr})
    experiments=[]
    for folder in sorted((ROOT/'training'/'experiments').glob('*')):
        if (folder/'result.json').exists():
            experiments.append({'id':folder.name,'config':json.loads((folder/'config.json').read_text()),
                                'result':json.loads((folder/'result.json').read_text()),
                                'resources':json.loads((folder/'resources.json').read_text()) if (folder/'resources.json').exists() else None})
    files=list((ROOT/'scripts').glob('*.py'))+list((ROOT/'web').glob('*'))
    report={'collected_at':datetime.now(timezone.utc).isoformat(),'checks':checks,'experiments':experiments,
            'code_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files if path.is_file()},
            'production_photographs_changed':False,'production_model_promoted':False}
    write_json(ROOT/'docs'/'evidence'/'verification_summary.json',report)
    print(json.dumps({'checks':[(c['command'][-1],c['exit_code']) for c in checks],'experiments':len(experiments)}))
    if any(c['exit_code'] for c in checks):
        raise SystemExit(1)


if __name__=='__main__':
    main()
