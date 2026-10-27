"""HTTP and browser acceptance tests on an isolated temporary dataset."""
import base64
import json
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from vision_web_server import make_handler
from train_visionar import write_json

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'docs'/'evidence'


def main():
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    report={}
    with tempfile.TemporaryDirectory(prefix='visionar-browser-') as temp:
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler('http://127.0.0.1:18091',dataset_root=Path(temp)/'dataset',warmup=False,experiments_root=Path(temp)/'experiments'))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        def request(route,payload=None,headers=None):
            req=urllib.request.Request(base+route,data=json.dumps(payload).encode() if payload is not None else None,
                headers={'Content-Type':'application/json',**(headers or {})})
            try:
                with urllib.request.urlopen(req,timeout=90) as response:
                    return response.status,json.loads(response.read())
            except urllib.error.HTTPError as error:
                return error.code,json.loads(error.read())
        try:
            photo=ROOT/'datasets'/'coco8-seg'/'images'/'val'/'000000000036.jpg'
            image='data:image/jpeg;base64,'+base64.b64encode(photo.read_bytes()).decode()
            status,detected=request('/api/detect',{'image':image,'imgsz':320})
            assert status==200 and detected['detections']
            report['real_inference']={'status':status,'detections':len(detected['detections']),'latency_ms':detected['latency_ms']}
            camera_photo=ROOT/'datasets'/'coco8-seg'/'images'/'val'/'000000000049.jpg'
            camera_image='data:image/jpeg;base64,'+base64.b64encode(camera_photo.read_bytes()).decode()
            status,capture=request('/api/dataset/capture-and-approve',{
                'image':camera_image,'filename':'camera-fixture.jpg','source':'camera fixture',
                'license':'private/test','split':'train','imgsz':320,
            })
            assert status==201 and capture['result']['imported'] and capture['approved']['approved']
            report['camera_capture_and_approval']={
                'status':status,'approved_boxes':len(capture['approved']['approved']),
                'class':capture['approved']['approved'][0]['class_name'],
            }
            status,bad=request('/api/dataset/candidate?id=../private')
            assert status==400
            status,bad=request('/api/dataset/import',{},headers={'Origin':'https://example.com'})
            assert status==403
            status,bad=request('/api/training/start',{'mode':'full'})
            assert status==409 and bad['reason']=='dataset_not_ready'
            report['security_and_training_blockers']='passed'
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser=p.chromium.launch(channel='msedge',headless=True)
                page=browser.new_page(viewport={'width':1440,'height':1000})
                errors=[]; page.on('pageerror',lambda error:errors.append(str(error)))
                page.goto(base)
                page.locator('[data-section="dataset"]').click()
                page.locator('#datasetFiles').set_input_files(str(photo))
                page.locator('#datasetImportProgress').filter(has_text='1 importadas').wait_for(timeout=30000)
                page.locator('.review-card').first.click()
                page.locator('#candidateImage').wait_for(state='visible')
                page.wait_for_function('document.querySelector("#candidateImage").naturalWidth>0')
                page.locator('#candidateCanvas').scroll_into_view_if_needed()
                canvas=page.locator('#candidateCanvas').bounding_box()
                page.mouse.move(canvas['x']+canvas['width']*.3,canvas['y']+canvas['height']*.3)
                page.mouse.down();page.mouse.move(canvas['x']+canvas['width']*.6,canvas['y']+canvas['height']*.7,steps=8);page.mouse.up()
                page.locator('#datasetClass').select_option('person')
                page.locator('#completeReview').check()
                page.locator('#addReviewObject').click()
                page.locator('#reviewDrafts').filter(has_text='persona').wait_for()
                page.locator('#saveReviewBatch').click()
                page.locator('#datasetMessage').filter(has_text='fotografías guardadas').wait_for(timeout=30000)
                status,stats=request('/api/dataset/stats')
                assert stats['reviewed_annotations']>=2 and stats['total_images']>=2
                report['browser_import_review']='passed; temporary fixture only'
                page.evaluate('window.scrollTo(0,0)')
                page.screenshot(path=str(EVIDENCE/'dashboard-desktop.png'),full_page=True)
                page.locator('[data-section="training"]').click()
                page.locator('#trainClasses option').first.wait_for(state='attached')
                assert page.locator('#startTraining').is_visible()
                page.screenshot(path=str(EVIDENCE/'training-desktop.png'),full_page=True)
                for width in (390,768,1440):
                    page.set_viewport_size({'width':width,'height':844})
                    if width<=760:
                        page.locator('#menuButton').click()
                    page.locator('[data-section="live"]').click()
                    page.wait_for_timeout(250)
                    overflow=page.evaluate('document.documentElement.scrollWidth > window.innerWidth')
                    assert not overflow, f'Horizontal overflow at {width}'
                    report[f'viewport_{width}']='passed'
                    if width==390:
                        page.screenshot(path=str(EVIDENCE/'dashboard-mobile.png'),full_page=True)
                assert not errors,errors
                report['browser_errors']=errors
                browser.close()
        finally:
            server.shutdown();server.server_close();thread.join()
    write_json(EVIDENCE/'dashboard_validation.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
