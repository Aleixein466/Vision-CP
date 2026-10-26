import base64
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from scripts.dataset_studio import DatasetStudio


def image(seed):
    pixels=np.random.default_rng(seed).integers(0,255,(80,100,3),dtype=np.uint8)
    return 'data:image/jpeg;base64,'+base64.b64encode(cv2.imencode('.jpg',pixels)[1]).decode()


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.studio=DatasetStudio(self.root)
        self.box={'x':.1,'y':.1,'w':.4,'h':.4}

    def tearDown(self):
        self.tmp.cleanup()

    def candidate(self,seed=1):
        return self.studio.import_candidate(image(seed),'fixture.jpg','synthetic test fixture','private')['candidate']['id']

    def test_suggestions_never_reviewed(self):
        identity=self.candidate()
        self.studio.set_suggestions(identity,[{'class_name':'person','confidence':1,'box':self.box}])
        self.assertEqual(self.studio.stats()['reviewed_annotations'],0)
        self.assertFalse(self.studio.is_reviewed({'reviewed':True,'annotation':'external_human_verified_rectangle'}))
        self.assertFalse(self.studio.is_reviewed({'annotation':'anything'}))

    def test_batch_rollback_and_multiple_objects(self):
        identity=self.candidate()
        with self.assertRaises(ValueError):
            self.studio.approve_objects(identity,[{'class_name':'person','box':self.box},{'class_name':'invalid','box':self.box}],'train')
        self.assertEqual(self.studio.stats()['annotations'],0)
        self.assertEqual(self.studio.candidate(identity)['status'],'pending')
        self.studio.approve_objects(identity,[{'class_name':name,'box':self.box} for name in ('person','shirt')],'train',True)
        self.assertEqual(self.studio.stats()['reviewed_annotations'],2)
        self.assertEqual(self.studio.stats()['total_images'],1)

    def test_exact_and_near_leakage(self):
        identity=self.candidate()
        self.studio.approve_candidate(identity,'person',self.box,'train')
        with self.assertRaises(ValueError):
            self.studio.approve_candidate(identity,'shirt',self.box,'test')
        self.assertFalse(self.studio.import_candidate(image(1),'copy.jpg','fixture','private')['imported'])

    def test_detection_export_five_fields_and_snapshot(self):
        for seed,split in enumerate(('train','val','test'),1):
            self.studio.approve_objects(self.candidate(seed),[{'class_name':'person','box':self.box}],split,True)
        config=self.studio.export_snapshot(self.root/'export',['person'],'detect',1)
        for label in config.parent.glob('labels/*/*.txt'):
            self.assertEqual(label.read_text().strip().split(),['0','0.300000','0.300000','0.400000','0.400000'])
        self.assertFalse(self.studio.training_audit(['person'],'segment',1)['ready'])

    def test_polygon_validation_and_separate_record(self):
        polygon=[[.1,.1],[.5,.1],[.4,.3],[.5,.5],[.1,.4]]
        self.studio.approve_objects(self.candidate(),[{'class_name':'person','task':'segment','polygon':polygon}],'train',True)
        records=self.studio._records()
        self.assertEqual({r['task'] for r in records},{'detect','segment'})
        with self.assertRaises(ValueError):
            self.studio.validate_polygon([[.1,.1],[.5,.5],[.1,.5],[.5,.1]])

    def test_concurrent_import_idempotent(self):
        other=DatasetStudio(self.root)
        with ThreadPoolExecutor(2) as pool:
            results=list(pool.map(lambda s:s.import_candidate(image(7),'x.jpg','fixture','private'),[self.studio,other]))
        self.assertEqual(sum(r['imported'] for r in results),1)

    def test_traversal_rejected(self):
        with self.assertRaises(ValueError):
            self.studio.candidate('../secret',True)

    def test_session_cannot_cross_splits(self):
        ids=[self.studio.import_candidate(image(seed),'x.jpg','camera','private',group_id='session')['candidate']['id'] for seed in (21,22)]
        self.studio.approve_candidate(ids[0],'person',self.box,'train')
        with self.assertRaises(ValueError):
            self.studio.approve_candidate(ids[1],'person',self.box,'val')

    def test_withdraw_preserves_history_and_excludes_export(self):
        result=self.studio.approve_candidate(self.candidate(),'person',self.box,'train')
        self.studio.withdraw_annotation(result['annotation_id'])
        self.assertEqual(len(self.studio._records()),1)
        self.assertEqual(self.studio.stats()['reviewed_annotations'],0)
        self.assertFalse(self.studio.training_audit(['person'],minimum=1)['ready'])

    def test_segmentation_export_is_real_polygon(self):
        polygon=[[.1,.1],[.5,.1],[.4,.3],[.5,.5],[.1,.4]]
        for seed,split in enumerate(('train','val','test'),1):
            self.studio.approve_objects(self.candidate(seed),[{'class_name':'person','task':'segment','polygon':polygon}],split,True)
        config=self.studio.export_snapshot(self.root/'segments',['person'],'segment',1)
        for label in config.parent.glob('labels/*/*.txt'):
            self.assertEqual(len(label.read_text().split()),11)

    def test_malformed_legacy_is_preserved_and_reported(self):
        other=self.root/'legacy';other.mkdir()
        original='{"id":"one","status":"pending"}\nmalformed\n'
        (other/'candidates.jsonl').write_text(original)
        migrated=DatasetStudio(other)
        self.assertEqual(len(migrated.store.errors()),1)
        self.assertEqual((other/'candidates.jsonl').read_text(),original)

    def test_reconcile_detects_orphan_and_preserves_files(self):
        identity=self.candidate()
        path=self.studio._candidate_path(identity)
        original=path.read_bytes()
        (self.root/'candidates'/'orphan.jpg').write_bytes(original)
        report=self.studio.reconcile()
        self.assertIn('orphan_candidate_file',{i['type'] for i in report['issues']})
        self.assertEqual(path.read_bytes(),original)
        self.assertTrue(self.studio.store.backup().is_file())


if __name__=='__main__':
    unittest.main()
