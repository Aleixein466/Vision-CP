import tempfile
import unittest
from pathlib import Path
from scripts.dataset_studio import DatasetStudio
from scripts.training_manager import TrainingManager


class TrainingSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.manager=TrainingManager(DatasetStudio(self.root/'dataset'),root=self.root/'experiments')

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_training_refuses_empty_reviewed_dataset(self):
        result=self.manager.start()
        self.assertEqual(result['reason'],'dataset_not_ready')
        self.assertIsNone(self.manager.process)

    def test_configuration_bounds(self):
        for kw in ({'batch':50},{'threads':100},{'imgsz':321},{'mode':'invalid'}):
            with self.assertRaises(ValueError):
                self.manager.start(mode='smoke',**kw) if 'mode' not in kw else self.manager.start(**kw)

    def test_no_unvalidated_promotion(self):
        with self.assertRaises(ValueError):
            self.manager.promote('nonexistent')

    def test_resume_traversal_rejected(self):
        with self.assertRaises(ValueError):
            self.manager.resume('../another-folder')
