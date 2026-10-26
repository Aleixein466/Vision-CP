import json
import base64
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class PilotFormatTests(unittest.TestCase):
    def test_yolo_coordinate_contract(self):
        row = "39 0.5 0.5 0.25 0.25".split()
        self.assertEqual(len(row), 5)
        self.assertTrue(all(0 <= float(v) <= 1 for v in row[1:]))

    def test_pillow_verifies_generated_jpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jpg"
            Image.new("RGB", (64, 64)).save(path)
            with Image.open(path) as im:
                im.verify()

    def test_iou_identity_and_disjoint(self):
        from scripts.evaluate_phase1 import iou
        self.assertEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_phase1_configuration_has_requested_classes(self):
        import yaml
        root = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load((root / "training/configs/phase1.yaml").read_text(encoding="utf-8"))
        self.assertEqual(set(cfg["target_classes"].values()), {"laptop", "cell phone", "bottle", "book", "car"})

    def test_spanish_speech_phrases(self):
        from scripts.speech_announcer import SPANISH, spanish_phrase
        self.assertEqual(spanish_phrase("bottle", 1), "Veo una botella")
        self.assertEqual(spanish_phrase("book", 3), "Veo 3 libros")
        import yaml
        root = Path(__file__).resolve().parents[1]
        names = yaml.safe_load((root / "training/configs/phase1_coco_val.yaml").read_text(encoding="utf-8"))["names"]
        self.assertEqual(set(names.values()), set(SPANISH))

    def test_web_assistant_assets_and_launcher(self):
        root = Path(__file__).resolve().parents[1]
        package = json.loads((root / "package.json").read_text(encoding="utf-8"))
        self.assertEqual(package["scripts"]["dev"], "python scripts/start_assistant.py --model smolvlm-500m")
        for relative_path in ("web/index.html", "web/styles.css", "web/enhancements.css", "web/app.js", "web/scene-tracker.js"):
            self.assertTrue((root / relative_path).is_file(), relative_path)
        html = (root / "web/index.html").read_text(encoding="utf-8")
        app = (root / "web/app.js").read_text(encoding="utf-8")
        self.assertIn('id="liveDetection"', html)
        self.assertIn('id="videoFile"', html)
        self.assertIn('id="autoComment"', html)
        self.assertIn('id="yoloScale"', html)
        self.assertIn('id="yoloTask"', html)
        self.assertIn('id="datasetSection"', html)
        self.assertIn('id="visionMode" type="hidden" value="general"', html)
        self.assertNotIn('Manos y objetos pequeños</option>', html)
        self.assertIn('scene-tracker.js', html)
        self.assertEqual(package["scripts"]["tracking:test"], "node scripts/test_scene_tracking.js")
        self.assertEqual(package["scripts"]["train:own"], "python scripts/train_visionar.py --execute")
        self.assertIn("renderObjects(data, !imageMode)", app)
        self.assertIn("imageMode ? 0.18 : 0.32", app)

    def test_visual_fusion_decodes_real_image_bytes(self):
        from scripts.visual_fusion import crop_image_data_url, decode_image_data_url
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.jpg"
            Image.new("RGB", (64, 48), (10, 20, 30)).save(path)
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            image = decode_image_data_url(f"data:image/jpeg;base64,{encoded}")
            self.assertEqual(image.shape[:2], (48, 64))
            cropped = decode_image_data_url(crop_image_data_url(
                f"data:image/jpeg;base64,{encoded}",
                {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5}, padding=0,
            ))
            self.assertEqual(cropped.shape[:2], (24, 32))

    def test_vlm_catalog_reports_installed_visual_models(self):
        from scripts.vlm_engine import VLMEngine
        catalog = VLMEngine().catalog()
        self.assertEqual({item["id"] for item in catalog}, {"qwen2-vl-2b", "smolvlm-500m"})
        self.assertTrue(all(item["vision"] for item in catalog))
        self.assertTrue(all(item["installed"] for item in catalog))

    def test_yolo_catalog_exposes_five_scales_and_segmentation(self):
        from scripts.visual_fusion import VisualFusionEngine
        catalog = VisualFusionEngine().model_catalog()
        self.assertEqual(len(catalog), 10)
        self.assertEqual({item["scale"] for item in catalog}, {"n", "s", "m", "l", "x"})
        self.assertEqual({item["task"] for item in catalog}, {"detect", "segment"})
        self.assertTrue(next(item for item in catalog if item["scale"] == "n" and item["task"] == "segment")["installed"])
        nano_detect = next(item for item in catalog if item["scale"] == "n" and item["task"] == "detect")
        nano_segment = next(item for item in catalog if item["scale"] == "n" and item["task"] == "segment")
        self.assertEqual(nano_detect["filename"], "yolo26n-objv1-150.pt")
        self.assertEqual(nano_segment["filename"], "yolo26n-objv1-seg.pt")
        self.assertEqual(nano_detect["vocabulary"], "Objects365 (365 clases)")
        small_detect = next(item for item in catalog if item["scale"] == "s" and item["task"] == "detect")
        small_segment = next(item for item in catalog if item["scale"] == "s" and item["task"] == "segment")
        self.assertEqual(small_detect["filename"], "yolo26s-objv1-150.pt")
        self.assertEqual(small_segment["filename"], "yolo26s-objv1-seg.pt")
        self.assertTrue(small_detect["installed"])
        self.assertTrue(small_segment["installed"])

    def test_cpu_runtime_is_bounded(self):
        from scripts.runtime_limits import configure_runtime
        configure_runtime()
        import cv2
        import torch
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(torch.get_num_interop_threads(), 1)
        self.assertEqual(cv2.getNumThreads(), 1)

    def test_own_dataset_studio_initializes_without_fake_samples(self):
        from scripts.dataset_studio import DatasetStudio, CLASSES
        with tempfile.TemporaryDirectory() as tmp:
            studio = DatasetStudio(Path(tmp))
            stats = studio.stats()
            self.assertEqual(stats["total_images"], 0)
            self.assertEqual(stats["annotations"], 0)
            self.assertEqual(stats["classes"], CLASSES)
            requested = {
                "person", "eyeglasses", "shirt", "nail_clipper", "cell_phone", "perfume_bottle",
                "potted_plant", "chair", "marker", "eraser", "tv", "fish", "cat", "dog",
                "motorcycle", "pencil_case", "wallet", "payment_card", "fork", "spoon", "knife",
                "pen", "backpack", "handbag", "playing_card", "adhesive_tape", "screwdriver",
                "paper_sheet", "table", "sofa", "cap", "sweatshirt", "sneakers", "wardrobe",
                "pillow", "keys", "usb_drive", "banknote",
            }
            self.assertTrue(requested.issubset(set(CLASSES)))
            self.assertTrue((Path(tmp) / "data.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
