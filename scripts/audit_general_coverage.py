"""Audita qué clases propias cubre el YOLO26 Objects365 instalado."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

try:
    from runtime_limits import configure_runtime
    from dataset_studio import CLASSES
except ModuleNotFoundError:
    from scripts.runtime_limits import configure_runtime
    from scripts.dataset_studio import CLASSES

configure_runtime()

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "training/checkpoints/yolo26n-objv1-150.pt"
OUTPUT = ROOT / "training/metrics/general_coverage.json"

ALIASES = {
    "person": ["person"], "eyeglasses": ["glasses"], "cell_phone": ["cell phone"],
    "potted_plant": ["potted plant"], "chair": ["chair"], "marker": ["marker"],
    "eraser": ["eraser", "board eraser"], "tv": ["tv"], "fish": ["fish", "goldfish"],
    "cat": ["cat"], "dog": ["dog"], "motorcycle": ["motorcycle"],
    "pencil_case": ["pencil case"], "fork": ["fork"], "spoon": ["spoon"],
    "knife": ["knife"], "pen": ["pen/pencil"], "backpack": ["backpack"],
    "handbag": ["handbag"], "playing_card": ["poker card"], "adhesive_tape": ["tape"],
    "screwdriver": ["screwdriver"], "paper_sheet": ["notepaper"],
    "table": ["dining table", "coffee table"], "sofa": ["couch"], "cap": ["hat"],
    "sneakers": ["sneakers"], "pillow": ["pillow"],
}


def main() -> None:
    model = YOLO(str(MODEL))
    model_names = {str(value).lower() for value in model.names.values()}
    native = {}
    missing = []
    for class_name in CLASSES:
        matches = [name for name in ALIASES.get(class_name, []) if name in model_names]
        if matches:
            native[class_name] = matches
        else:
            missing.append(class_name)
    report = {
        "audited_at": datetime.now().astimezone().isoformat(),
        "model": str(MODEL.relative_to(ROOT)).replace("\\", "/"),
        "pretrained_vocabulary_size": len(model_names),
        "requested_classes": len(CLASSES),
        "native_coverage_count": len(native),
        "native_coverage": native,
        "requires_own_data_or_vlm": missing,
        "attributes_handled_by_vlm": [
            "garment color", "visible shirt pattern", "object color", "visible material",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
