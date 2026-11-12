from __future__ import annotations

import csv
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


DATASET_ROOT = Path("data/raw/FLAME 3 CV Dataset (Sycan Marsh)")
MANIFEST_PATH = Path("reports/phase1/quartet_manifest.csv")
BOX_CSV_PATH = Path("reports/phase2_75/grouped_box_coordinates.csv")
OUTPUT_DIR = Path("reports/phase3_grouped_sam_pilot")
CHECKPOINT = Path("checkpoints/sam_vit_b_01ec64.pth")
THRESHOLD_C = 80.0
MINIMUM_COMPONENT_AREA = 50
GROUPING_GAP_PX = 25
CASES = {
    "00001": "easy",
    "00002": "weak",
    "00003": "fragmented",
    "00004": "multi_region",
    "00007": "border_touching",
}


def load_manifest() -> dict[str, dict[str, str]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        return {row["stem"]: row for row in rows if row["label"] == "Fire"}


def load_grouped_boxes() -> dict[str, list[tuple[int, int, int, int]]]:
    grouped: dict[str, list[tuple[int, int, int, int]]] = {stem: [] for stem in CASES}
    with BOX_CSV_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (
                row["stem"] in grouped
                and float(row["threshold_c"]) == THRESHOLD_C
                and int(row["minimum_component_area"]) == MINIMUM_COMPONENT_AREA
                and int(row["grouping_distance_px"]) == GROUPING_GAP_PX
            ):
                grouped[row["stem"]].append(
                    (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
                )
    missing = [stem for stem, boxes in grouped.items() if not boxes]
    if missing:
        raise ValueError(f"Missing CSV grouped boxes for stems: {missing}")
    return grouped


def save_mask(path: Path, mask: np.ndarray) -> None:
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def draw_overlay(image: np.ndarray, boxes: list[tuple[int, int, int, int]], mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255
    overlay = np.where(mask[:, :, None], (0.45 * green + 0.55 * overlay).astype(np.uint8), overlay)
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 0), 2)
    return overlay


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    manifest = load_manifest()
    boxes_by_stem = load_grouped_boxes()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT)).to("cuda")
    model.eval()
    predictor = SamPredictor(model)
    results = []
    try:
        for stem, category in CASES.items():
            row = manifest[stem]
            case_dir = OUTPUT_DIR / f"fire_{stem}_{category}"
            case_dir.mkdir(parents=True, exist_ok=True)
            rgb_path = DATASET_ROOT / row["corrected_rgb"]
            image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                results.append({"stem": stem, "category": category, "failure": "input_read_failed"})
                continue
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            boxes = boxes_by_stem[stem]
            cv2.imwrite(str(case_dir / "rgb_corrected_fov.JPG"), image_bgr)
            metadata = {
                "stem": stem,
                "category": category,
                "threshold_c": THRESHOLD_C,
                "minimum_component_area": MINIMUM_COMPONENT_AREA,
                "grouping_gap_px": GROUPING_GAP_PX,
                "grouped_box_count": len(boxes),
                "grouped_boxes_xyxy": [list(box) for box in boxes],
                "rgb_shape": list(image.shape),
                "box_source": str(BOX_CSV_PATH),
                "alignment_verified": bool(image.shape[:2] == (512, 640)),
                "failure": None,
            }
            with (case_dir / "grouped_boxes.json").open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2)
            combined_mask = np.zeros((512, 640), dtype=bool)
            scores = []
            predictor.set_image(image)
            torch.cuda.synchronize()
            start = time.perf_counter()
            box_times = []
            for index, box in enumerate(boxes, start=1):
                box_start = time.perf_counter()
                masks, box_scores, _logits = predictor.predict(
                    box=np.asarray(box, dtype=np.float32), multimask_output=False
                )
                torch.cuda.synchronize()
                box_times.append(time.perf_counter() - box_start)
                mask = masks[0].astype(bool)
                combined_mask |= mask
                scores.append(float(box_scores[0]))
                save_mask(case_dir / f"sam_mask_box_{index:02d}.png", mask)
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - start
            save_mask(case_dir / "sam_mask_combined.png", combined_mask)
            cv2.imwrite(
                str(case_dir / "rgb_grouped_boxes_sam_overlay.png"),
                cv2.cvtColor(draw_overlay(image_bgr, boxes, combined_mask), cv2.COLOR_RGB2BGR),
            )
            metadata.update({
                "sam_mask_area_percent": float(combined_mask.mean() * 100),
                "sam_mask_box_areas_percent": [
                    float(cv2.imread(str(case_dir / f"sam_mask_box_{index:02d}.png"), cv2.IMREAD_GRAYSCALE).astype(bool).mean() * 100)
                    for index in range(1, len(boxes) + 1)
                ],
                "confidence_scores": scores,
                "predicted_iou_scores": scores,
                "box_inference_seconds": box_times,
                "inference_seconds": inference_seconds,
                "mask_valid": bool(combined_mask.any()),
            })
            with (case_dir / "grouped_boxes.json").open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2)
            results.append(metadata)
    finally:
        del predictor
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report = {
        "model": "vit_b",
        "checkpoint": str(CHECKPOINT),
        "device": "cuda",
        "threshold_c": THRESHOLD_C,
        "minimum_component_area": MINIMUM_COMPONENT_AREA,
        "grouping_gap_px": GROUPING_GAP_PX,
        "results": results,
        "comparison_reference": "reports/phase3_sam_pilot",
    }
    (OUTPUT_DIR / "grouped_pilot_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(results), "successful": sum(item.get("failure") is None for item in results), "output": str(OUTPUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()