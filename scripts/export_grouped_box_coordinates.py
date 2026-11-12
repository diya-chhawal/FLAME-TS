from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_box_design import (
    DATASET_ROOT,
    OUTPUT_DIR,
    load_config,
    load_manifest,
    retained_components,
    strategy_boxes,
)


def main() -> None:
    config = load_config()
    rows = load_manifest()
    exported = []
    for row in rows:
        if row["label"] != "Fire":
            continue
        with Image.open(DATASET_ROOT / row["thermal_tiff"]) as image:
            values = np.asarray(image, dtype=np.float32)
        for threshold, minimum_area in config["pilot_configurations"]:
            components = retained_components(values, threshold, minimum_area, config["connectivity"])
            for distance in config["grouping_distances_px"]:
                boxes = strategy_boxes(components, "spatial_grouped", distance)
                for group_id, (x1, y1, x2, y2) in enumerate(boxes):
                    exported.append({
                        "stem": row["stem"],
                        "threshold_c": threshold,
                        "minimum_component_area": minimum_area,
                        "grouping_distance_px": distance,
                        "group_id": group_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    })
    output_path = OUTPUT_DIR / "grouped_box_coordinates.csv"
    fields = ["stem", "threshold_c", "minimum_component_area", "grouping_distance_px", "group_id", "x1", "y1", "x2", "y2"]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(exported)
    print(f"Export={output_path}")
    print(f"Rows={len(exported)}")


if __name__ == "__main__":
    main()