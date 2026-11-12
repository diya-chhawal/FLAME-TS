from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage


DATASET_ROOT = Path("data/raw/FLAME 3 CV Dataset (Sycan Marsh)")
MANIFEST_PATH = Path("reports/phase1/quartet_manifest.csv")
CONFIG_PATH = Path("configs/phase2_75_box_design.json")
OUTPUT_DIR = Path("reports/phase2_75")
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 512


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    distances = tuple(int(value) for value in config["grouping_distances_px"])
    if not distances or any(value < 0 for value in distances):
        raise ValueError("grouping distances must be non-negative")
    return {
        "pilot_configurations": [
            (float(item["threshold_c"]), int(item["minimum_component_area"]))
            for item in config["pilot_configurations"]
        ],
        "grouping_distances_px": distances,
        "connectivity": int(config["connectivity"]),
        "excessive_box_area_fraction": float(config["excessive_box_area_fraction"]),
        "visualization_cases": int(config["visualization_cases"]),
    }


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if any(row["quartet_status"] != "complete" for row in rows):
        raise ValueError("Phase 1 manifest contains incomplete quartets")
    return rows


def retained_components(values: np.ndarray, threshold: float, minimum_area: int, connectivity: int) -> list[tuple[int, int, int, int, int]]:
    structure = np.ones((3, 3), dtype=np.uint8) if connectivity == 8 else ndimage.generate_binary_structure(2, 1)
    labels, count = ndimage.label(values >= threshold, structure=structure)
    components = []
    for component_id in range(1, count + 1):
        coordinates = np.argwhere(labels == component_id)
        area = len(coordinates)
        if area < minimum_area:
            continue
        y_min, x_min = coordinates.min(axis=0)
        y_max, x_max = coordinates.max(axis=0)
        components.append((int(x_min), int(y_min), int(x_max), int(y_max), int(area)))
    return components


def box_from_components(components: list[tuple[int, int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(component[0] for component in components),
        min(component[1] for component in components),
        max(component[2] for component in components),
        max(component[3] for component in components),
    )


def box_area(box: tuple[int, int, int, int]) -> int:
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def largest_boxes(components: list[tuple[int, int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    return [box_from_components([max(components, key=lambda component: component[4])])] if components else []


def union_boxes(components: list[tuple[int, int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    return [box_from_components(components)] if components else []


def boxes_gap(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    horizontal = max(left[0] - right[2] - 1, right[0] - left[2] - 1, 0)
    vertical = max(left[1] - right[3] - 1, right[1] - left[3] - 1, 0)
    return float((horizontal * horizontal + vertical * vertical) ** 0.5)


def grouped_boxes(components: list[tuple[int, int, int, int, int]], distance: int) -> list[tuple[int, int, int, int]]:
    if not components:
        return []
    parents = list(range(len(components)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    component_boxes = [component[:4] for component in components]
    for left in range(len(components)):
        for right in range(left + 1, len(components)):
            if boxes_gap(component_boxes[left], component_boxes[right]) <= distance:
                union(left, right)
    groups: dict[int, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    for index, component in enumerate(components):
        groups[find(index)].append(component)
    return [box_from_components(group) for group in groups.values()]


def box_metrics(boxes: list[tuple[int, int, int, int]], excessive_fraction: float) -> dict[str, Any]:
    areas = [box_area(box) for box in boxes]
    image_area = IMAGE_WIDTH * IMAGE_HEIGHT
    return {
        "valid_box_count": len(boxes),
        "has_valid_box": bool(boxes),
        "mean_box_area_percent": float(np.mean(areas) * 100 / image_area) if areas else 0.0,
        "median_box_area_percent": float(np.median(areas) * 100 / image_area) if areas else 0.0,
        "boxes_touching_image_border": sum(box[0] == 0 or box[1] == 0 or box[2] == IMAGE_WIDTH - 1 or box[3] == IMAGE_HEIGHT - 1 for box in boxes),
        "boxes_excessively_large": sum(area / image_area >= excessive_fraction for area in areas),
        "boxes": boxes,
    }


def strategy_boxes(components: list[tuple[int, int, int, int, int]], strategy: str, distance: int | None = None) -> list[tuple[int, int, int, int]]:
    if strategy == "largest_component":
        return largest_boxes(components)
    if strategy == "union":
        return union_boxes(components)
    if strategy == "spatial_grouped":
        if distance is None:
            raise ValueError("spatial_grouped requires a distance")
        return grouped_boxes(components, distance)
    raise ValueError(strategy)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    grouping_keys = {
        (
            row["threshold_c"],
            row["minimum_component_area"],
            row["strategy"],
            row["grouping_distance_px"],
        )
        for row in rows
    }
    for threshold, minimum_area, strategy, distance in sorted(grouping_keys):
            selected = [
                row for row in rows
                if row["threshold_c"] == threshold
                and row["minimum_component_area"] == minimum_area
                and row["strategy"] == strategy
                and row["grouping_distance_px"] == distance
            ]
            summary.append({
                "threshold_c": threshold,
                "minimum_component_area": minimum_area,
                "strategy": strategy,
                "grouping_distance_px": distance,
                "fire_samples_with_valid_box": sum(row["has_valid_box"] for row in selected if row["label"] == "Fire"),
                "fire_samples_total": sum(row["label"] == "Fire" for row in selected),
                "fire_mean_boxes": float(np.mean([row["valid_box_count"] for row in selected if row["label"] == "Fire"])),
                "fire_median_boxes": float(np.median([row["valid_box_count"] for row in selected if row["label"] == "Fire"])),
                "fire_mean_box_area_percent": float(np.mean([row["mean_box_area_percent"] for row in selected if row["label"] == "Fire"])),
                "fire_median_box_area_percent": float(np.median([row["median_box_area_percent"] for row in selected if row["label"] == "Fire"])),
                "fire_boxes_touching_border": sum(row["boxes_touching_image_border"] for row in selected if row["label"] == "Fire"),
                "fire_boxes_excessively_large": sum(row["boxes_excessively_large"] for row in selected if row["label"] == "Fire"),
                "no_fire_samples_with_boxes": sum(row["has_valid_box"] for row in selected if row["label"] == "No Fire"),
            })
    return summary


def render_case(row: dict[str, str], values: np.ndarray, components: list[tuple[int, int, int, int, int]], boxes: list[tuple[int, int, int, int]], threshold: float, minimum_area: int, strategy: str, distance: int | None, name: str) -> None:
    with Image.open(DATASET_ROOT / row["corrected_rgb"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    mask = values >= threshold
    retained = np.zeros_like(mask, dtype=bool)
    for x_min, y_min, x_max, y_max, _area in components:
        retained[y_min : y_max + 1, x_min : x_max + 1] |= mask[y_min : y_max + 1, x_min : x_max + 1]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    figure.suptitle(f"{row['label']} {row['stem']} | {strategy} | {threshold:g} C + {minimum_area} px")
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Corrected-FOV RGB")
    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title("Thermal threshold mask")
    axes[1, 0].imshow(retained, cmap="gray")
    axes[1, 0].set_title("Retained components")
    axes[1, 1].imshow(rgb)
    axes[1, 1].imshow(np.ma.masked_where(~retained, retained), cmap="spring", alpha=0.35)
    for x_min, y_min, x_max, y_max in boxes:
        axes[1, 1].add_patch(plt.Rectangle((x_min, y_min), x_max - x_min + 1, y_max - y_min + 1, fill=False, edgecolor="cyan", linewidth=2))
    axes[1, 1].set_title(f"Boxes ({len(boxes)})" + (f", gap <= {distance}px" if distance is not None else ""))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.savefig(OUTPUT_DIR / "visualizations" / f"{name}.png", dpi=140)
    plt.close(figure)


def main() -> None:
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "visualizations").mkdir(exist_ok=True)
    rows = load_manifest()
    metrics = []
    grouped_box_coordinates = []
    components_by_key: dict[tuple[str, str, float, int], list[tuple[int, int, int, int, int]]] = {}
    values_by_key: dict[tuple[str, str], np.ndarray] = {}
    for row in rows:
        key = (row["label"], row["stem"])
        with Image.open(DATASET_ROOT / row["thermal_tiff"]) as image:
            values = np.asarray(image, dtype=np.float32)
        values_by_key[key] = values
        for threshold, minimum_area in config["pilot_configurations"]:
            components = retained_components(values, threshold, minimum_area, config["connectivity"])
            components_by_key[(row["label"], row["stem"], threshold, minimum_area)] = components
            strategies = [("largest_component", None), ("union", None)] + [("spatial_grouped", distance) for distance in config["grouping_distances_px"]]
            for strategy, distance in strategies:
                boxes = strategy_boxes(components, strategy, distance)
                if row["label"] == "Fire" and strategy == "spatial_grouped":
                    for group_id, (x1, y1, x2, y2) in enumerate(boxes):
                        grouped_box_coordinates.append({
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
                row_metrics = box_metrics(boxes, config["excessive_box_area_fraction"])
                metrics.append({"label": row["label"], "stem": row["stem"], "threshold_c": threshold, "minimum_component_area": minimum_area, "strategy": strategy, "grouping_distance_px": distance if distance is not None else "", **{key: value for key, value in row_metrics.items() if key != "boxes"}})
    with_boxes = summarize(metrics)
    with (OUTPUT_DIR / "box_strategy_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (OUTPUT_DIR / "box_strategy_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(with_boxes[0]))
        writer.writeheader()
        writer.writerows(with_boxes)
    with (OUTPUT_DIR / "grouped_box_coordinates.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["stem", "threshold_c", "minimum_component_area", "grouping_distance_px", "group_id", "x1", "y1", "x2", "y2"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(grouped_box_coordinates)

    cases = []
    for threshold, minimum_area in config["pilot_configurations"]:
        grouped = [
            item for item in metrics
            if item["label"] == "Fire"
            and item["threshold_c"] == threshold
            and item["minimum_component_area"] == minimum_area
            and item["strategy"] == "spatial_grouped"
            and item["grouping_distance_px"] == max(config["grouping_distances_px"])
        ]
        cases.extend([
            (max(grouped, key=lambda item: item["valid_box_count"])["stem"], "high_fragmentation"),
            (max(grouped, key=lambda item: item["mean_box_area_percent"])["stem"], "large_contiguous"),
            (min((item for item in grouped if item["has_valid_box"]), key=lambda item: item["mean_box_area_percent"])["stem"], "weak_small"),
            (max((item for item in grouped if item["valid_box_count"] >= 2), key=lambda item: item["valid_box_count"])["stem"], "multiple_regions"),
        ])
        no_fire = [
            item for item in metrics
            if item["label"] == "No Fire"
            and item["threshold_c"] == threshold
            and item["minimum_component_area"] == minimum_area
            and item["strategy"] == "spatial_grouped"
            and item["grouping_distance_px"] == max(config["grouping_distances_px"])
        ]
        cases.append((max(no_fire, key=lambda item: item["valid_box_count"])["stem"], "no_fire_edge"))
    for threshold, minimum_area in config["pilot_configurations"]:
        for stem, category in cases:
            if len([item for item in cases if item[1] == category]) > config["visualization_cases"] and category not in {"high_fragmentation", "large_contiguous", "weak_small", "multiple_regions", "no_fire_edge"}:
                continue
            label = "No Fire" if category == "no_fire_edge" else "Fire"
            components = components_by_key[(label, stem, threshold, minimum_area)]
            for strategy, distance in [("largest_component", None), ("union", None), ("spatial_grouped", max(config["grouping_distances_px"]))]:
                boxes = strategy_boxes(components, strategy, distance)
                render_case(rows[next(index for index, row in enumerate(rows) if row["label"] == label and row["stem"] == stem)], values_by_key[(label, stem)], components, boxes, threshold, minimum_area, strategy, distance, f"{category}_{int(threshold)}c_{minimum_area}px_{stem}_{strategy}")

    report = {"config": str(CONFIG_PATH), "metric_rows": len(metrics), "summary_rows": len(with_boxes), "visualization_count": len(list((OUTPUT_DIR / "visualizations").glob("*.png"))), "strategies": ["largest_component", "union", "spatial_grouped"], "grouping_rule": "Components are joined when Euclidean gap between their axis-aligned bounding boxes is at most the configured distance; grouping is transitive.", "excessive_box_area_fraction": config["excessive_box_area_fraction"], "boundary": "This is a box-design experiment only. No SAM masks, final pseudo-labels, model training, or permanent parameter selection were performed."}
    (OUTPUT_DIR / "box_design_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()