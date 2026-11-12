from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


METADATA_PATH = Path("reports/phase1/image_metadata.csv")
OUTPUT_DIR = Path("reports/phase3_split")
MANIFEST_OUTPUT = OUTPUT_DIR / "split_manifest.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "split_summary.json"

EXPECTED_COUNTS = {
    "train": {"total": 517, "Fire": 435, "No Fire": 82},
    "validation": {"total": 111, "Fire": 95, "No Fire": 16},
    "test": {"total": 110, "Fire": 92, "No Fire": 18},
}


def load_quartets() -> list[dict[str, str]]:
    with METADATA_PATH.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["modality"] == "corrected_rgb"
        ]
    return sorted(
        rows,
        key=lambda row: datetime.strptime(row["exif_datetime"], "%Y:%m:%d %H:%M:%S"),
    )


def assign_blocks(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    blocks: list[list[dict[str, str]]] = []
    for row in rows:
        timestamp = datetime.strptime(row["exif_datetime"], "%Y:%m:%d %H:%M:%S")
        if not blocks:
            blocks.append([row])
            continue
        previous = datetime.strptime(
            blocks[-1][-1]["exif_datetime"], "%Y:%m:%d %H:%M:%S"
        )
        if (timestamp - previous).total_seconds() >= 60:
            blocks.append([row])
        else:
            blocks[-1].append(row)
    return [
        {
            "id": index,
            "rows": block,
            "count": len(block),
            "fire": sum(row["label"] == "Fire" for row in block),
            "no_fire": sum(row["label"] == "No Fire" for row in block),
        }
        for index, block in enumerate(blocks, start=1)
    ]


def assign_splits(blocks: list[dict[str, object]]) -> dict[tuple[str, str], str]:
    assignments: dict[tuple[str, str], str] = {}

    def assign_rows(rows: list[dict[str, str]], split: str) -> None:
        for row in rows:
            key = (row["label"], row["stem"])
            if key in assignments:
                raise ValueError(f"Duplicate assignment for quartet {key}")
            assignments[key] = split

    block_by_id = {int(block["id"]): block for block in blocks}
    for block_id, split in {
        7: "train",
        1: "train",
        3: "train",
        5: "train",
        8: "train",
        9: "train",
        10: "train",
        11: "train",
        14: "train",
        4: "validation",
        12: "validation",
        6: "test",
        13: "test",
    }.items():
        assign_rows(block_by_id[block_id]["rows"], split)  # type: ignore[arg-type]

    b2_rows = block_by_id[2]["rows"]  # type: ignore[assignment]
    runs: list[list[dict[str, str]]] = []
    for row in b2_rows:
        if not runs or row["label"] != runs[-1][-1]["label"]:
            runs.append([row])
        else:
            runs[-1].append(row)
    if len(runs) != 12:
        raise ValueError(f"Expected 12 B2 runs, found {len(runs)}")
    for run_index in (1, 2):
        assign_rows(runs[run_index - 1], "test")
    for run_index in range(3, 9):
        assign_rows(runs[run_index - 1], "validation")
    for run_index in range(9, 13):
        assign_rows(runs[run_index - 1], "train")
    return assignments


def validate_and_write(rows: list[dict[str, str]], assignments: dict[tuple[str, str], str]) -> None:
    keys = [(row["label"], row["stem"]) for row in rows]
    if len(rows) != 738 or len(set(keys)) != 738:
        raise ValueError(f"Expected 738 unique quartets, found rows={len(rows)}, unique={len(set(keys))}")
    if set(keys) != set(assignments):
        raise ValueError("Some quartets are missing or unassigned")

    output_rows = []
    for row in rows:
        output_rows.append(
            {
                "label": row["label"],
                "stem": row["stem"],
                "timestamp": datetime.strptime(
                    row["exif_datetime"], "%Y:%m:%d %H:%M:%S"
                ).isoformat(sep=" "),
                "split": assignments[(row["label"], row["stem"])],
            }
        )
    counts = {}
    for split in ("train", "validation", "test"):
        split_rows = [row for row in output_rows if row["split"] == split]
        counts[split] = {
            "total": len(split_rows),
            "Fire": sum(row["label"] == "Fire" for row in split_rows),
            "No Fire": sum(row["label"] == "No Fire" for row in split_rows),
        }
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Split counts mismatch: {counts}")

    timestamp_values = [datetime.fromisoformat(row["timestamp"]) for row in output_rows]
    duplicate_timestamps = [
        timestamp for timestamp, count in Counter(timestamp_values).items() if count > 1
    ]
    cross_split_timestamps = []
    for timestamp in sorted(set(timestamp_values)):
        split_values = {row["split"] for row in output_rows if datetime.fromisoformat(row["timestamp"]) == timestamp}
        if len(split_values) > 1:
            cross_split_timestamps.append(timestamp.isoformat(sep=" "))

    ranges = {}
    for split in ("train", "validation", "test"):
        split_times = [
            datetime.fromisoformat(row["timestamp"])
            for row in output_rows
            if row["split"] == split
        ]
        ranges[split] = {
            "minimum": min(split_times).isoformat(sep=" "),
            "maximum": max(split_times).isoformat(sep=" "),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "stem", "timestamp", "split"])
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "source": str(METADATA_PATH),
        "temporal_block_gap_seconds": 60,
        "counts": counts,
        "timestamp_ranges": ranges,
        "duplicate_timestamps": [value.isoformat(sep=" ") for value in duplicate_timestamps],
        "cross_split_identical_timestamps": cross_split_timestamps,
        "unique_quartets": len(set(keys)),
        "missing_or_duplicate_stems_within_label": False,
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    rows = load_quartets()
    blocks = assign_blocks(rows)
    if [int(block["id"]) for block in blocks] != list(range(1, 15)):
        raise ValueError("Expected temporal blocks B01 through B14")
    assignments = assign_splits(blocks)
    validate_and_write(rows, assignments)


if __name__ == "__main__":
    main()