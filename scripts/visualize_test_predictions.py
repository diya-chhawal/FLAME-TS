import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.pidnet import get_pred_model
from scripts.flame3_pidnet_dataset import FLAME3SegmentationDataset


CHECKPOINT = "checkpoints/pidnet_flame3/best.pt"
MANIFEST = "reports/phase3_split/split_manifest.csv"
PSEUDOLABEL_DIR = "reports/phase3_pseudolabels/full"
OUTPUT_DIR = Path("reports/pidnet_test_diagnostic")
PSEUDOLABEL_METADATA = "reports/phase3_pseudolabels/full/metadata.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5


def load_model():
    model = get_pred_model("pidnet_s", 1)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )

    model.to(DEVICE)
    model.eval()

    return model


def denormalize_image(image):
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    image = image * std[:, None, None]
    image = image + mean[:, None, None]

    image = np.clip(
        image * 255.0,
        0,
        255,
    ).astype(np.uint8)

    return np.transpose(image, (1, 2, 0))


def load_test_fire_stems():
    split_manifest = pd.read_csv(
        MANIFEST,
        dtype={"stem": str},
    )

    metadata = pd.read_csv(
        PSEUDOLABEL_METADATA,
        dtype={"stem": str},
    )

    # Select Fire samples belonging to the frozen test split.
    test_fire = split_manifest[
        (split_manifest["split"] == "test")
        & (split_manifest["label"] == "Fire")
    ].copy()

    # Join pseudo-label information using both label and stem.
    # This avoids collisions if a stem exists in both classes.
    fire_manifest = test_fire.merge(
        metadata,
        on=["label", "stem", "split"],
        how="inner",
        suffixes=("", "_metadata"),
    )

    # Keep only samples with successfully generated SAM masks.
    fire_manifest = fire_manifest[
        fire_manifest["status"] == "success"
    ].copy()

    if "thermal_candidate_percent" not in fire_manifest.columns:
        raise ValueError(
            "thermal_candidate_percent missing after metadata merge. "
            f"Available columns: {fire_manifest.columns.tolist()}"
        )

    print(f"Usable test Fire samples: {len(fire_manifest)}")

    return fire_manifest

def select_representative_samples(fire_manifest):
    areas = fire_manifest["thermal_candidate_percent"]

    fire_manifest = fire_manifest.copy()

    fire_manifest["size_group"] = pd.qcut(
        areas,
        q=3,
        labels=["small", "medium", "large"],
        duplicates="drop",
    )

    selected = []

    for group in ["small", "medium", "large"]:
        group_df = fire_manifest[
            fire_manifest["size_group"] == group
        ].sort_values("thermal_candidate_percent")

        if len(group_df) == 0:
            continue

        indices = np.linspace(
            0,
            len(group_df) - 1,
            min(4, len(group_df)),
            dtype=int,
        )

        selected.append(
            group_df.iloc[indices]
        )

    selected = pd.concat(selected)

    return selected


def load_rgb(stem):
    path = (
        Path("data/raw/FLAME 3 CV Dataset (Sycan Marsh)")
        / "Fire"
        / "RGB"
        / "Corrected FOV"
        / f"{stem}.JPG"
    )

    image = Image.open(path).convert("RGB")

    return np.asarray(image)


def save_visualization(
    rgb,
    pseudo_mask,
    prediction,
    stem,
    thermal_area,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    axes[0].imshow(rgb)
    axes[0].set_title("RGB")

    axes[1].imshow(
        pseudo_mask,
        cmap="gray",
    )
    axes[1].set_title("SAM pseudo-label")

    axes[2].imshow(
        prediction,
        cmap="gray",
    )
    axes[2].set_title("PIDNet prediction")

    axes[3].imshow(rgb)
    axes[3].imshow(
        prediction,
        alpha=0.45,
    )
    axes[3].set_title("RGB + PIDNet")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Test Fire | {stem} | "
        f"thermal candidate area={thermal_area:.3f}%",
        fontsize=12,
    )

    output_path = (
        OUTPUT_DIR
        / f"{stem}_diagnostic.png"
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    fire_manifest = load_test_fire_stems()

    selected = select_representative_samples(
        fire_manifest
    )

    print(
        f"Selected {len(selected)} test Fire samples."
    )

    print(
        selected[
            [
                "stem",
                "thermal_candidate_percent",
                "size_group",
            ]
        ].to_string(index=False)
    )

    dataset = FLAME3SegmentationDataset(
        MANIFEST,
        PSEUDOLABEL_DIR,
        "test",
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    model = load_model()

    selected_keys = set(
        zip(
            selected["label"].astype(str),
            selected["stem"].astype(str),
        )
    )

    with torch.no_grad():

        for batch in loader:

            stem = str(batch["stem"][0])
            label = str(batch["label"][0])

            if (label, stem) not in selected_keys:
                continue

            image = batch["image"].to(DEVICE)

            mask = batch["mask"][0, 0].cpu().numpy()

            logits = model(image)

            if logits.shape[-2:] != mask.shape:
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=mask.shape,
                    mode="bilinear",
                    align_corners=False,
                )

            prediction = (
                torch.sigmoid(logits)[0, 0]
                >= THRESHOLD
            ).cpu().numpy()

            rgb = denormalize_image(
                batch["image"][0].cpu().numpy()
            )

            row = selected[
                (selected["label"].astype(str) == label)
                & (selected["stem"].astype(str) == stem)
            ].iloc[0]

            save_visualization(
                rgb=rgb,
                pseudo_mask=mask,
                prediction=prediction,
                stem=stem,
                thermal_area=row[
                    "thermal_candidate_percent"
                ],
            )

            print(
                f"saved: {stem}"
            )


if __name__ == "__main__":
    main()