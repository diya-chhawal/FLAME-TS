from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FLAME3SegmentationDataset(Dataset):
    """
    FLAME 3 RGB segmentation dataset using SAM pseudo-labels.

    RGB input:
        corrected-FOV RGB, 640x512

    Target:
        SAM pseudo-mask, 640x512

    Boundary target:
        WACV reference implementation:
        Canny -> crop 6px -> pad 6px -> 4x4 dilation.

    RGB preprocessing:
        [0, 1] scaling followed by ImageNet normalization.
    """

    def __init__(
        self,
        manifest_path,
        pseudolabel_dir,
        split,
        image_root="data/raw/FLAME 3 CV Dataset (Sycan Marsh)",
    ):
        self.manifest_path = Path(manifest_path)
        self.pseudolabel_dir = Path(pseudolabel_dir)
        self.image_root = Path(image_root)
        self.split = split

        manifest = pd.read_csv(self.manifest_path)

        metadata_path = self.pseudolabel_dir / "metadata.csv"
        metadata = pd.read_csv(metadata_path)

        manifest = manifest[
            manifest["split"] == self.split
        ].copy()

        # Join the frozen split assignment with SAM generation metadata.
        manifest = manifest.merge(
            metadata[
                ["label", "stem", "status", "mask_path"]
            ],
            on=["label", "stem"],
            how="left",
            validate="one_to_one",
        )

        # Only samples with an actual SAM pseudo-mask are usable.
# Keep:
#   - Fire samples with successful SAM pseudo-labels
#   - No Fire samples with explicit all-zero background masks
#
# Exclude Fire samples where no valid thermal prompt was available.
        usable = (
            manifest["status"].eq("success")
            | manifest["status"].eq("no_valid_box_background")
        )

        manifest = manifest[usable].copy()

        manifest = manifest.sort_values(
            ["timestamp", "label", "stem"]
        ).reset_index(drop=True)

        if len(manifest) == 0:
            raise RuntimeError(
                f"No usable segmentation samples found for split={split}"
            )

        self.manifest = manifest

    def __len__(self):
        return len(self.manifest)

    def _rgb_path(self, label, stem):
        return (
            self.image_root
            / label
            / "RGB"
            / "Corrected FOV"
            / f"{int(stem):05d}.JPG"
        )

    def _mask_path(self, row):
        mask_path = Path(str(row["mask_path"]))

        if not mask_path.is_absolute():
            mask_path = Path.cwd() / mask_path

        return mask_path

    @staticmethod
    def _make_boundary(mask):
        """
        Reproduce the WACV boundary-target construction.
        """
        edge = cv2.Canny(
            np.uint8(mask),
            0.1,
            0.2,
        )

        kernel = np.ones((4, 4), np.uint8)

        edge = edge[6:-6, 6:-6]
        edge = np.pad(
            edge,
            ((6, 6), (6, 6)),
            mode="constant",
        )

        edge = (
            cv2.dilate(
                edge,
                kernel,
                iterations=1,
            )
            > 50
        ).astype(np.float32)

        return edge

    def __getitem__(self, index):
        row = self.manifest.iloc[index]

        label = str(row["label"])
        stem = int(row["stem"])

        rgb_path = self._rgb_path(label, stem)
        mask_path = self._mask_path(row)

        if not rgb_path.exists():
            raise FileNotFoundError(
                f"RGB image not found: {rgb_path}"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Pseudo-mask not found: {mask_path}"
            )

        image = cv2.imread(
            str(rgb_path),
            cv2.IMREAD_COLOR,
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise RuntimeError(
                f"Failed to read RGB image: {rgb_path}"
            )

        if mask is None:
            raise RuntimeError(
                f"Failed to read pseudo-mask: {mask_path}"
            )

        if image.shape[:2] != (512, 640):
            raise RuntimeError(
                f"Unexpected RGB shape for {rgb_path}: "
                f"{image.shape}"
            )

        if mask.shape != (512, 640):
            raise RuntimeError(
                f"Unexpected mask shape for {mask_path}: "
                f"{mask.shape}"
            )

        # OpenCV BGR -> RGB.
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # Binary pseudo-label.
        mask = (mask > 127).astype(np.float32)

        boundary = self._make_boundary(mask)

        # HWC -> CHW and scale to [0, 1].
        image = (
            torch.from_numpy(image.copy())
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        # ImageNet normalization for ImageNet-pretrained PIDNet.
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32,
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32,
        ).view(3, 1, 1)

        image = (image - mean) / std

        mask = torch.from_numpy(
            mask.copy()
        ).unsqueeze(0)

        boundary = torch.from_numpy(
            boundary.copy()
        ).unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "boundary": boundary,
            "label": label,
            "stem": f"{stem:05d}",
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "validation", "test"],
    )

    args = parser.parse_args()

    dataset = FLAME3SegmentationDataset(
        manifest_path="reports/phase3_split/split_manifest.csv",
        pseudolabel_dir="reports/phase3_pseudolabels/full",
        split=args.split,
    )

    print(f"split: {args.split}")
    print(f"samples: {len(dataset)}")

    sample = dataset[0]

    print("image:", tuple(sample["image"].shape))
    print("mask:", tuple(sample["mask"].shape))
    print("boundary:", tuple(sample["boundary"].shape))
    print("label:", sample["label"])
    print("stem:", sample["stem"])

    print(
        "image range:",
        float(sample["image"].min()),
        float(sample["image"].max()),
    )

    print(
        "mask values:",
        torch.unique(sample["mask"]).tolist(),
    )

    print(
        "boundary values:",
        torch.unique(sample["boundary"]).tolist(),
    )