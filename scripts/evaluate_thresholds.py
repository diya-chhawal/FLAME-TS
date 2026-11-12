import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from models.pidnet import get_pred_model
from scripts.flame3_pidnet_dataset import FLAME3SegmentationDataset


CHECKPOINT = "checkpoints/pidnet_flame3/best.pt"
MANIFEST = "reports/phase3_split/split_manifest.csv"
PSEUDOLABEL_DIR = "reports/phase3_pseudolabels/full"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_model():
    model = get_pred_model("pidnet_s", 1)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    state_dict = checkpoint["model_state_dict"]

    # Training checkpoint contains auxiliary PIDNet heads
    # that are not present in inference mode.
    model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()

    return model


def calculate_metrics(predictions, targets):
    predictions = predictions.view(predictions.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (predictions * targets).sum(dim=1)
    pred_area = predictions.sum(dim=1)
    target_area = targets.sum(dim=1)

    union = pred_area + target_area - intersection

    iou = torch.where(
        union > 0,
        intersection / union,
        torch.ones_like(union),
    )

    dice_denominator = pred_area + target_area

    dice = torch.where(
        dice_denominator > 0,
        2.0 * intersection / dice_denominator,
        torch.ones_like(dice_denominator),
    )

    return iou, dice


dataset = FLAME3SegmentationDataset(
    MANIFEST,
    PSEUDOLABEL_DIR,
    "test",
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

model = load_model()

print(f"test samples: {len(dataset)}")


for threshold in THRESHOLDS:

    all_iou = []
    all_dice = []

    fire_iou = []
    fire_dice = []
    fire_zero = 0

    nofire_pred_area = []
    nofire_zero = 0
    nofire_count = 0

    overall_zero = 0

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            labels = batch["label"]
            stems = batch["stem"]

            outputs = model(images)

            # PIDNet main segmentation output.
            logits = outputs

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            predictions = (
                torch.sigmoid(logits) >= threshold
            ).float()

            iou, dice = calculate_metrics(
                predictions,
                masks,
            )

            all_iou.extend(iou.cpu().tolist())
            all_dice.extend(dice.cpu().tolist())

            for i in range(len(labels)):

                label = labels[i]
                pred = predictions[i]

                pred_pixels = pred.sum().item()

                if pred_pixels == 0:
                    overall_zero += 1

                if label == "Fire":

                    fire_iou.append(iou[i].item())
                    fire_dice.append(dice[i].item())

                    if pred_pixels == 0:
                        fire_zero += 1

                elif label == "No Fire":

                    nofire_count += 1
                    nofire_pred_area.append(pred_pixels)

                    if pred_pixels == 0:
                        nofire_zero += 1

    overall_iou = sum(all_iou) / len(all_iou)
    overall_dice = sum(all_dice) / len(all_dice)

    fire_mean_iou = sum(fire_iou) / len(fire_iou)
    fire_mean_dice = sum(fire_dice) / len(fire_dice)

    nofire_zero_rate = (
        nofire_zero / nofire_count
        if nofire_count
        else 0.0
    )

    mean_nofire_pixels = (
        sum(nofire_pred_area) / len(nofire_pred_area)
        if nofire_pred_area
        else 0.0
    )

    print()
    print(f"threshold={threshold:.1f}")

    print(
        f"  overall IoU={overall_iou:.6f} "
        f"Dice={overall_dice:.6f}"
    )

    print(
        f"  Fire IoU={fire_mean_iou:.6f} "
        f"Dice={fire_mean_dice:.6f} "
        f"zero_pred={fire_zero}/{len(fire_iou)}"
    )

    print(
        f"  No Fire zero_pred={nofire_zero}/{nofire_count} "
        f"({nofire_zero_rate:.3f})"
    )

    print(
        f"  No Fire mean_pred_pixels={mean_nofire_pixels:.1f}"
    )

    print(
        f"  overall_zero_pred={overall_zero}/{len(dataset)}"
    )