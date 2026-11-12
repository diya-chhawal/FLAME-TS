import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from models.pidnet import get_seg_model
from scripts.flame3_pidnet_dataset import FLAME3SegmentationDataset
from scripts.pidnet_loss import PIDNetLoss


SEED = 42
BATCH_SIZE = 2
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-2
CHECKPOINT_DIR = Path("checkpoints/pidnet_flame3")
VIS_DIR = Path("reports/pidnet_overfit_diagnostic")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(logits, targets, threshold=0.5):
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = torch.nn.functional.interpolate(
            logits,
            size=targets.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    predictions = (torch.sigmoid(logits) >= threshold).float()

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

    return iou.mean().item(), dice.mean().item()


def save_prediction_visualization(
    image_tensor,
    target_tensor,
    prediction_tensor,
    stem,
    epoch,
):
    """
    Save a 3-panel visualization:

        RGB | pseudo-label | PIDNet prediction
    """

    image = (
        image_tensor.detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = image * std + mean
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    target = (
        target_tensor.detach()
        .cpu()
        .squeeze()
        .numpy()
    )

    prediction = (
        prediction_tensor.detach()
        .cpu()
        .squeeze()
        .numpy()
    )

    target = (target * 255).astype(np.uint8)
    prediction = (prediction * 255).astype(np.uint8)

    rgb_image = Image.fromarray(image, mode="RGB")
    target_image = Image.fromarray(target, mode="L")
    prediction_image = Image.fromarray(prediction, mode="L")

    height, width = rgb_image.height, rgb_image.width

    panel = Image.new(
        "RGB",
        (width * 3, height),
        "white",
    )

    panel.paste(rgb_image, (0, 0))
    panel.paste(
        target_image.convert("RGB"),
        (width, 0),
    )
    panel.paste(
        prediction_image.convert("RGB"),
        (width * 2, 0),
    )

    VIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        VIS_DIR
        / f"epoch_{epoch:02d}_{stem}.png"
    )

    panel.save(output_path)


def save_epoch_predictions(model, dataset, device, epoch):
    """
    Run the current model on all 8 diagnostic training samples
    and save RGB / pseudo-label / prediction visualizations.
    """

    model.eval()

    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]

            image = sample["image"].unsqueeze(0).to(device)
            target = sample["mask"]

            outputs = model(image)
            logits = outputs[1]

            if logits.shape[-2:] != target.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            prediction = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            save_prediction_visualization(
                image_tensor=sample["image"],
                target_tensor=target,
                prediction_tensor=prediction[0],
                stem=sample["stem"],
                epoch=epoch,
            )


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("device:", device)

    # --------------------
    # Dataset
    # --------------------

    trainset = FLAME3SegmentationDataset(
        manifest_path="reports/phase3_split/train_dev_manifest.csv",
        pseudolabel_dir="reports/phase3_pseudolabels/full",
        split="train_dev",
    )

    valset = FLAME3SegmentationDataset(
        manifest_path="reports/phase3_split/dev_manifest.csv",
        pseudolabel_dir="reports/phase3_pseudolabels/full",
        split="dev",
    )

    trainloader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    valloader = DataLoader(
        valset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print("train samples:", len(trainset))
    print("validation samples:", len(valset))

    # --------------------
    # Model
    # --------------------

    class Config:
        pass

    cfg = Config()
    cfg.MODEL = Config()
    cfg.MODEL.NAME = "pidnet_s"

    model = get_seg_model(
        cfg,
        imgnet_pretrained=False,
    ).to(device)

    checkpoint_path = Path(
        "pretrained_models/imagenet/PIDNet_S_ImageNet.pth.tar"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    pretrained_state = checkpoint["state_dict"]
    model_state = model.state_dict()

    matched_state = {
        k: v
        for k, v in pretrained_state.items()
        if k in model_state
        and v.shape == model_state[k].shape
    }

    model_state.update(matched_state)

    model.load_state_dict(
        model_state,
        strict=False,
    )

    print(
        "ImageNet pretrained parameters loaded:",
        len(matched_state),
    )

    # --------------------
    # Loss / optimizer
    # --------------------

    criterion = PIDNetLoss().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_iou = -1.0

    # --------------------
    # Training
    # --------------------

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0
        train_iou = 0.0
        train_dice = 0.0
        train_batches = 0

        for batch_idx, batch in enumerate(trainloader):

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            boundaries = batch["boundary"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(images)

            losses = criterion(
                outputs,
                masks,
                boundaries,
            )

            batch_iou, batch_dice = compute_metrics(
                outputs[1].detach(),
                masks,
            )

            train_iou += batch_iou
            train_dice += batch_dice
            train_batches += 1

            losses["total"].backward()
            optimizer.step()

            train_loss += losses["total"].item()

            if batch_idx == 0:
                print("first train batch:")
                print(
                    "  image:",
                    tuple(images.shape),
                )
                print(
                    "  mask:",
                    tuple(masks.shape),
                )
                print(
                    "  boundary:",
                    tuple(boundaries.shape),
                )
                print(
                    "  outputs:",
                    [tuple(x.shape) for x in outputs],
                )
                print(
                    "  loss total:",
                    losses["total"].item(),
                )
                print(
                    "  loss seg_aux:",
                    losses["seg_aux"].item(),
                )
                print(
                    "  loss seg_main:",
                    losses["seg_main"].item(),
                )
                print(
                    "  loss boundary:",
                    losses["boundary"].item(),
                )
                print(
                    "  loss boundary_semantic:",
                    losses["boundary_semantic"].item(),
                )
        train_loss /= len(trainloader)
        train_iou /= train_batches
        train_dice /= train_batches

        # --------------------
        # Validation
        # --------------------

        model.eval()

        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        val_batches = 0

        with torch.no_grad():

            for batch in valloader:

                images = batch["image"].to(
                    device,
                    non_blocking=True,
                )

                masks = batch["mask"].to(
                    device,
                    non_blocking=True,
                )

                boundaries = batch["boundary"].to(
                    device,
                    non_blocking=True,
                )

                outputs = model(images)

                losses = criterion(
                    outputs,
                    masks,
                    boundaries,
                )

                val_loss += losses["total"].item()

                iou, dice = compute_metrics(
                    outputs[1],
                    masks,
                )

                val_iou += iou
                val_dice += dice
                val_batches += 1

        val_loss /= len(valloader)
        val_iou /= val_batches
        val_dice /= val_batches

        print()
        print(
            f"epoch {epoch + 1} complete"
        )
        print(
            "train loss:",
            train_loss,
        )
        print(
            "train IoU:",
            train_iou,
        )
        print(
            "train Dice:",
            train_dice,
        )
        print(
            "validation loss:",
            val_loss,
        )
        print(
            "validation IoU:",
            val_iou,
        )
        print(
            "validation Dice:",
            val_dice,
        )

        # --------------------
        # Checkpoint
        # --------------------

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_iou": train_iou,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice,
        }

        torch.save(
            checkpoint,
            CHECKPOINT_DIR / "last.pt",
        )

        if val_iou > best_val_iou:

            best_val_iou = val_iou

            torch.save(
                checkpoint,
                CHECKPOINT_DIR / "best.pt",
            )

            print("saved: best.pt")

        # --------------------
        # Save visualizations
        # --------------------

        save_epoch_predictions(
            model=model,
            dataset=trainset,
            device=device,
            epoch=epoch + 1,
        )

        print(
            "saved visualizations:",
            VIS_DIR,
        )


if __name__ == "__main__":
    main()