import os
import gc
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

CHECKPOINT = "checkpoints/sam_vit_b_01ec64.pth"
IMAGE_DIR = r"data\raw\FLAME 3 CV Dataset (Sycan Marsh)\Fire\RGB\Corrected FOV"

images = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if not images:
    raise FileNotFoundError("No Fire corrected-FOV images found.")

image_path = os.path.join(IMAGE_DIR, images[0])

print(f"Test image: {image_path}")
print("Loading SAM ViT-B...")

sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
sam.to("cuda")
sam.eval()

print("Model loaded: True")
print(f"GPU: {torch.cuda.get_device_name(0)}")

image = cv2.imread(image_path)

if image is None:
    raise RuntimeError(f"Could not read image: {image_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"Image shape: {image.shape}")

predictor = SamPredictor(sam)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("Running SAM image encoder...")

try:
    predictor.set_image(image)

    torch.cuda.synchronize()

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("SAM image encoding: SUCCESS")
    print(f"Embedding shape: {predictor.get_image_embedding().shape}")
    print(f"Peak VRAM: {peak_vram:.3f} GiB")

except RuntimeError as e:
    print("SAM image encoding: FAILED")
    print(f"Error: {e}")

finally:
    del predictor
    del sam
    gc.collect()
    torch.cuda.empty_cache()

print("GPU memory cleaned up.")