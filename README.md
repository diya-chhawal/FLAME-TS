# FLAME-TS: UAV Wildfire Segmentation

## 1. Project Overview

FLAME-TS is a UAV wildfire segmentation project that adapts a teacher–student learning methodology to the **FLAME 3 Computer Vision Subset (Sycan Marsh)**.

The project is based on the WACV 2025 research paper:

**Detecting Wildfires on UAVs with Real-time Segmentation Trained by Larger Teacher Models**

The primary objective is to generate segmentation pseudo-labels using thermal-derived bounding boxes and the Segment Anything Model (SAM), and subsequently train a lightweight PIDNet-S student model for RGB-based wildfire segmentation.

### Proposed Pipeline

```text
FLAME 3 RGB + Radiometric Thermal Images
                  |
                  v
       Thermal Temperature Thresholding
                  |
                  v
       Connected Component Analysis
                  |
                  v
       Weak Bounding Box Generation
                  |
                  v
          SAM ViT-B Teacher
                  |
                  v
        Pseudo Segmentation Masks
                  |
                  v
           PIDNet-S Student
                  |
                  v
          Fire Segmentation
```

**Important:** Thermal-derived regions are weak spatial prompts, not ground-truth segmentation annotations. SAM-generated masks are pseudo-labels and should not be treated as manually verified segmentation ground truth.

---

## 2. Research References

### Primary Research Paper

Pesonen et al., WACV 2025.

_Detecting Wildfires on UAVs with Real-time Segmentation Trained by Larger Teacher Models._

### Reference Implementation

[Official WACV Implementation](https://gitlab.com/fgi_nls/public/wildfire-real-time-segmentation)

### Dataset

[FLAME 3 Computer Vision Subset – Sycan Marsh](https://www.kaggle.com/datasets/brycehopkins/flame-3-computer-vision-subset-sycan-marsh)

### Official FLAME 3 Processing Repository

[Flame-Data-Pipeline](https://github.com/BryceHopkins14/Flame-Data-Pipeline)

---

## 3. Dataset Description

The project uses the FLAME 3 Computer Vision Subset collected at Sycan Marsh, Oregon.

The dataset contains 738 image quartets.

| Category  | Number of Samples |
| --------- | ----------------: |
| Fire      |               622 |
| No Fire   |               116 |
| **Total** |           **738** |

Each sample consists of four corresponding files:

1. Raw RGB image
2. Corrected-FOV RGB image
3. Thermal JPG image
4. Radiometric Celsius TIFF image

### Image Properties

| Modality            | Resolution  | Format |
| ------------------- | ----------- | ------ |
| Raw RGB             | 4000 × 3000 | JPG    |
| Corrected-FOV RGB   | 640 × 512   | JPG    |
| Thermal JPG         | 640 × 512   | JPG    |
| Radiometric Thermal | 640 × 512   | TIFF   |

The corrected-FOV RGB images are used as the input to the PIDNet-S student model.

Radiometric TIFF images contain per-pixel temperature values in Celsius and are used to derive weak spatial supervision.

### Dataset Directory Structure

```text
data/raw/FLAME 3 CV Dataset (Sycan Marsh)/
│
├── Fire/
│   ├── RGB/
│   │   ├── Raw/
│   │   └── Corrected FOV/
│   │
│   └── Thermal/
│       ├── Raw JPG/
│       └── Celsius TIFF/
│
└── No Fire/
    ├── RGB/
    │   ├── Raw/
    │   └── Corrected FOV/
    │
    └── Thermal/
        ├── Raw JPG/
        └── Celsius TIFF/
```

The original dataset is not included in this GitHub repository because of its size. Download it separately and place it under `data/raw/`.

---

## 4. Hardware and Development Environment

The project was developed and tested on the following hardware:

| Component          | Specification                      |
| ------------------ | ---------------------------------- |
| Laptop             | ASUS VivoBook 15 Pro OLED          |
| Processor          | AMD Ryzen 7 5800HS                 |
| CPU Cores          | 8 cores / 16 threads               |
| GPU                | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU VRAM           | 4 GB                               |
| RAM                | 16 GB DDR4                         |
| Operating System   | Windows                            |
| Python             | 3.11                               |
| PyTorch            | 2.6.0+cu124                        |
| Torchvision        | 0.21.0+cu124                       |
| CUDA compatibility | 12.7                               |
| SciPy              | 1.17.1                             |

### Environment Setup

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

The current PyTorch dependency is specific to Windows, Python 3.11 and CUDA 12.4. Other hardware configurations may require a different PyTorch installation.

### Segment Anything Installation

The project uses Meta's Segment Anything implementation.

Install it using:

```powershell
pip install git+https://github.com/facebookresearch/segment-anything.git
```

---

## 5. Repository Structure

```text
FLAME-TS/
│
├── configs/
│   ├── phase2_5_component_analysis.json
│   ├── phase2_75_box_design.json
│   └── thermal_analysis.json
│
├── models/
│   ├── model_utils.py
│   ├── pidnet.py
│   └── __init__.py
│
├── scripts/
│   ├── inspect_dataset.py
│   ├── analyze_thermal.py
│   ├── analyze_components.py
│   ├── analyze_box_design.py
│   ├── find_temporal_split.py
│   ├── create_split_manifest.py
│   ├── create_dev_split.py
│   ├── export_grouped_box_coordinates.py
│   ├── run_sam_pilot.py
│   ├── run_grouped_sam_pilot.py
│   ├── generate_sam_pseudolabels.py
│   ├── flame3_pidnet_dataset.py
│   ├── pidnet_loss.py
│   ├── train_pidnet.py
│   ├── evaluate_thresholds.py
│   ├── diagnose_pidnet.py
│   └── visualize_test_predictions.py
│
├── reports/
│   ├── phase1/
│   ├── phase2/
│   ├── phase2_5/
│   ├── phase2_75/
│   ├── phase3_sam_pilot/
│   ├── phase3_grouped_sam_pilot/
│   ├── phase3_split/
│   └── phase3_pseudolabels/
│
├── data/
│   └── raw/                  # Download separately
│
├── checkpoints/              # Download separately
│
├── pretrained_models/        # Download separately
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 6. Experimental Phases

### Phase 1: Dataset Inspection

**Objective:** Understand the dataset structure, verify modality pairing and inspect image properties.

Implemented:

- Dataset directory inspection.
- Image metadata extraction.
- Quartet pairing validation.
- RGB and thermal image dimension verification.
- Radiometric TIFF temperature analysis.
- Timestamp extraction and analysis.

Generated artifacts:

```text
scripts/inspect_dataset.py

reports/phase1/
├── dataset_tree.txt
├── image_metadata.csv
├── quartet_manifest.csv
└── group_identifier_report.json
```

Key findings:

- 738 complete image quartets.
- 2,952 total files.
- No missing modalities.
- No invalid quartets.
- No duplicate content hashes.
- Corrected RGB images have a resolution of 640 × 512.
- Radiometric TIFF images contain float32 temperature values.

### Phase 2: Thermal Analysis

**Objective:** Identify suitable thermal thresholds for distinguishing fire-related regions from background.

Tested thresholds:

- 40°C
- 60°C
- 80°C
- 100°C
- 150°C
- 200°C

Eight-connected component analysis was used to investigate the spatial distribution of threshold-exceeding pixels.

Key findings:

| Threshold | Fire Samples Detected | No Fire Samples Detected |
| --------- | --------------------: | -----------------------: |
| 40°C      |               622/622 |                    7/116 |
| 60°C      |               622/622 |                    0/116 |
| 80°C      |               622/622 |                    0/116 |
| 100°C     |               622/622 |                    0/116 |
| 150°C     |               621/622 |                    0/116 |
| 200°C     |               538/622 |                    0/116 |

The candidate threshold range was narrowed to 60–100°C.

Generated artifacts:

```text
scripts/analyze_thermal.py

reports/phase2/
```

### Phase 2.5: Connected Component Filtering

**Objective:** Remove small thermal regions and identify suitable component-filtering configurations.

Experiment configurations:

| Parameter              | Values                        |
| ---------------------- | ----------------------------- |
| Thermal thresholds     | 60°C, 80°C, 100°C             |
| Minimum component area | 10, 50, 100, 500, 1000 pixels |
| Connectivity           | 8-connected                   |

Three configurations were selected for further comparison:

| Configuration     | Fire Samples with Valid Regions | No Fire Valid Regions |
| ----------------- | ------------------------------: | --------------------: |
| 60°C + 100 pixels |                         610/622 |                     0 |
| 80°C + 50 pixels  |                         612/622 |                     0 |
| 100°C + 10 pixels |                         620/622 |                     0 |

Generated artifacts:

```text
scripts/analyze_components.py

reports/phase2_5/
```

### Phase 2.75: Bounding Box Design

**Objective:** Compare different methods of converting thermal connected components into SAM bounding box prompts.

Three strategies were investigated:

1. Largest-component bounding box
2. Union bounding box
3. Spatially grouped bounding boxes

Spatial grouping was tested with distances of 10, 25 and 50 pixels.

#### Final Strategy Selection

**Primary strategy: Largest-component bounding box**

**Secondary strategy: Grouped bounding boxes with 25-pixel grouping distance**

Reasoning:

- Largest-component boxes produced cleaner and more conservative prompts.
- Grouped boxes could capture multiple spatially separated fire regions.
- Grouped boxes also introduced additional background leakage.
- Union boxes frequently enclosed excessive background areas.

Generated artifacts:

```text
scripts/analyze_box_design.py
scripts/export_grouped_box_coordinates.py

reports/phase2_75/
├── box_strategy_metrics.csv
├── box_strategy_summary.csv
├── box_design_report.json
└── grouped_box_coordinates.csv
```

### Phase 3: SAM Teacher and Pseudo-Label Generation

**Objective:** Use thermal-derived bounding boxes to prompt SAM and generate RGB segmentation pseudo-labels.

#### SAM Model Selection

The original WACV paper used SAM ViT-H.

However, the available GPU has only 4 GB VRAM.

Measured peak GPU memory requirements:

| SAM Model | Peak VRAM | Feasibility  |
| --------- | --------: | ------------ |
| ViT-H     | 5.604 GiB | Not feasible |
| ViT-L     | 4.298 GiB | Not feasible |
| ViT-B     | 2.705 GiB | Feasible     |

Therefore, SAM ViT-B was selected.

This is a hardware-driven adaptation of the original paper, not an exact reproduction.

#### Pilot Experiments

A 24-sample pilot was conducted to examine SAM output quality across different fire conditions.

The pilot included:

- Easy fire regions
- Weak fire regions
- Fragmented fire regions
- Multiple fire regions
- Border-touching fire regions
- No Fire control samples

A grouped-box pilot was also conducted using five representative samples.

The final prompt-strategy decision was:

- Primary: Largest-component box.
- Secondary ablation: Grouped-25-pixel boxes.

#### Final Pseudo-Label Configuration

| Parameter              | Value                    |
| ---------------------- | ------------------------ |
| RGB input              | Corrected-FOV RGB        |
| Thermal input          | Radiometric Celsius TIFF |
| Thermal threshold      | 80°C                     |
| Minimum component area | 50 pixels                |
| Connectivity           | 8-connected              |
| Bounding box strategy  | Largest component        |
| Teacher model          | SAM ViT-B                |
| Device                 | CUDA                     |

#### Pseudo-Label Generation Results

| Category                                 | Count |
| ---------------------------------------- | ----: |
| Successful Fire SAM masks                |   612 |
| Fire samples without valid thermal boxes |    10 |
| No Fire explicit background masks        |   116 |
| Total usable pseudo-labels               |   728 |
| Processing failures                      |     0 |

No Fire samples without valid thermal boxes were assigned explicit all-zero background masks.

Fire samples without valid thermal boxes were excluded from pseudo-label training.

Generated artifacts:

```text
scripts/generate_sam_pseudolabels.py

reports/phase3_pseudolabels/full/
├── metadata.csv
├── summary.json
├── summary_by_split.json
├── train/
├── validation/
└── test/
```

The generated masks are excluded from Git because of their size. They must be transferred separately if another developer needs to continue training without regenerating them.

---

## 7. Dataset Splitting

The original split was designed to preserve temporal structure while ensuring both classes were represented.

### Original Frozen Split

| Split      |   Total |    Fire | No Fire |
| ---------- | ------: | ------: | ------: |
| Train      |     517 |     435 |      82 |
| Validation |     111 |      95 |      16 |
| Test       |     110 |      92 |      18 |
| **Total**  | **738** | **622** | **116** |

The split manifest is:

```text
reports/phase3_split/split_manifest.csv
```

### Representative Development Split

During training analysis, the original validation set was found to contain substantially smaller fire regions than the training set.

To investigate this distribution mismatch, a representative development split was created from the original training data.

The original validation and test sets were left untouched.

| Split                  | Total Manifest Samples | Fire | No Fire |
| ---------------------- | ---------------------: | ---: | ------: |
| Train development      |                    440 |  370 |      70 |
| Development validation |                     77 |   65 |      12 |

After excluding Fire samples without valid SAM masks:

- Effective training samples: 433.
- Effective development samples: 76.

Generated artifacts:

```text
scripts/create_split_manifest.py
scripts/create_dev_split.py

reports/phase3_split/
├── split_manifest.csv
├── split_summary.json
├── dev_manifest.csv
└── train_dev_manifest.csv
```

**Important:** The original test set is frozen. Do not modify it or use it for model selection or threshold tuning.

---

## 8. PIDNet-S Student Model

**Objective:** Train a lightweight RGB segmentation model using SAM-generated pseudo-labels.

The student architecture is PIDNet-S, adapted from the official WACV reference implementation.

### Model Configuration

| Parameter                  | Value                     |
| -------------------------- | ------------------------- |
| Architecture               | PIDNet-S                  |
| Parameters                 | 7,623,393                 |
| Input channels             | 3                         |
| Output classes             | 1                         |
| Input resolution           | 640 × 512                 |
| Training output resolution | 80 × 64 before upsampling |
| Training initialization    | ImageNet pretrained       |

The model uses three training outputs:

- Auxiliary semantic output.
- Main semantic output.
- Boundary output.

### Dataset Loader

Implemented in:

```text
scripts/flame3_pidnet_dataset.py
```

The loader:

- Reads corrected-FOV RGB images.
- Reads SAM-generated pseudo-label masks.
- Normalizes RGB images using ImageNet mean and standard deviation.
- Generates boundary targets using the WACV reference methodology.
- Excludes Fire samples without valid pseudo-labels.
- Retains explicit all-zero No Fire masks.

### Loss Function

Implemented in:

```text
scripts/pidnet_loss.py
```

The loss follows the released WACV implementation:

- Auxiliary semantic loss.
- Main semantic loss.
- Boundary loss.
- Boundary-aware semantic loss.

The combined loss is:

```text
Total Loss =
    0.4 × Auxiliary Semantic Loss
  + 20  × Main Semantic Loss
  + 1   × Boundary Loss
  + 1   × Boundary-aware Semantic Loss
```

Semantic loss uses binary cross-entropy with positive class weight 2.5.

Boundary loss uses class-balanced weighted binary cross-entropy with coefficient 20.

The boundary-aware semantic loss uses a predicted boundary threshold of 0.8.

### Training Configuration

| Parameter            | Value                        |
| -------------------- | ---------------------------- |
| Optimizer            | AdamW                        |
| Learning rate        | 0.0001                       |
| Weight decay         | 0.01                         |
| Batch size           | 2                            |
| Epochs               | 40                           |
| Random seed          | 42                           |
| Initialization       | ImageNet pretrained PIDNet-S |
| Checkpoint selection | Development IoU              |

Training script:

```text
scripts/train_pidnet.py
```

---

## 9. Training and Evaluation Results

### Development Results

The best development checkpoint was obtained at epoch 39.

| Metric                    |  Result |
| ------------------------- | ------: |
| Best epoch                |      39 |
| Development IoU           | 0.33844 |
| Development Dice          | 0.40009 |
| Training IoU at epoch 39  | 0.72575 |
| Training Dice at epoch 39 | 0.81753 |

The gap between training and development performance indicates overfitting.

### Frozen Test Results

The epoch-39 checkpoint was evaluated on the original frozen test set.

Threshold evaluation was performed at values from 0.1 to 0.9.

The predetermined 0.5 threshold is retained as the main reported result because it was not selected using test data.

| Metric                                 |  Result |
| -------------------------------------- | ------: |
| Test samples                           |     109 |
| Overall IoU                            | 0.17045 |
| Overall Dice                           | 0.17409 |
| Fire-only IoU                          | 0.00637 |
| Fire-only Dice                         | 0.01072 |
| Fire samples with empty predictions    |   48/91 |
| No Fire samples with empty predictions |   18/18 |

### Interpretation

The overall test metrics are misleading because empty predictions on No Fire samples receive perfect IoU and Dice when both prediction and target are empty.

The Fire-only IoU of approximately 0.0064 indicates a severe generalization failure.

The model performs substantially worse on the frozen test set than on the development split.

Therefore, the current PIDNet-S checkpoint should be considered an unsuccessful generalization baseline, not a completed wildfire segmentation solution.

---

## 10. Known Limitations

1. The dataset contains only 738 samples from a single geographical region.
2. Thermal thresholding provides weak spatial supervision rather than pixel-level ground truth.
3. SAM-generated masks are not manually verified segmentation annotations.
4. SAM can under-cover weak or fragmented fire regions.
5. SAM can include background vegetation when prompted with imperfect thermal boxes.
6. Fire regions occupy a small fraction of the image.
7. Training, development and test fire-region distributions differ.
8. The current PIDNet-S model exhibits substantial overfitting.
9. Frozen test Fire-only segmentation performance is extremely poor.
10. The current model is not ready for deployment or claims of successful generalization.

---

## 11. Next Steps

The next developer should continue from the current baseline rather than restart the project.

### Immediate Tasks

1. Complete qualitative inspection of the 12 representative frozen test Fire samples.
2. Identify whether failures are associated with fire size, scene characteristics, or pseudo-label quality.
3. Compare SAM pseudo-labels with PIDNet predictions.
4. Investigate training, development and test distribution differences.
5. Examine the effect of small foreground regions and class imbalance.

### Further Experiments

1. Investigate pseudo-label quality and filtering.
2. Investigate training augmentation and sampling strategies.
3. Investigate loss weighting and foreground learning.
4. Evaluate alternative training configurations using development data.
5. Compare each experiment against the epoch-39 baseline.
6. Evaluate the final selected model on the frozen test set only after development decisions are complete.

Do not tune thresholds or select checkpoints based on frozen test performance.

---

## 12. Important Files for Continuation

| Purpose                     | File                                            |
| --------------------------- | ----------------------------------------------- |
| Dataset inspection          | `scripts/inspect_dataset.py`                    |
| Thermal analysis            | `scripts/analyze_thermal.py`                    |
| Component analysis          | `scripts/analyze_components.py`                 |
| Box strategy analysis       | `scripts/analyze_box_design.py`                 |
| Split creation              | `scripts/create_split_manifest.py`              |
| Development split           | `scripts/create_dev_split.py`                   |
| SAM pseudo-label generation | `scripts/generate_sam_pseudolabels.py`          |
| PIDNet dataset loader       | `scripts/flame3_pidnet_dataset.py`              |
| PIDNet loss                 | `scripts/pidnet_loss.py`                        |
| PIDNet training             | `scripts/train_pidnet.py`                       |
| Threshold evaluation        | `scripts/evaluate_thresholds.py`                |
| Test visualization          | `scripts/visualize_test_predictions.py`         |
| PIDNet architecture         | `models/pidnet.py`                              |
| Dataset manifest            | `reports/phase1/quartet_manifest.csv`           |
| Box strategy metrics        | `reports/phase2_75/box_strategy_metrics.csv`    |
| Frozen split manifest       | `reports/phase3_split/split_manifest.csv`       |
| Development manifest        | `reports/phase3_split/dev_manifest.csv`         |
| SAM metadata                | `reports/phase3_pseudolabels/full/metadata.csv` |

---

## 13. Reproducibility and Handoff Notes

Before continuing:

- Download the FLAME 3 dataset separately.
- Install dependencies using `requirements.txt`.
- Download the SAM ViT-B checkpoint.
- Download the PIDNet-S ImageNet pretrained checkpoint.
- Transfer the generated SAM pseudo-label masks separately if they are to be reused.
- Transfer the epoch-39 PIDNet checkpoint separately if it is to be used as the baseline.
- Preserve the original frozen split manifest.
- Keep all new experiments separate from previous reports.
- Record the configuration and metrics for every experiment.
- Do not report development metrics as test metrics.
- Do not treat SAM pseudo-labels as ground truth.

**Current project status:** Thermal analysis, weak bounding-box design, SAM pseudo-label generation, dataset preparation, and initial PIDNet-S training have been completed. The current model has a serious test-set generalization problem. Further investigation and experimentation are required.

---

## 14. Acknowledgements

This project builds upon the FLAME 3 dataset and adapts the teacher–student wildfire segmentation methodology presented in the WACV 2025 research paper.

- FLAME 3 dataset and official processing pipeline.
- Meta Segment Anything Model.
- PIDNet architecture and WACV reference implementation.
