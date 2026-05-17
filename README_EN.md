# CASE-AI Quality Inspection Enhanced - Steel Welding Defect Detection System

An industrial quality inspection system based on **YOLO + VLM**, integrating the Baidu Steel Defect Detection Competition high score solution (score **33.48**,a score exceeding 30 is excent), supporting defect detection, visual analysis, data export, and model iterative optimization.

## 📋 Project Overview

This project detects six major categories of steel surface defects using the YOLO12 model architecture, combined with VLM (Vision-Language Model) for interpretable defect analysis. The training scheme is optimized based on the Baidu Steel Defect Detection Learning Competition, achieving a final score of **33.48**.

### Competition Background

Defect detection technology is widely used in industrial scenarios, such as body surface defect detection in automobile manufacturing, part appearance defect detection, and workpiece crack detection. Among them, metal surface defect recognition technology can play an important role in quality control during production and manufacturing stages.

This dataset comes from the **NEU Surface Defect Detection Dataset**, which collects 6 typical hot-rolled strip steel surface defects: rolled-in scale (RS), patches (Pa), crazing (Cr), pitted surface (PS), inclusion (In), and scratches (Sc).

There are differences between different categories of defects, such as patches, crazing, and pitted surface. There are also variations within the same defect category. For example, scratches can be horizontal, vertical, or diagonal. In addition, due to the influence of lighting and material variations, images of the same defect category may also have different grayscale levels.

Data Source: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html

Competition Official Link: https://aistudio.baidu.com/competition/detail/808/0/introduction

### Defect Categories

| ID | Category Name | Description |
|----|--------------|-------------|
| 0 | crazing | Surface cracks |
| 1 | inclusion | Surface inclusions |
| 2 | pitted_surface | Pitted surface |
| 3 | scratches | Surface scratches |
| 4 | patches | Surface patches |
| 5 | rolled-in_scale | Rolled-in scale |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model

```bash
# Basic training (200 epochs)
python train_advanced-33.48.py --device 0 --epochs 200

# Enhanced training (200 epochs + fine-tuned augmentation)
python train_advanced-33.48.py --device 0 --epochs 200 --enhanced

# Large model training (yolo12s)
python train_advanced-33.48.py --device 0 --epochs 200 --model_size s
```

### 3. Model Prediction

```bash
# Single model prediction + TTA
python train_advanced-33.48.py --predict_only --model ./runs_advanced/train_n_ep200/weights/best.pt --tta

# Multi-model ensemble prediction (WBF fusion)
python train_advanced-33.48.py --ensemble \
    --model1 ./runs_advanced/train_n_ep200/weights/best.pt \
    --model2 ./runs_advanced/train_s_ep200/weights/best.pt
```

### 4. Launch Web Interface

```bash
python app.py
```

Visit http://127.0.0.1:7860

## 📁 Core Programs

### Training Program

| File | Description |
|------|-------------|
| `train_advanced-33.48.py` | Main training script (Baidu competition solution, score 33.48) |

### Detection & Application Programs

| File | Description |
|------|-------------|
| `app.py` | Gradio Web interface (5 functional modules) |
| `detection_engine.py` | Unified detection engine (YOLO + VLM) |
| `generate_submission.py` | Generate competition submission file |
| `db_manager.py` | Database management |
| `data_export.py` | Data export |

## 🔬 Core Training Strategy

### Training from Scratch (No Pre-trained Model)

The core training program in this project does **NOT** download a pre-trained YOLO model and then fine-tune it. Instead, it trains **from scratch** (`pretrained=False`) based on a custom YOLOv12 configuration file (`yolov12.yaml`).

```python
# Create model from yaml configuration, train from scratch
model = YOLO('yolov12.yaml')  # pretrained=False
```

**Why choose training from scratch?**

- **Domain Adaptability**: Steel defect detection differs significantly from general datasets like COCO in target features. Pre-trained weights may introduce irrelevant prior knowledge.
- **Avoid Negative Transfer**: COCO pre-trained models learn features of everyday objects. Direct fine-tuning may cause the model to be insensitive to industrial defect features.
- **Complete Domain Feature Learning**: Training from scratch allows the model to focus entirely on learning steel surface defect features, which is more suitable for industrial scenarios.

### YOLO Data Augmentation Principles

Data augmentation is a key method to improve model generalization. This project adopts an **online augmentation** strategy, where data is transformed in real-time during training. Each epoch produces different augmentation effects, equivalent to showing the model "different versions" of the same image.

#### Basic Augmentation (5 Parameters)

| Parameter | Value | Principle |
|-----------|-------|-----------|
| `mosaic` | 1.0 | Randomly stitches 4 training images into 1. **Effect**: Increases the frequency of small objects, improving the model's ability to detect small defects; enhances understanding of complex backgrounds |
| `mixup` | 0.1 | Mixes two images at random ratios. **Effect**: Increases sample diversity, prevents overfitting; helps the model learn smoother decision boundaries |
| `copy_paste` | 0.1 | Copies defect regions from one image and pastes them onto another. **Effect**: Increases sample count for rare defect categories, alleviates class imbalance |
| `scale` | 0.5 | Randomly scales image size (0.5~1.5x). **Effect**: Simulates defects captured at different distances, improves model's adaptability to multi-scale defects |
| `fliplr` | 0.5 | 50% probability of horizontal flip. **Effect**: Defects have no fixed orientation horizontally; flipping effectively expands the dataset |

#### Fine-tuned Augmentation (3 Key Parameters)

| Parameter | Value | Principle |
|-----------|-------|-----------|
| `flipud` | 0.5 | 50% probability of vertical flip. **Effect**: Defects have no fixed orientation vertically either; combined with horizontal flip for comprehensive data expansion |
| `degrees` | 12.0 | Random rotation -12°~+12°. **Effect**: Simulates camera angle changes, strengthens model's rotation invariance for defects |
| `hsv_s` | 0.5 | Saturation augmentation (±50%). **Effect**: Steel surfaces are greatly affected by lighting; enhancing saturation variation improves model robustness under different lighting conditions |

#### Overall Effect of Data Augmentation

```
Original Image ──┬── Mosaic ──→ 4 images combined into 1, increases small targets
                 ├── MixUp ──→ Two images mixed, smooths decision boundaries
                 ├── Copy-Paste ──→ Copies defects, balances classes
                 ├── Scale ──→ Random scaling, multi-scale adaptation
                 ├── Flip ──→ Flipping, orientation invariance
                 ├── Rotate ──→ Rotation, angle invariance
                 └── HSV ──→ Color transformation, lighting robustness
```

Through the combination of these augmentation strategies, the model can see sample variations far exceeding the original dataset during training, significantly improving generalization ability and detection accuracy.

## 🎯 Optimization Strategies

1. **Increased Training Epochs**: epochs increased from 100 to 200
2. **Data Augmentation**: 5 basic augmentation parameters + 3 key fine-tuning parameters (see principles above)
3. **Model Ensemble**: Support yolo12n + yolo12s multi-model WBF fusion
4. **Dynamic Confidence Tuning**: Automatically search for optimal confidence threshold
5. **TTA Inference Enhancement**: Test Time Augmentation to improve inference accuracy

## 🏭 National Standards Integration & Industrial Deployment Capability

This system is not just an AI detection model, but a **quality inspection solution that complies with national standards and is ready for industrial deployment**.

### Integrated National Standards

| Standard No. | Standard Name | Application Scenario |
|-------------|---------------|---------------------|
| GB/T 14977-2008 | General requirements for surface quality of hot-rolled steel plates | Defect grading (A/B/C/D/E), depth/area limits, treatment rules |
| GB/T 3274-2017 | Hot-rolled plates and strips of carbon structural steels and high strength low alloy structural steels | Surface defect limits, cleaning requirements, delivery ratio with defects |
| GB/T 10561-2023 | Determination of content of non-metallic inclusions in steel | Inclusion classification (A/B/C/D/DS), micro-rating, qualified level |
| GB/T 8923.1-2011 | Preparation of steel substrates before application of paints - Visual assessment of surface cleanliness Part 1 | Rust grade (A/B/C/D), cleaning grade (Sa/St/Pt) |
| GB/T 8923.3-2009 | Preparation of steel substrates before application of paints - Visual assessment of surface cleanliness Part 3 | Defect treatment grade, cleaning requirements |

### Defect Quantification Analysis

The system performs **multi-dimensional quantification analysis** on each detected defect, outputting key parameters required for industrial inspection:

| Quantification Metric | Description |
|----------------------|-------------|
| Defect Size | Width × Height (pixels/mm) |
| Defect Area | pixels²/mm² |
| Relative Area | Ratio to image/workpiece area |
| Aspect Ratio | Defect morphology feature |
| Severity Level | Minor/Moderate/Severe (based on national standard thresholds) |
| Over Limit | Automatic determination of compliance with national standards |

### Automatic National Standard Judgment

The system automatically **matches national standards** based on defect type and quantification metrics, outputting:

- **National Standard Judgment**: Based on GB/T 14977-2008 and other standards, automatically determines defect grade (A/B/C/D/E)
- **Technical Treatment Recommendations**: For each defect, provides specific technical solutions such as removal, grinding, welding, or scrapping
- **Process Treatment Recommendations**: Process decision recommendations such as scrap/rework/concession acceptance
- **Reference Standards**: Clearly lists applicable national standard clauses

### Defect Grading and Treatment Reference Table

| Defect Type | National Standard Grade | Judgment Criteria | Treatment Recommendation |
|------------|------------------------|-------------------|-------------------------|
| Crazing | Grade A | Not allowed | Must be completely removed, scrap if depth exceeds tolerance |
| Inclusion | Grade A | Not allowed | Surface inclusion must be removed, scrap if area exceeds limit |
| Rolled-in Scale | Grade A | Not allowed | Must be completely removed, scrap if depth > 0.3mm |
| Pitted Surface | Grade B/C/D/E | Graded by depth and area | Grade B allowed, Grade C grinding, Grade D/E welding or scrap |
| Scratches | Grade B/C | Graded by depth | Grade B no treatment, Grade C grinding, scrap if depth > 0.3mm |
| Patches | Grade B/C | Graded by area | Area ≤ 0.5cm² no treatment, > 0.5cm² grinding |

### Industrial Workflow

```
Daily Inspection → AI Auto Detection → Defect Quantification → National Standard Judgment → Treatment Recommendations
    ↓
Manual Review → Correct/False Positive/Missed Marking → Bad Case Export → Re-annotation → Model Retraining
    ↓
Model Iterative Optimization → Deploy New Version → Continuous Accuracy Improvement
```

## 📊 Web Interface Features

| Tab | Feature | Description |
|-----|---------|-------------|
| Defect Detection | Upload image for detection | Select YOLO/VLM model, display results in real-time |
| Detection Records | View history | Manual review (correct/false positive/missed) |
| Statistical Analysis | Data statistics | Detection count, accuracy, defect category distribution |
| Data Export | Export Bad Cases | Export error samples for retraining |
| Competition Submission | Batch prediction | Batch predict test set to generate submission.csv |

## 🔧 Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| --device | 0 | GPU device ID |
| --epochs | 200 | Training epochs |
| --batch | Auto | Batch size (64 for imgsz=640, 48 for imgsz=800) |
| --imgsz | 640 | Input image size |
| --patience | 80 | Early stopping patience |
| --model_size | n | Model size: n(nano) or s(small) |
| --enhanced | False | Enable fine-tuned augmentation |
| --predict_only | False | Prediction only mode |
| --model | - | Single model path |
| --conf | 0.25 | Confidence threshold |
| --iou | 0.45 | NMS IoU threshold |
| --tta | False | Enable TTA inference |
| --skip_tune | False | Skip confidence tuning |
| --ensemble | False | Multi-model ensemble mode |
| --model1 | - | First model path |
| --model2 | - | Second model path |

## 🖥️ Environment Requirements

- Python >= 3.8
- GPU: Recommended 32GB VRAM or above (RTX 5090 32GB)
- CUDA >= 12.0

## 📂 Project Structure

```
CASE-AI-Quality-Inspection-Enhanced/
├── train_advanced-33.48.py      # Main training script
├── app.py                       # Web application
├── detection_engine.py          # Detection engine
├── generate_submission.py       # Generate submission file
├── db_manager.py                # Database management
├── data_export.py               # Data export
├── requirements.txt             # Dependencies
├── .gitignore                   # Git ignore file
└── README_EN.md                 # Project documentation (English)
```

## 📝 License

MIT License
