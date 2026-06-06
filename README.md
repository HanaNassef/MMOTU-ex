# MMOTU XAI Implementation

## Post-Classification Explainability Framework for Ovarian Tumor Diagnosis

This project implements a complete research pipeline for ovarian tumor diagnosis using the MMOTU (OTU_2D) dataset, with a focus on post-classification explainability (XAI).

### Features
- **Backbone Models:** DenseNet121, ResNet50, EfficientNet-B3, MobileNetV3, Swin Transformer, ViT.
- **Training:** Patient-level stratified splits, Mixed Precision (AMP), Gradient Monitoring, and custom loss functions (Weighted CE, Focal Loss).
- **XAI Methods:** Grad-CAM, Grad-CAM++, Score-CAM, Eigen-CAM, Saliency Maps, Integrated Gradients, and DeepSHAP.
- **Evaluation:** Alignment metrics (SC, CC, WCIS, ExBale), Faithfulness (Insertion/Deletion AUC), and Screening analysis (Youden's J).
- **Visualization:** Grid comparison of CAM overlays, ROC curves, training logs, and comprehensive summary reports.

---

### Project Structure

```
ovarian_xai/
├── main.py                     # Single entry point
├── configs/                    # Experiment configurations
├── data/                       # Dataset and splitting logic
├── models/                     # Model factory and heads
├── training/                   # Training loops and losses
├── xai/                        # Explainability methods
├── evaluation/                 # Metrics and statistical tests
├── visualization/              # Plotting and reporting
└── utils/                      # Logging and reproducibility
```

### Installation

```bash
pip install -r requirements.txt
```

### Usage

Run the full pipeline:
```bash
python main.py --config configs/default.yaml
```

Run in debug mode (fast test):
```bash
python main.py --debug
```

Resume from checkpoint:
```bash
python main.py --resume results/checkpoints/exp_001_densenet121_best.pt
```

---

### Results
All outputs are saved to the `results/` directory, including:
- `results/figures/`: Performance and XAI plots.
- `results/logs/`: Detailed training metrics and CSV logs.
- `results/xai/`: Per-image XAI alignment metrics.
- `results/summary_report.txt`: A human-readable summary of the entire experiment.
