# MMOTU-ex: Post-Classification Explainability Framework

## Project Overview
MMOTU-ex is a research-oriented machine learning pipeline designed for ovarian tumor diagnosis using the MMOTU (OTU_2D) dataset. Its primary focus is on **Explainable AI (XAI)**, providing a comprehensive framework to train multiple backbone architectures and evaluate their interpretability using various post-classification explanation methods.

The framework supports patient-level stratified splitting, multi-class classification (8 classes), and a rigorous evaluation suite that includes alignment metrics (compared against ground-truth segmentations) and faithfulness metrics (insertion/deletion AUC).

### Key Technologies
- **Core:** Python 3.10+, PyTorch 2.1+, Torchvision 0.16+
- **XAI Libraries:** `grad-cam`, `captum`, `shap`
- **Data & Analytics:** `pandas`, `numpy`, `scikit-learn`, `scipy`, `pyyaml`
- **Visualization:** `matplotlib`, `seaborn`, `opencv-python`

### Architecture
The project follows a modular stage-based pipeline:
1.  **Stage 1: Data Preparation:** Patient-level stratified splitting and dataset indexing.
2.  **Stage 2: Training:** Fine-tuning backbones (DenseNet, ResNet, EfficientNet, MobileNet, Swin, ViT) with custom heads.
3.  **Stage 3: XAI Generation:** Computing heatmaps using CAM-based (Grad-CAM, Score-CAM, etc.) and Gradient-based (Saliency, Integrated Gradients) methods.
4.  **Stage 4: Evaluation:** Calculating SC, CC, WCIS, and ExBale metrics, along with Faithfulness AUC and statistical significance tests.
5.  **Stage 5: Visualization:** Generating training curves, CAM overlays, violin plots, and ROC curves.
6.  **Stage 6: Reporting:** Producing a comprehensive summary report of the experiment.

---

## Building and Running

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Execution Commands
- **Full Pipeline:** Run the entire experiment as defined in the config.
  ```bash
  python main.py --config configs/default.yaml
  ```
- **Debug Mode:** Run a fast test with fewer epochs and subsetted data.
  ```bash
  python main.py --debug
  ```
- **Resume Training:** Continue from a specific model checkpoint.
  ```bash
  python main.py --resume results/checkpoints/<ckpt_name>.pt
  ```
- **Skip Training:** Run XAI and Evaluation using existing checkpoints.
  ```bash
  python main.py --skip_training
  ```
- **Targeted Models:** Specify a subset of models to process.
  ```bash
  python main.py --models "densenet121,resnet50"
  ```

---

## Development Conventions

### Code Structure
- **`main.py`**: The central orchestrator. It uses a `ConfigNamespace` to manage parameters from `configs/*.yaml`.
- **`models/`**: Use `factory.py` to add new architectures. All backbones use `ClassificationHead` from `heads.py`.
- **`xai/`**: XAI methods are categorized into `cam_methods.py`, `gradient_methods.py`, and `shap_methods.py`. Use `XAIRunner` to integrate new methods.
- **`evaluation/`**: Contains metrics for quantifying explanation quality.
    - `alignment_metrics.py`: Spatial overlap metrics (SC, CC, WCIS).
    - `faithfulness.py`: Perturbation-based metrics (Insertion/Deletion AUC).
- **`utils/`**: Handling logging, reproducibility (seeds), and checkpointing.

### Configuration
All hyperparameters and pipeline settings are defined in `configs/default.yaml`. Avoid hardcoding paths or thresholds; instead, add them to the YAML and access them via the `config` object in `main.py`.

### Results and Artifacts
Outputs are structured within the `results/` directory (or as configured):
- `checkpoints/`: Model weights (`.pt`).
- `logs/`: Training CSVs and execution logs.
- `figures/`: All generated plots and visualizations.
- `xai/`: Raw XAI metric dataframes.
- `summary_report.txt`: Final synthesized findings.

### Testing
While there is no dedicated `tests/` directory, the `--debug` flag in `main.py` serves as the primary integration test for verifying the full pipeline on a subset of data.
