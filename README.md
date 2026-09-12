# Seizure Detection using EEG Signals with Explainable AI (XAI)

Binary classification of EEG signals (Normal vs. Seizure) using both classical Machine Learning and Deep Learning models, evaluated under realistic Gaussian noise conditions to simulate real-world EEG measurement variability.

---

## 📋 Project Overview

This project uses the **BEED (Bangalore EEG Epilepsy Dataset)** to build and compare multiple classification models for seizure detection. A key focus of this work is **noise robustness testing** — rather than reporting artificially perfect scores on clean data, Gaussian noise is deliberately injected into the normalized feature space to simulate real-world EEG measurement variability (electrode movement, muscle artifacts, equipment noise, etc.).

The project includes:
- **Classical ML pipeline** (`noise_robustness_ml.py`) — 5 traditional ML classifiers
- **Deep Learning pipeline** (`noise_robustness_dl.py`) — 6 neural network architectures
- **Explainability module** (`beed_xai.py`) — XAI techniques applied to interpret model predictions

---

## 🧠 Dataset

**BEED — Bangalore EEG Epilepsy Dataset**

- **Total samples**: 8,000
- **EEG channels (features)**: 16 (`X1` to `X16`)
- **Original labels**: 4-class (`0`: Normal, `1`: Pre-Seizure, `2`: Seizure, `3`: Post-Seizure)
- **Reframed task**: Binary classification
  - `0` = Normal
  - `1` = Seizure (Pre-Seizure + Seizure + Post-Seizure combined)

The dataset is imbalanced, with Seizure-related samples outnumbering Normal samples roughly 3:1 in the binary framing. Class weighting is applied throughout to correct for this.

---

## 🎯 Why Gaussian Noise Injection?

Clean, curated EEG datasets like BEED tend to produce unrealistically perfect model scores (often 99–100% accuracy), because the data is collected under controlled conditions without the artifacts present in real clinical EEG recordings.

To address this, **Gaussian noise (σ = 0.5) is injected into the z-scored (standardized) features**, independently for train, validation, and test splits.

### Design details:

| Aspect | Approach | Reasoning |
|---|---|---|
| **When noise is added** | After `StandardScaler` normalization | Keeps noise scale consistent (unitless, in std-dev terms) across all 16 channels regardless of their raw value ranges |
| **Noise seeds** | Independent seeds per split (train=100, val=150, test=200) | Prevents the model from learning the noise pattern itself (data leakage); keeps experiments reproducible |
| **σ = 0.5 chosen empirically** | Tested σ = 0.05–1.0+ | σ < 0.25 → still >99.5% accuracy (too clean); σ = 0.5 → realistic accuracy spread (95–99%) without AUC saturating at 100%; σ ≥ 1.0 → accuracy drops below 95% (too degraded) |

---

## ⚙️ Methodology

### Preprocessing (both pipelines)
1. Load BEED dataset from CSV
2. Convert 4-class labels to binary (Normal vs. Seizure)
3. Stratified train/test split (80/20) — preserves class ratio in both sets
4. `StandardScaler` fit **only on training data**, then applied to test/validation data (prevents data leakage)
5. Independent Gaussian noise injection per split
6. Class weight computation (`sklearn.utils.class_weight.compute_class_weight`) to handle class imbalance

---

## 🤖 Machine Learning Models (`noise_robustness_ml.py`)

| Model | Key Configuration |
|---|---|
| Logistic Regression | `class_weight='balanced'`, `max_iter=1000` |
| Decision Tree | `max_depth=10`, `class_weight='balanced'` |
| Random Forest | `n_estimators=200`, `max_depth=15`, `class_weight='balanced'` |
| K-Nearest Neighbors | `n_neighbors=7`, `metric='euclidean'` |
| SVM (RBF kernel) | `class_weight='balanced'`, `probability=True` |

**Evaluation**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, and 5-fold Stratified Cross-Validation. Confusion matrices, ROC curves, and feature importance (for Random Forest and Decision Tree) are generated and saved.

### 📊 ML Results (Gaussian Noise σ = 0.5)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | CV Mean | CV Std |
|---|---|---|---|---|---|---|---|
| **SVM** | **98.69%** | 98.6% | 99.67% | **99.13%** | **99.6%** | 98.75% | 0.2% |
| Random Forest | 97.94% | 97.63% | 99.67% | 98.64% | 98.83% | 97.94% | 0.38% |
| KNN | 96.69% | 95.77% | **100.0%** | 97.84% | 96.7% | 96.39% | 0.41% |
| Decision Tree | 95.31% | 95.92% | 97.92% | 96.91% | 91.92% | 94.55% | 0.46% |
| Logistic Regression | 66.38% | 83.17% | 69.17% | 75.52% | 63.71% | 64.06% | 0.61% |

**Key observations:**
- **SVM** achieves the best overall performance across nearly all metrics, benefiting from its RBF kernel's ability to model non-linear decision boundaries.
- **Logistic Regression significantly underperforms (66.38% accuracy)** — as a purely linear model, it cannot capture the non-linear relationships between EEG channels, especially after noise injection increases feature complexity.
- **KNN achieves perfect recall (100%)**, meaning it catches every seizure case in the test set with zero false negatives — clinically valuable despite a lower precision than SVM/RF, since missing a seizure (false negative) is more dangerous than a false alarm.

---

## 🧬 Deep Learning Models (`noise_robustness_dl.py`)

| Model | Architecture Summary |
|---|---|
| **MLP** | Fully-connected network (256→128→64→1), BatchNorm + Dropout |
| **1D-CNN** | 3 Conv1D layers (128→256→128 filters), Global Average Pooling |
| **BiLSTM** | 2 stacked Bidirectional LSTM layers (128→64 units) |
| **CNN-BiLSTM Hybrid** | Conv1D feature extraction → Bidirectional LSTM for temporal dependencies |
| **BiGRU** | 2 stacked Bidirectional GRU layers (128→64 units) |
| **Transformer** | Multi-head self-attention (2 blocks, 4 heads), residual connections + LayerNorm |

**Training configuration:**
- Optimizer: Adam (`lr=0.001`, gradient clipping `clipnorm=1.0`)
- Loss: Binary Cross-Entropy
- Early stopping on `val_auc` (patience=10, restores best weights)
- Learning rate reduction on plateau (`ReduceLROnPlateau`)
- Class weights applied during training
- Max epochs: 100, Batch size: 64

### 📊 DL Results (Gaussian Noise σ = 0.5)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| **BiGRU** | 97.88% | 99.66% | 97.5% | 98.57% | **99.81%** |
| **MLP** | **98.94%** | 98.92% | **99.67%** | **99.29%** | 99.8% |
| BiLSTM | 98.44% | 98.92% | 99.0% | 98.96% | 99.76% |
| CNN-BiLSTM Hybrid | 98.56% | 99.5% | 98.58% | 99.04% | 99.75% |
| CNN_1D | 98.5% | 98.68% | 99.33% | 99.0% | 99.69% |
| Transformer | 95.62% | 98.46% | 95.67% | 97.04% | 99.2% |

**Key observations:**
- Results are sorted by AUC-ROC (BiGRU ranks first at 99.81%), but **MLP actually has higher accuracy (98.94%) and F1-Score (99.29%)** — the two models are effectively tied, and "best" depends on which metric is prioritized for the clinical use case.
- **Transformer underperforms relative to recurrent/convolutional models** (95.62% accuracy, lowest of all DL models) — likely because the 16-feature input is too short a "sequence" and the dataset too small for self-attention to outperform architectures with stronger inductive biases for this data shape.
- All recurrent and convolutional architectures (BiGRU, MLP, BiLSTM, CNN-BiLSTM, CNN_1D) perform very closely (within ~1% of each other across most metrics), suggesting the noise-injected BEED dataset is well-suited to multiple architecture types.

---

## 📈 ML vs DL — Summary Comparison

| Approach | Best Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|---|
| Classical ML | SVM | 98.69% | 99.13% | 99.6% |
| Deep Learning | BiGRU / MLP | 97.88% / 98.94% | 98.57% / 99.29% | 99.81% / 99.8% |

Classical ML (SVM) remains highly competitive with — and in some respects matches — deep learning approaches on this dataset. This is a reasonable outcome given the relatively low-dimensional feature space (16 channels); deep learning architectures typically show their strongest advantage on higher-dimensional or larger-scale data.

---

## 📁 Project Structure

```
seizure-detection-xai/
├── data/
│   └── BEED_Data.csv
├── outputs/
│   ├── beed_ml_results_gaussian_noise/
│   │   ├── cm_*.png                    # Confusion matrices per model
│   │   ├── ml_model_comparison.png
│   │   ├── cv_comparison.png
│   │   ├── roc_all_ml_models.png
│   │   ├── feature_importance.png
│   │   └── ml_results_summary.csv
│   └── beed_dl_results_gaussian_noise/
│       ├── cm_*.png                    # Confusion matrices per model
│       ├── history_*.png               # Training curves per model
│       ├── class_distribution.png
│       ├── model_comparison.png
│       ├── roc_all_models.png
│       ├── results_summary.csv
│       └── *.keras                     # Saved trained models
├── noise_robustness_ml.py              # Classical ML pipeline
├── noise_robustness_dl.py              # Deep Learning pipeline
├── beed_xai.py                         # Explainability (XAI) module
├── README.md
└── .gitignore
```

---

## 🛠️ Setup & Installation

### Requirements
- Python 3.11 (recommended — TensorFlow and related packages may not yet support the latest Python releases)

### Installation

```bash
# Create virtual environment
py -3.11 -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn scipy xgboost shap lime tensorflow keras mne
```

### Running the pipelines

```bash
# Classical ML pipeline
python noise_robustness_ml

# Deep Learning pipeline
python noise_robustness_dl

# XAI / explainability module
python beed_xai
```

Outputs (metrics, plots, trained models) are saved automatically to the `outputs/` directory.

---

## 🔑 Key Takeaways

1. **Noise robustness matters**: Testing under realistic Gaussian noise conditions (rather than clean data) gives a far more credible estimate of real-world model performance for clinical EEG applications.
2. **Non-linear models dominate**: SVM (RBF kernel), Random Forest, and most DL architectures substantially outperform Logistic Regression, indicating the seizure-detection task requires modeling non-linear relationships between EEG channels.
3. **Recall matters clinically**: In seizure detection, false negatives (missed seizures) carry far greater clinical risk than false positives, making recall a critical metric alongside accuracy and F1-score — KNN's perfect recall score is notable in this context.
4. **Classical ML remains competitive**: With only 16 features, well-tuned classical ML (SVM) performs on par with deep learning architectures, showing that deep learning is not automatically superior for lower-dimensional structured data.
5. **Explainability is central to this work**: Given the clinical stakes of seizure detection, model predictions are further interpreted using XAI techniques (see `beed_xai.py`) to build clinician trust and validate that models rely on physiologically meaningful EEG patterns.

---

## 🙋 Author

Sakshi — Final-year CSE (AI) student, NIIT University
