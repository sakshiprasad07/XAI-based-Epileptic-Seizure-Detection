# Seizure Detection using ML/DL with Explainable AI (XAI)

## Overview

This project focuses on detecting epileptic seizures using the **BEED dataset**. Multiple Machine Learning (ML) and Deep Learning (DL) models were trained and evaluated on the dataset, and the best-performing model from each category was selected.

To make the predictions interpretable and clinically useful, **SHAP (SHapley Additive exPlanations)** was applied to the best models. This adds transparency to the predictions — allowing medical practitioners to understand *why* a model classified a given instance as a seizure or non-seizure event, rather than relying on a black-box decision. The goal is to bridge the gap between high-performing models and real-world clinical trust.

## Dataset

- **Name:** BEED (Bonn EEG Epilepsy Dataset)
- **File:** `data/BEED_Data.csv`

*(Add a short note here on what the dataset contains — e.g., number of samples, features, class labels — if you'd like this section more detailed.)*

## Repository Structure

```
Seizure-Detection-XAI/
├── data/
│   └── BEED_Data.csv          # Dataset used for training/testing
├── new.py                     # (describe purpose)
├── beed_xai.py                 # Applies SHAP explainability on best ML/DL models
├── noise_robustness_dl         # Evaluates DL model robustness under noisy conditions
├── noise_robustness_ml         # Evaluates ML model robustness under noisy conditions
├── outputs/
│   ├── beed_dl_results_gaussian_noise/   # DL model results under Gaussian noise
│   ├── beed_ml_results_gaussian_noise/   # ML model results under Gaussian noise
│   └── beed_xai_results/                 # SHAP explainability outputs
└── README.md
```

## How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(Create a `requirements.txt` if you haven't already — let me know your libraries and I can generate one.)*

2. **Run model training / evaluation scripts**
   ```bash
   python new.py
   ```

3. **Test model robustness against noise**
   ```bash
   python noise_robustness_ml.py
   python noise_robustness_dl.py
   ```

4. **Generate SHAP explainability results**
   ```bash
   python beed_xai.py
   ```

> Update the commands above with correct filenames/extensions if they differ from what's shown here.

## Results & Outputs

- **`outputs/beed_ml_results_gaussian_noise/`** — Performance of the best ML model when Gaussian noise is introduced into the input data, used to test robustness.
- **`outputs/beed_dl_results_gaussian_noise/`** — Performance of the best DL model under the same noise conditions, for comparison against the ML model.
- **`outputs/beed_xai_results/`** — SHAP visualizations and explanation outputs showing which features (EEG signal characteristics) most influenced each prediction. These plots can help medical practitioners understand model decisions and build trust in AI-assisted seizure detection.

## Motivation

Black-box predictions are difficult to trust in clinical settings. By applying SHAP to the best-performing models, this project aims to make seizure detection predictions **transparent and interpretable**, so that medical professionals can verify the reasoning behind a diagnosis rather than accepting the model's output blindly.

## Future Work

Want to test it on more datasets and get it tested by healthcare practitioners 
