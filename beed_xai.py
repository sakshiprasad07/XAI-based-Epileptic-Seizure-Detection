# =============================================================================
# BEED — Bangalore EEG Epilepsy Dataset
# XAI Analysis using SHAP
# Models: MLP (Best DL) + SVM (Best ML)
# Visualization: SHAP Heatmap + Beeswarm + Bar + Class Comparison
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# =============================================================================
# CONFIGURATION — update paths if needed
# =============================================================================

DATA_PATH      = "d:/Sakshi work/RnD/seizure-detection-xai/data/BEED_Data.csv"
MLP_MODEL_PATH = "d:/Sakshi work/RnD/seizure-detection-xai/outputs/beed_dl_results/mlp_model.keras"
OUTPUT_DIR     = "d:/Sakshi work/RnD/seizure-detection-xai/outputs/beed_xai_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED    = 42
TEST_SIZE      = 0.2
N_SHAP_SAMPLES = 500   # keep ≤500 for speed

FEATURE_COLS = [f'X{i}' for i in range(1, 17)]

CHANNEL_LABELS = {
    'X1' : 'Fp1 (L.Frontal)',   'X2' : 'Fp2 (R.Frontal)',
    'X3' : 'F3 (L.Frontal)',    'X4' : 'F4 (R.Frontal)',
    'X5' : 'F7 (L.Fronto-T)',   'X6' : 'F8 (R.Fronto-T)',
    'X7' : 'T3 (L.Temporal)',   'X8' : 'T4 (R.Temporal)',
    'X9' : 'T5 (L.Post-T)',     'X10': 'T6 (R.Post-T)',
    'X11': 'P3 (L.Parietal)',   'X12': 'P4 (R.Parietal)',
    'X13': 'O1 (L.Occipital)',  'X14': 'O2 (R.Occipital)',
    'X15': 'C3 (L.Central)',    'X16': 'C4 (R.Central)'
}
READABLE_COLS = [CHANNEL_LABELS[c] for c in FEATURE_COLS]

print("=" * 60)
print("XAI Analysis — SHAP")
print("Models: MLP (Best DL) + SVM (Best ML)")
print("=" * 60)

# =============================================================================
# STEP 1: LOAD & PREPARE DATA
# =============================================================================

print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
df['y_bin'] = (df['y'] > 0).astype(int)

X = df[FEATURE_COLS].values.astype(np.float32)
y = df['y_bin'].values.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# Stratified subset for SHAP
np.random.seed(RANDOM_SEED)
idx_0    = np.where(y_test == 0)[0][:N_SHAP_SAMPLES // 2]
idx_1    = np.where(y_test == 1)[0][:N_SHAP_SAMPLES // 2]
shap_idx = np.concatenate([idx_0, idx_1])
X_shap   = X_test[shap_idx]
y_shap   = y_test[shap_idx]
X_shap_df = pd.DataFrame(X_shap, columns=READABLE_COLS)

print(f"SHAP samples: {len(X_shap)} "
      f"({(y_shap==0).sum()} Normal, {(y_shap==1).sum()} Seizure)")

# Background for explainers
background = shap.sample(X_train, 100, random_state=RANDOM_SEED)

# =============================================================================
# HELPER: FIX SHAP SHAPE → always returns (n_samples, n_features)
# =============================================================================

def fix_shap_shape(shap_raw, n_samples, n_features):
    """Normalise any SHAP output to shape (n_samples, n_features)."""
    if isinstance(shap_raw, list):
        arr = np.array(shap_raw[1]) if len(shap_raw) > 1 else np.array(shap_raw[0])
    else:
        arr = np.array(shap_raw)

    if arr.ndim == 3:
        if arr.shape[0] == 2:
            arr = arr[1]                   # (2, n_samples, n_features) → take class 1
        elif arr.shape[2] == 2:
            arr = arr[:, :, 1]             # (n_samples, n_features, 2) → take class 1
        else:
            arr = arr[:, :, 0]

    if arr.shape != (n_samples, n_features):
        arr = arr.reshape(n_samples, n_features)

    return arr.astype(np.float32)

# =============================================================================
# PLOT HELPERS
# =============================================================================

def plot_shap_heatmap(shap_values, y_labels, model_name, filename):
    sort_order = np.argsort(y_labels)
    sv_sorted  = shap_values[sort_order].T      # (channels, samples)
    y_sorted   = y_labels[sort_order]

    fig, ax = plt.subplots(figsize=(16, 7))
    vmax = np.percentile(np.abs(sv_sorted), 95)
    im   = ax.imshow(sv_sorted, aspect='auto', cmap='RdBu_r',
                     interpolation='nearest', vmin=-vmax, vmax=vmax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('SHAP Value\n(+ = Seizure, − = Normal)', fontsize=10)

    ax.set_yticks(range(len(READABLE_COLS)))
    ax.set_yticklabels(READABLE_COLS, fontsize=9)
    ax.set_xlabel('Samples (sorted: Normal → Seizure)', fontsize=11)
    ax.set_title(f'SHAP Heatmap — {model_name}\n'
                 'Red = pushes toward Seizure | Blue = toward Normal',
                 fontsize=12, fontweight='bold')

    n_normal = (y_sorted == 0).sum()
    ax.axvline(x=n_normal - 0.5, color='black', linewidth=2, linestyle='--')
    ax.text(n_normal / 2, -1.2, 'Normal', ha='center', fontsize=10,
            color='steelblue', fontweight='bold',
            transform=ax.get_xaxis_transform())
    ax.text(n_normal + (len(y_sorted) - n_normal) / 2, -1.2, 'Seizure',
            ha='center', fontsize=10, color='crimson', fontweight='bold',
            transform=ax.get_xaxis_transform())

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/{filename}")


def plot_shap_beeswarm(shap_values, X_df, model_name, filename):
    shap_exp = shap.Explanation(
        values        = shap_values,
        base_values   = np.zeros(len(shap_values)),
        data          = X_df.values,
        feature_names = X_df.columns.tolist()
    )
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_exp, max_display=16, show=False)
    plt.title(f'SHAP Beeswarm — {model_name}',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/{filename}")


def plot_shap_bar(shap_values, X_df, model_name, filename):
    mean_abs   = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(mean_abs)))
    ax.barh(range(len(mean_abs)), mean_abs[sorted_idx], color=colors)
    ax.set_yticks(range(len(mean_abs)))
    ax.set_yticklabels([X_df.columns[i] for i in sorted_idx], fontsize=9)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
    ax.set_title(f'SHAP Feature Importance — {model_name}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/{filename}")


def plot_shap_class_comparison(shap_values, y_labels, model_name, filename):
    mean_normal  = shap_values[y_labels == 0].mean(axis=0)
    mean_seizure = shap_values[y_labels == 1].mean(axis=0)
    data = np.stack([mean_normal, mean_seizure], axis=1)   # (16, 2)

    fig, ax = plt.subplots(figsize=(6, 8))
    vmax = np.abs(data).max()
    im   = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Normal', 'Seizure'], fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(READABLE_COLS)))
    ax.set_yticklabels(READABLE_COLS, fontsize=9)

    for i in range(len(READABLE_COLS)):
        for j in range(2):
            ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center',
                    fontsize=7,
                    color='white' if abs(data[i, j]) > vmax * 0.5 else 'black')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        'Mean SHAP Value', fontsize=9)
    ax.set_title(f'Avg Channel Contribution\nNormal vs Seizure — {model_name}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/{filename}")


# =============================================================================
# PART A: SVM — BEST ML MODEL
# =============================================================================

print("\n" + "=" * 60)
print("PART A: XAI for SVM (Best ML Model)")
print("=" * 60)

print("Retraining SVM...")
svm_model = SVC(kernel='rbf', class_weight='balanced',
                probability=True, random_state=RANDOM_SEED)
svm_model.fit(X_train, y_train)
print(f"SVM accuracy: {(svm_model.predict(X_test)==y_test).mean()*100:.2f}%")

print("\nComputing SHAP values for SVM (may take 2-3 mins)...")
svm_explainer = shap.KernelExplainer(svm_model.predict_proba, background)
svm_shap_raw  = svm_explainer.shap_values(X_shap, nsamples=100)

svm_shap = fix_shap_shape(svm_shap_raw, len(X_shap), len(FEATURE_COLS))
print(f"SVM SHAP shape (fixed): {svm_shap.shape}")

plot_shap_heatmap(svm_shap, y_shap, 'SVM', 'svm_shap_heatmap.png')
plot_shap_beeswarm(svm_shap, X_shap_df, 'SVM', 'svm_shap_beeswarm.png')
plot_shap_bar(svm_shap, X_shap_df, 'SVM', 'svm_shap_bar.png')
plot_shap_class_comparison(svm_shap, y_shap, 'SVM', 'svm_shap_class_comparison.png')

# =============================================================================
# PART B: MLP — BEST DL MODEL
# =============================================================================

print("\n" + "=" * 60)
print("PART B: XAI for MLP (Best DL Model)")
print("=" * 60)

print(f"Loading MLP from: {MLP_MODEL_PATH}")
mlp_model = keras.models.load_model(MLP_MODEL_PATH)

print("\nComputing SHAP values for MLP (GradientExplainer)...")
mlp_explainer = shap.GradientExplainer(mlp_model, background)
mlp_shap_raw  = mlp_explainer.shap_values(X_shap)

mlp_shap = fix_shap_shape(mlp_shap_raw, len(X_shap), len(FEATURE_COLS))
print(f"MLP SHAP shape (fixed): {mlp_shap.shape}")

plot_shap_heatmap(mlp_shap, y_shap, 'MLP', 'mlp_shap_heatmap.png')
plot_shap_beeswarm(mlp_shap, X_shap_df, 'MLP', 'mlp_shap_beeswarm.png')
plot_shap_bar(mlp_shap, X_shap_df, 'MLP', 'mlp_shap_bar.png')
plot_shap_class_comparison(mlp_shap, y_shap, 'MLP', 'mlp_shap_class_comparison.png')

# =============================================================================
# PART C: SIDE-BY-SIDE COMPARISON — MLP vs SVM
# =============================================================================

print("\n" + "=" * 60)
print("PART C: MLP vs SVM — SHAP Heatmap Comparison")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, shap_vals, title in zip(
    axes,
    [mlp_shap, svm_shap],
    ['MLP (Best DL)', 'SVM (Best ML)']
):
    sort_order = np.argsort(y_shap)
    sv_sorted  = shap_vals[sort_order].T
    y_sorted   = y_shap[sort_order]
    vmax       = np.percentile(np.abs(sv_sorted), 95)

    im = ax.imshow(sv_sorted, aspect='auto', cmap='RdBu_r',
                   interpolation='nearest', vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(READABLE_COLS)))
    ax.set_yticklabels(READABLE_COLS, fontsize=8)
    ax.set_xlabel('Samples (Normal → Seizure)', fontsize=10)
    ax.set_title(f'SHAP Heatmap — {title}', fontsize=11, fontweight='bold')

    n_normal = (y_sorted == 0).sum()
    ax.axvline(x=n_normal - 0.5, color='black', linewidth=2, linestyle='--')
    ax.text(n_normal / 2, -1.5, 'Normal', ha='center', fontsize=9,
            color='steelblue', fontweight='bold',
            transform=ax.get_xaxis_transform())
    ax.text(n_normal + (len(y_sorted) - n_normal) / 2, -1.5, 'Seizure',
            ha='center', fontsize=9, color='crimson', fontweight='bold',
            transform=ax.get_xaxis_transform())

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

plt.suptitle('SHAP Heatmap Comparison — MLP vs SVM\n'
             'Red = pushes toward Seizure | Blue = toward Normal',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/comparison_mlp_vs_svm_heatmap.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/comparison_mlp_vs_svm_heatmap.png")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("XAI ANALYSIS COMPLETE")
print("=" * 60)
print(f"  Models    : MLP (Best DL), SVM (Best ML)")
print(f"  Samples   : {len(X_shap)}")
print(f"  Output    : {OUTPUT_DIR}/")
print(f"\n  Files generated:")
print(f"    svm_shap_heatmap.png")
print(f"    svm_shap_beeswarm.png")
print(f"    svm_shap_bar.png")
print(f"    svm_shap_class_comparison.png")
print(f"    mlp_shap_heatmap.png")
print(f"    mlp_shap_beeswarm.png")
print(f"    mlp_shap_bar.png")
print(f"    mlp_shap_class_comparison.png")
print(f"    comparison_mlp_vs_svm_heatmap.png")
print("=" * 60)