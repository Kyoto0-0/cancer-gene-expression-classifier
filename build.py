"""
Cancer Type Classification from Gene Expression Data
Bio-computational project — TCGA Pan-Cancer dataset
"""
import os
import tarfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, ConfusionMatrixDisplay)

# Try TensorFlow; fall back to sklearn if not installed
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    USE_KERAS = True
    print("✔ Using TensorFlow/Keras")
except ImportError:
    from sklearn.neural_network import MLPClassifier
    USE_KERAS = False
    print("⚠ TensorFlow not found — using scikit-learn MLPClassifier")

np.random.seed(42)
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============================================================
# SECTION 1: DOWNLOAD DATA
# ============================================================
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz")
ARCHIVE = "data/tcga.tar.gz"

def find_file(name, root):
    for dirpath, _, files in os.walk(root):
        if name in files:
            return os.path.join(dirpath, name)
    return None

if find_file("data.csv", "data") is None:
    print("Downloading dataset (~30 MB)...")
    urllib.request.urlretrieve(URL, ARCHIVE)
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall("data")
    print("✔ Download complete")
else:
    print("✔ Data already present")

data_path = find_file("data.csv", "data")
labels_path = find_file("labels.csv", "data")

# ============================================================
# SECTION 2: LOAD DATA
# ============================================================
X = pd.read_csv(data_path, index_col=0)
y = pd.read_csv(labels_path, index_col=0)
print(f"\nExpression matrix: {X.shape[0]} samples × {X.shape[1]} genes")
print(f"Classes: {sorted(y.iloc[:, 0].unique())}")

le = LabelEncoder()
y_enc = le.fit_transform(y.iloc[:, 0].values)

# ============================================================
# SECTION 3: FEATURE SELECTION (Top 500 variable genes)
# ============================================================
gene_var = X.var(axis=0)
top_genes = gene_var.nlargest(500).index
X_top = X[top_genes].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_top)

# ============================================================
# SECTION 4: PCA VISUALIZATION
# ============================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for i, cls in enumerate(le.classes_):
    mask = y_enc == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=cls, alpha=0.7, s=25)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("PCA of TCGA Pan-Cancer Gene Expression")
plt.legend(title="Cancer type")
plt.tight_layout()
plt.savefig("results/pca_plot.png", dpi=150)
plt.close()
print("✔ Saved results/pca_plot.png")

# ============================================================
# SECTION 5: TRAIN / TEST SPLIT
# ============================================================
X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"\nTrain: {X_tr.shape[0]}  |  Test: {X_te.shape[0]}")

# ============================================================
# SECTION 6: MODEL
# ============================================================
if USE_KERAS:
    model = keras.Sequential([
        layers.Input(shape=(X_tr.shape[1],)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(le.classes_), activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(X_tr, y_tr, validation_split=0.2,
                        epochs=60, batch_size=32, verbose=1)
    y_pred = model.predict(X_te, verbose=0).argmax(axis=1)

    # Save training curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history["accuracy"], label="train")
    ax[0].plot(history.history["val_accuracy"], label="val")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(history.history["loss"], label="train")
    ax[1].plot(history.history["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150)
    plt.close()
else:
    model = MLPClassifier(hidden_layer_sizes=(256, 128),
                          max_iter=500, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

# ============================================================
# SECTION 7: EVALUATION
# ============================================================
acc = accuracy_score(y_te, y_pred)
print(f"\n🎯 Test accuracy: {acc*100:.2f}%\n")
print(classification_report(y_te, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_te, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(cmap="Blues", colorbar=False)
plt.title(f"Confusion Matrix — Accuracy {acc*100:.1f}%")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.close()
print("✔ Saved results/confusion_matrix.png")

# ============================================================
# SECTION 8: HEATMAP OF TOP 50 GENES
# ============================================================
top50 = gene_var.nlargest(50).index
df_top = X[top50].copy()
df_top["Class"] = le.inverse_transform(y_enc)
mean_expr = df_top.groupby("Class").mean()

plt.figure(figsize=(16, 4))
sns.heatmap(mean_expr, cmap="viridis", cbar_kws={"label": "Mean expression"})
plt.title("Mean Expression of Top 50 Variable Genes by Cancer Type")
plt.xlabel("Gene"); plt.ylabel("Cancer type")
plt.tight_layout()
plt.savefig("results/heatmap.png", dpi=150)
plt.close()
print("✔ Saved results/heatmap.png")

# ============================================================
# SECTION 9: SUMMARY
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Dataset: TCGA Pan-Cancer (UCI)")
print(f"Samples: {X.shape[0]}  |  Genes: {X.shape[1]}")
print(f"Classes: {', '.join(le.classes_)}")
print(f"Features used: top 500 most variable genes")
print(f"Model: {'Keras NN (256-128)' if USE_KERAS else 'sklearn MLP (256-128)'}")
print(f"Test accuracy: {acc*100:.2f}%")
print("="*60)
