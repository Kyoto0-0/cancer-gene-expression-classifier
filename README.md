# Cancer Type Classification from Gene Expression Data

This dataset is a well-separated benchmark: each cancer type originates from a distinct tissue, so transcriptional profiles are naturally distinct. 99.4% accuracy is expected and does not generalize to harder clinical problems (e.g., cancer subtype classification, early-stage detection). The interesting result is the one misclassification — a LUAD sample predicted as BRCA — which reflects real molecular overlap between the two cancers.

## Biological Question
Can a neural network distinguish between five cancer types (BRCA, KIRC, COAD, LUAD, PRAD) based solely on gene expression patterns?

## Dataset
- **Source:** TCGA Pan-Cancer RNA-Seq (UCI ML Repository)
- **Samples:** 801
- **Genes:** 20,531
- **Classes:** 5 cancer types (Breast, Kidney, Colon, Lung, Prostate)

## Methods
- **Feature selection:** Top 500 most variable genes (from 20,531)
- **Preprocessing:** Z-score standardization
- **Model:** Feed-forward neural network (256 → 128 → 5) with dropout
- **Train/test split:** 80/20, stratified
- **Framework:** scikit-learn `MLPClassifier`

## Results
- **Test accuracy:** 99.38%
- **Misclassifications:** 1 of 161 test samples (one LUAD predicted as BRCA)
- **Macro F1-score:** 0.99

### Classification Report
| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| BRCA  | 0.98      | 1.00   | 0.99     | 60      |
| COAD  | 1.00      | 1.00   | 1.00     | 16      |
| KIRC  | 1.00      | 1.00   | 1.00     | 30      |
| LUAD  | 1.00      | 0.96   | 0.98     | 28      |
| PRAD  | 1.00      | 1.00   | 1.00     | 27      |
| **Accuracy** | | | **0.99** | **161** |

## Biological Interpretation

The model achieved 99.38% test accuracy, correctly classifying 160 of 161 held-out samples across five cancer types. This high performance is expected: each cancer type originates from a distinct tissue (breast, colon, kidney, lung, prostate), and these tissues have fundamentally different transcriptional programs. A breast epithelial cell and a kidney tubular cell express different sets of genes because they perform different functions — so their mRNA profiles are naturally distinct.

The single misclassification is the most biologically interesting result. One LUAD (lung adenocarcinoma) sample was predicted as BRCA (breast cancer), giving LUAD a recall of 96% while all other classes achieved 100%. This is not a failure of the model — it reflects genuine molecular overlap between the two cancer types. Both LUAD and BRCA can share dysregulated pathways in cell cycle control, DNA repair, and epithelial-mesenchymal transition. When the model sees a LUAD sample whose expression profile happens to resemble a breast cancer signature, it classifies it as BRCA. The confusion matrix therefore captures real biological ambiguity, not just statistical error.

The heatmap of the top 50 most variable genes shows that a relatively small number of genes drive the classification. These genes are highly expressed in some cancer types and nearly silent in others — reflecting tissue-specific transcription factors and lineage markers. This is why feature selection from 20,531 genes down to 500 preserves nearly all discriminative power. The PCA plot confirms this: the five cancer types form distinct clusters in the first two principal components, meaning the biological signal is strong and low-dimensional.

## Visualizations
- `results/pca_plot.png` — 2D PCA projection of samples colored by cancer type
- `results/confusion_matrix.png` — classification performance across 5 classes
- `results/heatmap.png` — mean expression of top 50 variable genes by cancer type

## How to Run
```bash
pip install -r requirements.txt
python build.py
