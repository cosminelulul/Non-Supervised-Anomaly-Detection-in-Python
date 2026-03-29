# Anomaly Detection in Python

A benchmark study of anomaly detection methods applied to a synthetic fraud detection dataset. The project explores one statistical approach and two machine learning algorithms, evaluating their effectiveness and limitations in an unsupervised setting.

---

## Project Overview

Anomaly detection is the process of identifying data points that deviate significantly from the rest of the observations. This project covers three types of anomalies:

- **Point anomalies** — a single data point far from the rest (e.g., a €10,000 transaction on a low-limit card)
- **Contextual anomalies** — data that is normal in one context but abnormal in another (e.g., 30°C in winter)
- **Collective anomalies** — a group of events that signal a problem only when they appear together

The goal is to benchmark three detection methods in a realistic, applicable scenario and understand the limitations of unsupervised approaches.

---

## Dataset

The dataset is a **synthetic dataset** generated in Python, designed to simulate bank card fraud. It is modelled after the [Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) from Kaggle, which contains transactions made by European cardholders. The original dataset was anonymized to protect cardholder identity.

The synthetic version was tailored to a realistic class ratio and size:

| Feature | Description |
|---|---|
| `id` | Unique transaction identifier |
| `V1`–`V28` | Anonymized features representing transaction attributes |
| `Amount` | Transaction amount |
| `Class` | **Benchmark target variable** — `1` = Fraudulent, `0` = Legitimate |

**Dataset composition:**
- Total records: ~150,000
- Non-fraudulent transactions: ~144,180 (~96.1%)
- Fraudulent transactions: ~5,820 (~3.9%)

> **Note:** The methods are run in an **unsupervised** fashion — the `Class` variable is only used post-hoc to evaluate performance, not as a training signal.

---

## Methods

### 1. Robust Z-Score (Statistical)

Standard Z-Score is sensitive to outliers because it relies on the mean and standard deviation. The **Robust Z-Score** addresses this by using the **Median Absolute Deviation (MAD)** instead:

$$\text{MAD} = \text{median}(|x_i - \tilde{x}|)$$

$$M_i = \frac{0.6745 \cdot (x_i - \tilde{x})}{\text{MAD}}$$

Scores are aggregated across all features per row, and a threshold is set using the empirical fraud rate. **Limitations:** cannot directly capture multivariate relationships — a transaction may appear normal on each individual feature but anomalous in combination.

### 2. Isolation Forest (Machine Learning)

Based on the principle that anomalies are easier to isolate than normal points. The algorithm builds random decision trees by selecting a random feature and a random split threshold. Points that require fewer partitions to isolate are considered outliers.

- **Inlier:** requires many partitions
- **Outlier:** requires few partitions

Scales well to large, multivariate datasets with a computational cost of approximately O(n log n).

### 3. Local Outlier Factor (Machine Learning)

A density-based method. LOF compares the local density of a point against the density of its neighbours. If a point is in a sparse region while its neighbours are tightly clustered, it is flagged as a local outlier.

- **LOF score ≲ 1** → Inlier
- **LOF score > 1** → Outlier

`n_neighbors` was set to 580 (vs. the default of 20) as the default configuration yielded ~50% worse performance on this dataset. Computational cost is approximately O(n²), making it significantly more expensive than the other methods.

---

## Evaluation

All three methods are evaluated using a confusion matrix, ROC AUC, PR AUC, and the following metrics computed from the confusion matrix:

- **Recall** — proportion of actual frauds correctly flagged
- **Precision** — proportion of flagged transactions that are actually fraudulent
- **F1 Score** — harmonic mean of Precision and Recall
- **Accuracy** — overall correct classification rate

The contamination parameter for all ML models is set to the **empirical fraud rate** derived from the dataset.

### Results Summary

| Method | Recall | Precision | F1 Score |
|---|---|---|---|
| Isolation Forest | Best | Moderate | ~0.788 |
| Robust Z-Score | Good | Moderate | ~0.724 |
| Local Outlier Factor | Poor | Poor | Low |

---

## Conclusions

**Isolation Forest** achieved the best overall performance (F1 ≈ 0.788). It handles large, multivariate datasets efficiently and has a low computational cost relative to its performance.

**Robust Z-Score** performed well for a statistical method (F1 ≈ 0.724) and is computationally cheap, making it a good baseline. However, it cannot model interactions between features — a transaction that looks normal on each feature individually may still be anomalous in combination.

**Local Outlier Factor** delivered the weakest performance at the highest computational cost. Due to the dataset's size and high data density, LOF without a semi-supervised approach is not well-suited for this task.

> **Important caveat:** An accuracy of ~78% is considered a **poor result** in the context of fraud detection. In production fraud systems, even a small false negative rate means missed fraud at scale, and a high false positive rate causes significant friction for legitimate customers. These results highlight the fundamental limitations of unsupervised anomaly detection for fraud — it can serve as an exploratory or supporting tool, but is not production-ready on its own.

---

## Project Structure

```
.
├── AnomalDetect.ipynb          # Main notebook
├── data/
│   └── creditcard_Fraud.csv    # Synthetic fraud dataset
└── scripts/
    ├── MAD_DistrubitionGraph.py # Distribution visualisation for Z-Score
    ├── ISO_Graphs.py            # Isolation Forest visualisations
    └── LOF_Graph.py             # LOF visualisations
```

---

## Dependencies

```
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
```

Install with:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

---

## References

- Original dataset: [Credit Card Fraud Detection Dataset 2023 — Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
- Scikit-learn: [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html), [LocalOutlierFactor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
- Rousseeuw & Croux (1993) — Alternatives to the Median Absolute Deviation
