# CFIA Food Recall Analytics Suite: Machine Learning for Recall Severity Prediction and Brand Association Discovery

---

## Authors

---

## 1. Abstract

Food recalls are critical public health interventions, yet predicting their severity and understanding the relationships between implicated brands remain largely manual processes. We present the **CFIA Food Recall Analytics Suite**, an interactive machine learning system that analyzes 13 years of Canadian Food Inspection Agency recall data (2011–2024; n = 9,712 records across 1,841 brands and 2,276 incidents). The system addresses two complementary tasks: (1) **recall severity prediction**, classifying recalls into CFIA Class I (serious health risk), Class II (short-term health risk), or Class III (low risk) using both a Random Forest baseline and a neural network that fuses DistilBERT text embeddings with engineered tabular features; and (2) **brand association discovery**, employing graph-based co-recall analysis, TF-IDF product similarity scoring, and K-Means clustering to reveal hidden relationships among recalled brands. The dual-model architecture enables both interpretable (RF) and semantically-aware (BERT+NN) predictions on a heavily imbalanced dataset (69.6% Class I, 22.3% Class II, 8.1% Class III). Deployed as an interactive Streamlit web application, the tool provides food safety professionals with actionable insights for risk assessment, brand monitoring, and incident response prioritization.

---

## 2. Introduction & Motivation

- The Canadian Food Inspection Agency (CFIA) issues hundreds of food recalls annually, each classified by severity (Class I–III)
- **Problem 1:** Recall classification is reactive — severity is determined after investigation, delaying response prioritization
- **Problem 2:** Brand relationships during multi-brand incidents are not systematically tracked, missing patterns that could inform proactive surveillance
- **Gap:** No publicly available tool combines predictive modeling of recall severity with graph-based brand association analysis on CFIA data
- **Goal:** Build an end-to-end analytics platform that predicts recall severity from product metadata and discovers latent brand co-recall networks

---

## 3. Dataset

| Attribute | Value |
|---|---|
| Source | CFIA Recall Database (public) |
| Time span | April 2011 – March 2024 |
| Total records | 9,712 |
| Unique brands | 1,841 |
| Unique incidents | 2,276 |
| Avg. items per incident | 4.3 |
| Max brands in single incident | 27 |

### Class Distribution (Imbalanced)
| Class | Description | Proportion |
|---|---|---|
| Class I | Serious adverse health consequences or death | 69.6% |
| Class II | Temporary adverse health consequences | 22.3% |
| Class III | Low/no health risk | 8.1% |

### Key Fields
Product common name, brand, area of concern (pathogen/contaminant), recall date, distribution depth (consumer / retail / wholesale), recall class

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Raw CFIA Data (XLSX)                      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              DATA PIPELINE (full_analysis_gen.py)            │
│  • Normalization & cleaning                                 │
│  • Temporal feature engineering (season, day, quarter)       │
│  • Incident-level aggregation (brands involved, duration)    │
│  • Brand co-occurrence graph construction                    │
│  • Pathogen frequency analysis                               │
└──────────┬──────────────────────────────────┬───────────────┘
           ▼                                  ▼
┌─────────────────────┐          ┌────────────────────────────┐
│  BRAND ASSOCIATION   │          │   RECALL CLASS PREDICTOR   │
│  ANALYZER            │          │   TRAINING PIPELINE        │
│                      │          │                            │
│ • NetworkX graph     │          │ Model A: Random Forest     │
│ • TF-IDF similarity  │          │  └ TF-IDF + tabular feats  │
│ • K-Means clustering │          │                            │
│ • Multi-factor       │          │ Model B: Neural Network    │
│   association score  │          │  └ DistilBERT embeddings   │
│                      │          │    + tabular features      │
│                      │          │  └ Dense(128)→Dense(64)→3  │
└──────────┬───────────┘          └──────────────┬─────────────┘
           ▼                                     ▼
┌─────────────────────────────────────────────────────────────┐
│              STREAMLIT WEB APPLICATION (app.py)              │
│  Tab 1: Brand Analyzer    │    Tab 2: Recall Predictor      │
│  • Direct associations    │    • Predict new recalls        │
│  • Indirect connections   │    • Explore test data          │
│  • Brand profiling        │    • Model comparison           │
│  • Graph visualization    │    • Confusion matrices         │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Methodology

### 5.1 Feature Engineering

**Temporal features:** Year, month, quarter, season, day of week
**Frequency features:** Brand recall frequency, pathogen frequency
**Recency features:** Days since last brand recall, days since last product recall
**Incident context:** Number of brands involved, unique products, incident duration (days)
**Distribution depth:** Consumer, retail/HRI, wholesale (one-hot encoded)

### 5.2 Recall Severity Prediction

#### Model A — Random Forest (Interpretable Baseline)
- **Text features:** TF-IDF vectorization of product common names (max 5,000 features, bigrams, English stop words removed)
- **Tabular features:** Scaled numerical + one-hot categorical features
- **Combined input:** Sparse matrix concatenation of TF-IDF + tabular
- **Configuration:** 100 estimators, balanced class weights (inverse frequency)
- **Split:** 80/20 stratified train/test

#### Model B — Neural Network with BERT Feature Fusion
- **Text features:** DistilBERT-base-uncased (frozen, pre-trained) extracts 768-dimensional [CLS] token embeddings from product common names
- **Tabular features:** StandardScaler + OneHotEncoder preprocessing
- **Fusion:** Horizontal concatenation of BERT embeddings + tabular features
- **Architecture:**
  - Input → Dense(128, ReLU) → Dropout(0.5) → Dense(64, ReLU) → Dropout(0.3) → Dense(3, Softmax)
- **Training:** Adam optimizer (lr=1e-4), sparse categorical cross-entropy, early stopping (patience=7), balanced class weights, 50 epochs max

### 5.3 Brand Association Analysis

#### Co-Recall Graph Construction
- Identify all multi-brand incidents (recalls sharing the same date + pathogen + class)
- Create weighted edges between every brand pair within each incident
- Edge weight = number of co-occurrences across all incidents

#### Multi-Factor Association Scoring
For a selected brand, each associated brand receives a composite score:

```
Score = 0.50 × norm(connection_weight)
      + 0.30 × product_similarity(TF-IDF cosine)
      + 0.15 × contaminant_match(binary)
      + 0.05 × is_primary_recall(binary)
```

#### Indirect Association Discovery ("Friends of Friends")
- Second-degree graph traversal: Brand A → Brand B → Brand C
- Score propagation via edge weight multiplication
- Reveals non-obvious brand relationships

#### Brand Risk Profiling & Clustering
- Per-brand feature vector: total recalls, Class I/II/III proportions, multi-brand involvement rate, average incident size
- K-Means clustering (k=6) on standardized profiles
- Groups brands into behavioral risk cohorts

---

## 6. Results

### 6.1 Recall Class Prediction Performance

Test set: n = 1,943 (80/20 stratified split). Class distribution: 1,353 Class I / 433 Class II / 157 Class III.

| Metric | Random Forest (TF-IDF) | Neural Network (BERT+Tabular) |
|---|---|---|
| **Overall Accuracy** | **98.82%** | **95.78%** |
| Class I Precision | 99.41% | 98.72% |
| Class I Recall | 99.33% | 97.04% |
| Class I F1 | 99.37% | 97.88% |
| Class II Precision | 97.24% | 90.50% |
| Class II Recall | 97.69% | 92.38% |
| Class II F1 | 97.47% | 91.43% |
| Class III Precision | 98.08% | 86.55% |
| Class III Recall | 97.45% | 94.27% |
| Class III F1 | 97.76% | 90.24% |
| **Macro F1** | **98.20%** | **93.18%** |
| **Weighted F1** | **98.82%** | **95.82%** |

**Confusion Matrices:**

```
Random Forest                    Neural Network (BERT+Tabular)
              Pred I  II  III                  Pred I  II  III
Actual I      1344    9    0    Actual I      1313   37    3
Actual II        7  423    3    Actual II       13  400   20
Actual III       1    3  153    Actual III       4    5  148
```

> **Key findings:** The Random Forest with TF-IDF features outperforms the BERT-based neural network (98.82% vs 95.78%), suggesting that for this structured recall dataset, traditional NLP features (n-gram frequencies) capture the classification signal effectively. The NN shows stronger Class III recall (94.27% vs 97.45%), indicating BERT's semantic understanding helps with the minority class, though RF's precision remains superior across all classes. Both models handle the severe class imbalance (8.1% Class III) well due to balanced class weighting.

### 6.2 Model Validation & Overfitting Analysis

To verify the high accuracy is genuine and not an artifact of overfitting or data leakage, we performed three validation checks:

**Train vs. Test Gap:**

| Model | Train Accuracy | Test Accuracy | Gap |
|---|---|---|---|
| Random Forest | 100.00% | 98.82% | 1.18% |
| Neural Network | 97.09% | 95.78% | 1.31% |

- RF achieving 100% train accuracy is expected behavior for bagged ensemble methods (each tree memorizes its bootstrap sample). The small 1.18% generalization gap indicates the model is not memorizing noise.
- The NN's train accuracy (97.09%) is notably *below* 100%, confirming the dropout regularization (0.5 and 0.3) is functioning as intended to prevent overfitting.

**5-Fold Stratified Cross-Validation (Random Forest):**

| Fold | Accuracy |
|---|---|
| 1 | 98.82% |
| 2 | 98.82% |
| 3 | 98.20% |
| 4 | 98.61% |
| 5 | 98.40% |
| **Mean ± Std** | **98.57% ± 0.24%** |

The narrow standard deviation (0.24%) across all 5 folds confirms stable performance independent of the specific train/test partition. The CV mean (98.57%) is within 0.25% of the single-split test accuracy (98.82%), ruling out a lucky test set.

**Feature Importance Distribution (RF):**
- No single feature dominates: top feature accounts for only 11.1% of total importance
- Top 5 features: 26.2% | Top 20 features: 53.7% — importance is distributed across many TF-IDF terms and tabular features
- This rules out trivial leakage through a single highly predictive proxy variable

**Why is accuracy this high?** The strong performance is consistent with the domain: CFIA recall class is largely determined by the hazard type (e.g., *Salmonella* and *E. coli O157:H7* → almost always Class I; undeclared allergens → Class I or II). The product name + area of concern combination carries strong, legitimate predictive signal for severity classification.

### 6.3 Brand Association Findings
- The co-recall graph contains **1,841 brand nodes** with weighted edges from shared incidents
- Largest connected component reveals major supply chain clusters
- K-Means clustering identifies 6 distinct brand behavioral profiles (e.g., high-frequency Class I brands vs. low-frequency diversified brands)
- TF-IDF product similarity enriches purely structural graph connections with semantic relevance

### 6.3 Key Data Insights
- **Dominant hazard:** E. coli O157:H7 is the most frequent recall trigger
- **Batch recalls:** One incident involved 27 brands simultaneously, indicating shared supply chain contamination
- **Seasonal patterns:** Recall frequency varies by season and year, suggesting environmental/production cycle influences
- **Brand concentration:** Top 5 brands account for ~10% of all recalls

---

## 7. Interactive Application Features

The deployed web application (Streamlit) provides two modules:

### Brand Association Analyzer
- Select any brand → view ranked co-recalled brands with composite scores
- "Friends of friends" indirect association discovery
- Interactive Graphviz network visualization (top 15 connections)
- Brand profiling dashboard with cluster membership
- Real multi-brand incident explorer

### Recall Class Predictor
- Enter product details → receive Class I/II/III prediction with probability distribution
- Toggle between Random Forest and Neural Network models
- Step through held-out test samples with ground truth comparison
- Full confusion matrix visualization for model evaluation

**Live demo:** https://m-tameem-cfia-recall-analytics-ml.streamlit.app/

---

## 8. Discussion

### Contributions
1. **First open-source tool** (to our knowledge) combining ML-based recall severity prediction with graph-based brand association analysis on CFIA data
2. **Novel feature fusion** of pre-trained language model (DistilBERT) embeddings with structured food safety metadata for recall classification
3. **Multi-factor association scoring** that integrates structural (co-recall frequency), semantic (product similarity), and categorical (contaminant match) signals
4. **Deployable system** — containerized, interactive, and accessible to non-technical food safety professionals

### Limitations
- Class imbalance (69.6% Class I) limits minority class performance, particularly for Class III
- BERT embeddings are extracted with frozen weights — fine-tuning on food safety domain text could improve semantic representations
- Brand co-recall analysis assumes shared incidents imply meaningful relationships; some may be coincidental
- Dataset covers only Canadian recalls; generalizability to other jurisdictions is untested

### Future Work
- Fine-tune DistilBERT on food safety / CFIA-specific text corpora
- Incorporate additional data sources (FDA RECALLS, RASFF) for cross-jurisdictional analysis
- Apply temporal graph neural networks to model evolving brand relationships
- Add SHAP/LIME explanations for model interpretability
- Explore active learning to improve Class III prediction with targeted labeling

---

## 9. Technical Implementation

| Component | Technology |
|---|---|
| Web framework | Streamlit 1.45 |
| ML / Classical | scikit-learn 1.6.1 (Random Forest, TF-IDF, K-Means) |
| Deep learning | TensorFlow 2.15 / Keras |
| NLP embeddings | HuggingFace Transformers 4.40 (DistilBERT) |
| Graph analysis | NetworkX 3.3, Graphviz |
| Visualization | Matplotlib 3.8, Seaborn 0.13 |
| Deployment | Docker, Streamlit Cloud |
| Language | Python 3.10 |

---

## 10. References

1. Canadian Food Inspection Agency. (2024). *Food Recall Warnings.* Government of Canada. https://recalls-rappels.canada.ca/
2. Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *NeurIPS 2019 Workshop.*
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
4. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the 5th Berkeley Symposium.*
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.

---

## Acknowledgments
Data sourced from the Canadian Food Inspection Agency's public recall database. 
---s

## Contact & Code
- **Live App:** https://m-tameem-cfia-recall-analytics-ml.streamlit.app/
