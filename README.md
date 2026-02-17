# CFIA Food Recall Analytics Suite

A Streamlit web application for analyzing Canadian Food Inspection Agency (CFIA) food recall data from 2011–2024. The suite combines interactive data exploration, brand association analysis, and machine learning-powered recall severity prediction.

## Features

### Brand Association Analyzer
- **Direct associations** — given a brand, find all historically co-recalled brands ranked by a weighted score (connection strength, product similarity, contaminant match)
- **Indirect connections** — "friends of friends" traversal to surface second-degree risk links
- **Brand profiling & clustering** — K-Means clustering of brands by recall behavior (frequency, severity distribution, multi-brand rate)
- **Interactive graph visualization** — Graphviz-rendered network of brand connections
- **Multi-brand recall examples** — real incidents involving multiple simultaneous brands

### Recall Class Predictor
- Predict whether a new recall will be **Class I** (high risk), **Class II** (moderate), or **Class III** (low)
- Two model options:
  - **Random Forest** — TF-IDF text features + tabular engineered features
  - **Neural Network** — DistilBERT embeddings + tabular features (Keras)
- **Test data explorer** — step through held-out examples and inspect per-sample predictions
- **Confusion matrix** — full test-set evaluation for either model

## Dataset

| Metric | Value |
|---|---|
| Total recall records | 9,712 |
| Date range | April 2011 – March 2024 |
| Unique brands | 1,841 |
| Unique incidents | 2,276 |
| Class I recalls | 69.6 % |
| Class II recalls | 22.3 % |
| Class III recalls | 8.1 % |

Source: [CFIA Recalls and Safety Alerts](https://www.canada.ca/en/public-health/services/food-safety.html)

## Tech Stack

| Layer | Libraries |
|---|---|
| Web framework | Streamlit 1.45 |
| Data | Pandas, NumPy, OpenPyXL |
| ML / DL | scikit-learn, TensorFlow 2.15, Transformers (DistilBERT) |
| Graphs | NetworkX, Graphviz |
| Visualization | Matplotlib, Seaborn |

## Repository Layout

```
cfia/
├── app.py                          # Main Streamlit app (all modules unified)
├── full_analysis_gen.py            # Batch data analysis pipeline
├── graphtime.py                    # Standalone brand association analyzer
├── newfrontend.py                  # Standalone recall class predictor UI
├── training_recall_class_predictor.py  # ML training script
├── cfiadata.XLSX                   # Raw CFIA dataset
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies (graphviz)
├── Dockerfile                      # Container definition
├── docker-compose.yml              # One-command local deployment
├── Makefile                        # Convenience commands
├── nn_model_recall_class.h5        # Trained neural network weights
├── nn_model_recall_class.keras     # Trained neural network (Keras format)
├── rf_model_recall_class.pkl       # Trained Random Forest model
├── preprocessed_data_recall_class/ # Preprocessed train/test splits
└── cfia_analysis_output_v3/        # Analysis CSVs used by the app
```

## Quick Start

### Option 1 — Docker (recommended, one command)

```bash
docker compose up
```

Then open [http://localhost:8501](http://localhost:8501).

### Option 2 — Streamlit Community Cloud (free hosting)

1. Fork this repository on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**
3. Select your fork, set branch to `main` and entrypoint to `app.py`
4. Click **Deploy** — Streamlit handles the rest automatically

### Option 3 — Local Python

```bash
# 1. Clone
git clone https://github.com/M-Tameem/cfia.git
cd cfia

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install system dependency (macOS/Linux)
brew install graphviz           # macOS
# sudo apt-get install graphviz  # Ubuntu/Debian

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Run
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Makefile Commands

```bash
make run        # Run the app locally
make docker-up  # Start via Docker Compose
make docker-down # Stop Docker containers
make install    # Install Python dependencies
make clean      # Remove __pycache__ and .pyc files
```

## Retrain the Models

```bash
python training_recall_class_predictor.py
```

This regenerates all files in `preprocessed_data_recall_class/` and overwrites the saved model files. Requires the raw dataset `cfiadata.XLSX` and a working TensorFlow + Transformers environment.

## Regenerate Analysis Outputs

```bash
python full_analysis_gen.py
```

Rewrites everything in `cfia_analysis_output_v3/` from the raw XLSX.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TF_USE_LEGACY_KERAS` | `1` | Required for TF 2.15 + Keras compatibility |
| `STREAMLIT_SERVER_PORT` | `8501` | Port the app listens on |

These are set automatically in the Docker and Makefile configurations.

## License

This project is for educational and research purposes. CFIA recall data is publicly available under the [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada).
