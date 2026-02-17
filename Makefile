.PHONY: run install clean docker-up docker-down docker-build retrain regenerate

# Run the app locally
run:
	TF_USE_LEGACY_KERAS=1 streamlit run app.py

# Install Python dependencies
install:
	pip install -r requirements.txt

# Build the Docker image
docker-build:
	docker compose build

# Start the app via Docker Compose
docker-up:
	docker compose up -d
	@echo "App running at http://localhost:8501"

# Stop Docker containers
docker-down:
	docker compose down

# Retrain all ML models (requires raw data)
retrain:
	TF_USE_LEGACY_KERAS=1 python scripts/training_recall_class_predictor.py

# Regenerate analysis output CSVs from raw data
regenerate:
	python scripts/full_analysis_gen.py data/cfiadata.XLSX output/

# Remove Python cache files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	echo "Clean complete"
