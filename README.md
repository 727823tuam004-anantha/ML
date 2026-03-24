# MLOps Project - 727823TUAM004

This repository contains an end-to-end MLOps pipeline for the diabetes dataset.

## Structure

- `data/diabetes_data.csv` - source dataset
- `src/` - scripts (data prep, training, evaluation, MLflow)
- `outputs/` - artifacts (processed, models, metrics, best_model)
- `azure/` - Azure ML component and pipeline YAML definitions
- `report/` - project report placeholder

## Quick start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare data:
   ```bash
   python src/data_prep.py
   ```

3. Train model:
   ```bash
   python src/train_pipeline.py
   ```

4. Evaluate:
   ```bash
   python src/evaluate.py
   ```

5. Train with MLflow (optional):
   ```bash
   python src/train_mlflow.py
   ```
