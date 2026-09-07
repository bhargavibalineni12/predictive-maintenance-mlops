# Predictive Maintenance MLOps

An end-to-end Machine Learning and MLOps project for **engine condition classification** using sensor measurements.

This project demonstrates how a machine learning model can move from raw data through data validation, model experimentation, model selection, artifact packaging, API serving, automated testing, and Docker containerization.

> **Note:** The available dataset contains independent sensor observations and a binary `Engine Condition` target. Therefore, this project performs **engine condition classification for predictive maintenance**, rather than Remaining Useful Life (RUL) or future time-to-failure prediction.

---

## Project Architecture

```text
Engine Sensor Data
        ↓
Data Ingestion
        ↓
Data Validation
        ↓
Stratified Train/Test Split
        ↓
Baseline Model Training
        ↓
5-Fold Cross Validation
        ↓
MLflow Experiment Tracking
        ↓
Hyperparameter Tuning
        ↓
Model Selection
        ↓
Threshold Selection using OOF Predictions
        ↓
Final Evaluation on Untouched Test Set
        ↓
Model + Metadata Artifacts
        ↓
FastAPI Inference API
        ↓
Automated Tests
        ↓
Docker Container
```

---

## Dataset

The dataset contains **19,535 observations** with six numerical sensor features and one binary target variable.

### Features

- Engine rpm
- Lub oil pressure
- Fuel pressure
- Coolant pressure
- lub oil temp
- Coolant temp

### Target

```text
Engine Condition
```

The target contains binary values:

```text
0
1
```

The project intentionally refers to them as class `0` and class `1` unless their physical meaning is established from authoritative dataset documentation.

---

## Data Validation

Before model training, the data pipeline performs validation checks for:

- Required columns
- Missing values
- Target values
- Numerical feature types
- Duplicate rows

The raw dataset currently passes these validation checks.

---

## Data Splitting

The dataset is divided into:

```text
Training set: 80%
Test set:     20%
```

A stratified split is used to preserve the target distribution.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The final test set is kept untouched during model comparison, hyperparameter tuning, and classification-threshold selection.

---

## Model Experimentation

Four baseline machine learning models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Each model was evaluated using **5-fold cross-validation**.

The following metrics were compared:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Average Precision

---

## MLflow Experiment Tracking

MLflow is used to track machine learning experiments.

For each experiment, the project records:

- Model name
- Model parameters
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Average Precision

This provides a reproducible way to compare model experiments instead of selecting a model based on manually observed results.

---

## Baseline Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6618 | 0.6797 | 0.8770 | 0.7658 | 0.6883 | 0.7833 |
| Decision Tree | 0.5863 | 0.6735 | 0.6673 | 0.6704 | 0.5577 | 0.6593 |
| Random Forest | 0.6532 | 0.6952 | 0.8011 | 0.7444 | 0.6770 | 0.7770 |
| XGBoost | 0.6432 | 0.6911 | 0.7850 | 0.7351 | 0.6627 | 0.7707 |

These results represent the baseline configurations used in this project.

---

## Hyperparameter Tuning

Logistic Regression and Random Forest were selected for further tuning.

`GridSearchCV` with 5-fold cross-validation was used with:

```text
scoring = average_precision
```

### Tuned Logistic Regression

Best parameters:

```text
C      = 0.01
solver = liblinear
```

Best cross-validation Average Precision:

```text
0.7835
```

### Tuned Random Forest

Best parameters:

```text
n_estimators      = 200
max_depth         = 10
min_samples_split = 5
min_samples_leaf  = 1
```

Best cross-validation Average Precision:

```text
0.7944
```

Among the tuning experiments performed, the tuned Random Forest achieved the strongest Average Precision and was selected as the final model candidate.

---

## Classification Threshold Selection

A classification model normally uses a probability threshold of `0.5`.

Instead of automatically using the default threshold, this project selects the threshold using **out-of-fold predictions from the training data**.

```text
Training Data
     ↓
Cross-Validated Out-of-Fold Predictions
     ↓
Precision-Recall Curve
     ↓
F1 Evaluation
     ↓
Threshold Selection
```

This ensures that the final test set is **not used for threshold optimization**.

Selected threshold:

```text
0.3724
```

The threshold is stored in the model metadata and reused during inference.

---

## Final Test Evaluation

After the model configuration and classification threshold were locked, the model was evaluated on the untouched test set.

| Metric | Score |
|---|---:|
| Accuracy | 0.6504 |
| Precision | 0.6481 |
| Recall | 0.9744 |
| F1 Score | 0.7785 |
| ROC-AUC | 0.6989 |
| Average Precision | 0.7981 |

### Confusion Matrix

```text
[[ 141 1303]
 [  63 2400]]
```

Using scikit-learn's standard binary class ordering:

```text
True Negatives  = 141
False Positives = 1303
False Negatives = 63
True Positives  = 2400
```

The selected threshold provides very high recall for class `1`, but it also produces a large number of false positives.

This demonstrates an important production ML consideration: **model thresholds should be selected according to business costs and operational requirements, rather than optimizing a metric without considering its consequences.**

---

## Model Packaging

The selected trained model is saved as:

```text
artifacts/model/random_forest_model.joblib
```

The corresponding model metadata is stored as:

```text
artifacts/metadata/model_metadata.json
```

The metadata contains:

- Model name
- Model version
- Feature names
- Selected classification threshold
- Hyperparameters
- Final test metrics

This ensures that the inference application uses the same feature ordering and classification threshold selected during model development.

---

## Standalone Inference

The inference layer loads the saved model and metadata without retraining the model.

Prediction flow:

```text
Input Sensor Measurements
        ↓
Load Saved Model
        ↓
Predict Probability
        ↓
Apply Stored Threshold
        ↓
Return Prediction
```

This separates model training from production inference.

---

## FastAPI Model Serving

The trained model is exposed through a REST API using **FastAPI**.

Available endpoints:

```text
GET  /
GET  /health
POST /predict
```

### Health Check

```text
GET /health
```

returns the API health status and model version.

### Prediction Request

Example:

```json
{
  "engine_rpm": 700,
  "lub_oil_pressure": 2.493592,
  "fuel_pressure": 11.790927,
  "coolant_pressure": 3.178981,
  "lub_oil_temp": 84.144163,
  "coolant_temp": 81.632187
}
```

Example response:

```json
{
  "prediction": 1,
  "probability": 0.6581,
  "threshold": 0.3724
}
```

FastAPI automatically provides interactive Swagger API documentation.

When running locally:

```text
http://localhost:8000/docs
```

---

## Automated Testing

The project uses **pytest** for automated testing.

Current tests verify:

- API health endpoint
- Valid prediction request
- Missing input field validation
- Model inference output
- Probability range
- Classification threshold range

Run all tests:

```bash
python -m pytest tests/ -v
```

Current test status:

```text
4 passed
```

---

## Docker Containerization

The FastAPI inference application is containerized using Docker.

The Docker image contains:

```text
Application Code
        +
Model Artifacts
        +
Python Dependencies
        +
FastAPI / Uvicorn
```

### Build Docker Image

```bash
docker build -f docker/Dockerfile -t predictive-maintenance-api .
```

### Run Docker Container

```bash
docker run --name predictive-maintenance-container -p 8000:8000 predictive-maintenance-api
```

The container exposes the application on:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

The containerized API has been tested successfully for both health checks and model predictions.

---

## Project Structure

```text
predictive-maintenance-mlops/
│
├── app/
│   ├── __init__.py
│   └── app.py
│
├── artifacts/
│   ├── metadata/
│   │   └── model_metadata.json
│   └── model/
│       └── random_forest_model.joblib
│
├── data/
│   └── raw/
│       └── engine_data.csv
│
├── docker/
│   └── Dockerfile
│
├── scripts/
│   ├── run_baseline_experiments.py
│   ├── run_model_selection.py
│   ├── run_tuning_experiments.py
│   ├── test_ingestion.py
│   └── test_prediction.py
│
├── src/
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── preprocessing.py
│   │   └── validation.py
│   │
│   └── models/
│       ├── evaluate.py
│       ├── experiment.py
│       ├── predict.py
│       └── train.py
│
├── tests/
│   ├── test_api.py
│   └── test_model.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Technologies Used

**Machine Learning**

- Python
- Pandas
- Scikit-learn
- XGBoost

**MLOps**

- MLflow
- Git
- GitHub
- Pytest

**Model Serving**

- FastAPI
- Uvicorn

**Containerization**

- Docker

---

## MLOps Roadmap

The following enhancements are planned as the project evolves toward a more production-style MLOps architecture:

- [x] GitHub Actions CI pipeline
- [x] Automated testing on every push
- [x] Automated Docker image build
- [x] Push Docker image to Amazon ECR
- [ ] AWS deployment
- [ ] CloudWatch logging and monitoring
- [ ] Model and data monitoring
- [ ] Production artifact storage and model registry
- [ ] Automated retraining workflow

The roadmap will be updated as each stage is implemented.

---

## Key MLOps Concepts Demonstrated

This project currently demonstrates:

- Modular ML code
- Data validation
- Reproducible data splitting
- Multiple model experimentation
- Cross-validation
- MLflow experiment tracking
- Hyperparameter tuning
- Leakage-aware threshold selection
- Model artifact packaging
- Metadata-driven inference
- REST API model serving
- Automated testing
- Docker containerization
- Git-based version control

---

## Author

**Bhargavi Balineni**

Aspiring MLOps Engineer | Cloud Engineering Experience