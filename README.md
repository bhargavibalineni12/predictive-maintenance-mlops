# 🚀 Predictive Maintenance MLOps

An end-to-end **Machine Learning and MLOps project** for engine-condition classification from sensor measurements.

The project covers the complete ML lifecycle — from **data validation and model experimentation** to **MLflow experiment tracking, automated testing, Docker containerization, CI/CD, AWS deployment, real-time inference, prediction logging, and data-drift monitoring**.

> **Problem Framing:**  
> The available dataset contains independent sensor observations and a binary `Engine Condition` target. Therefore, this project performs **engine-condition classification for predictive maintenance**, rather than Remaining Useful Life (RUL) or future time-to-failure prediction.

---

## 🏗️ System Architecture

![Predictive Maintenance MLOps Architecture](docs/images/mlops-architecture.png)

The solution is organized into four major layers:

1. **Model Development** — data validation, preprocessing, model comparison, cross-validation, hyperparameter tuning, MLflow tracking, threshold selection, and artifact packaging.

2. **CI/CD** — GitHub Actions runs automated tests and validates that the Docker image builds successfully.

3. **AWS Deployment** — GitHub Actions authenticated to AWS using IAM OIDC, published a versioned Docker image to Amazon ECR, and the container was deployed using Amazon SageMaker AI Serverless Inference.

4. **Monitoring** — inference inputs and prediction metadata are logged, while Population Stability Index (PSI) compares production feature distributions with the reference distribution to detect input drift.

---

## 🔄 End-to-End Workflow

```text
Engine Sensor Dataset
        ↓
Data Ingestion & Validation
        ↓
Stratified Train/Test Split
        ↓
Baseline Model Training
        ↓
5-Fold Cross-Validation
        ↓
MLflow Experiment Tracking
        ↓
Hyperparameter Tuning
        ↓
Random Forest Model Selection
        ↓
Training-Only OOF Threshold Selection
        ↓
Untouched Final Test Evaluation
        ↓
Model + Metadata Artifacts
        ↓
FastAPI Inference Application
        ↓
Automated Testing
        ↓
Docker Containerization
        ↓
GitHub Actions CI
        ↓
Amazon ECR
        ↓
Amazon SageMaker Serverless
        ↓
Real-Time Cloud Inference
        ↓
Prediction Logging
        ↓
PSI-Based Data Drift Detection
```

---

## 🛠️ Technology Stack

| Area | Technologies |
|---|---|
| **Machine Learning** | Python, Pandas, NumPy, Scikit-learn, XGBoost |
| **Experiment Tracking** | MLflow |
| **Model Serving** | FastAPI, Uvicorn |
| **Testing** | pytest |
| **Containerization** | Docker |
| **Version Control** | Git, GitHub |
| **CI/CD** | GitHub Actions |
| **Cloud** | AWS |
| **Container Registry** | Amazon ECR |
| **Model Deployment** | Amazon SageMaker AI, Serverless Inference |
| **Authentication** | AWS IAM, GitHub OIDC |
| **Monitoring** | Structured Prediction Logging, Population Stability Index (PSI) |

---

## 📊 Dataset & Validation

The dataset contains **19,535 observations**, six numerical sensor features, and one binary target variable.

### Features

- `Engine rpm`
- `Lub oil pressure`
- `Fuel pressure`
- `Coolant pressure`
- `lub oil temp`
- `Coolant temp`

### Target

```text
Engine Condition
```

The target contains two classes:

```text
0
1
```

The project intentionally refers to them as **class `0` and class `1`** rather than assigning physical labels such as *normal* or *faulty*, because their exact physical meaning is not established by authoritative dataset documentation.

### Data Validation

Before training, the pipeline validates:

- Required columns
- Missing values
- Target values
- Numerical feature types
- Duplicate rows

The raw dataset passes the implemented validation checks.

### Train/Test Split

The dataset is divided into:

```text
Training set: 80%
Test set:     20%
```

A **stratified split** preserves the target distribution.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The final test set remains untouched during:

- Model comparison
- Hyperparameter tuning
- Classification-threshold selection

This reduces the risk of leaking test-set information into model-development decisions.

---

## 🧠 Model Development & Experiment Tracking

Four baseline machine-learning models were evaluated:

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

### Baseline Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6618 | 0.6797 | 0.8770 | 0.7658 | 0.6883 | 0.7833 |
| Decision Tree | 0.5863 | 0.6735 | 0.6673 | 0.6704 | 0.5577 | 0.6593 |
| Random Forest | 0.6532 | 0.6952 | 0.8011 | 0.7444 | 0.6770 | 0.7770 |
| XGBoost | 0.6432 | 0.6911 | 0.7850 | 0.7351 | 0.6627 | 0.7707 |

These results represent the baseline configurations used in this project.

---

## 🔬 MLflow Experiment Tracking

**MLflow** is used to track machine-learning experiments.

For each experiment, the project records:

- Model name
- Model parameters
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Average Precision

This provides a reproducible way to compare experiments instead of selecting models based only on manually observed results.

---

## 🔧 Hyperparameter Tuning

Logistic Regression and Random Forest were selected for further tuning.

`GridSearchCV` with **5-fold cross-validation** was used with:

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
n_estimators       = 200
max_depth          = 10
min_samples_split  = 5
min_samples_leaf   = 1
```

Best cross-validation Average Precision:

```text
0.7944
```

Among the tuning experiments performed, the **tuned Random Forest** achieved the strongest Average Precision and was selected as the final model candidate.

> **Evaluation Note:**  
> Model selection and tuning use the same training dataset through cross-validation. A nested cross-validation design would provide a more rigorous estimate of post-selection generalization and is included as a possible future improvement.

---

## 🎯 Classification Threshold Selection

Binary classifiers commonly begin with a probability threshold of `0.5`.

Instead of automatically using the default threshold, this project generates **out-of-fold (OOF) probabilities from the training data** and selects the classification threshold without using the final test set.

```text
Training Data
     ↓
Cross-Validated OOF Probabilities
     ↓
Precision / Recall / F1 Analysis
     ↓
Threshold Selection
```

Selected threshold:

```text
0.3724
```

The threshold is stored in the model metadata and reused during inference.

---

## 📈 Final Model Evaluation

After the model configuration and classification threshold were locked, the model was evaluated on the **untouched test set**.

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

The selected threshold produces **very high recall for class `1`**, but it also produces a large number of false positives.

This demonstrates an important production ML consideration:

> **Classification thresholds should ultimately reflect business costs and operational requirements rather than optimizing a metric in isolation.**

---

## 📦 Model Packaging & Inference

The selected trained model is saved as:

```text
artifacts/model/random_forest_model.joblib
```

The corresponding inference metadata is stored as:

```text
artifacts/metadata/model_metadata.json
```

The metadata contains:

- Model name
- Model version
- Expected feature names
- Selected classification threshold
- Hyperparameters
- Final test metrics

This ensures that the inference application uses the same feature ordering and classification threshold selected during model development.

### Standalone Inference Flow

```text
Input Sensor Measurements
        ↓
Load Saved Model + Metadata
        ↓
Predict Probability
        ↓
Apply Stored Threshold
        ↓
Return Prediction
```

The inference layer loads the saved artifacts directly without retraining.

---

## 🌐 FastAPI Model Serving

The trained model is exposed through a REST API using **FastAPI**.

### Local API Endpoints

```text
GET  /
GET  /health
POST /predict
```

### SageMaker-Compatible Endpoints

```text
GET  /ping
POST /invocations
```

`/ping` implements the health-check interface expected by the SageMaker inference container.

`/invocations` handles inference requests when the container is hosted by SageMaker.

### Example Request

```json
{
  "engine_rpm": 700,
  "lub_oil_pressure": 3.5,
  "fuel_pressure": 6.0,
  "coolant_pressure": 2.0,
  "lub_oil_temp": 75,
  "coolant_temp": 80
}
```

### Example Response

```json
{
  "prediction": 1,
  "probability": 0.7557877114110082,
  "threshold": 0.3723840180220012
}
```

---

## 🧪 Automated Testing

The project uses **pytest** to test the inference application, model behavior, and monitoring components.

Test coverage includes:

- API health checks
- Valid inference requests
- Missing-field/input validation
- Model inference output
- Probability-range validation
- Classification-threshold validation
- Prediction logging
- PSI calculation
- Normal-distribution drift checks
- Intentionally shifted production traffic
- Structured drift-report generation

Run the complete test suite with:

```bash
python -m pytest tests/ -v
```

Latest validated result:

```text
10 passed
```

The same test suite is executed automatically by GitHub Actions.

---

## 🐳 Docker Containerization

The FastAPI application, model artifacts, metadata, and Python dependencies are packaged into a custom Docker inference image.

### Build the Image

```bash
docker build -f docker/Dockerfile -t predictive-maintenance-api .
```

### Run Locally

```bash
docker run --rm -p 8080:8080 predictive-maintenance-api
```

The SageMaker-compatible service listens on:

```text
Port 8080
```

Swagger documentation is available locally at:

```text
http://127.0.0.1:8080/docs
```

Before AWS deployment, the container was validated locally through:

```text
GET  /ping
POST /invocations
```

---

## ⚙️ CI/CD with GitHub Actions

### Current Cost-Safe CI

The current GitHub Actions workflow intentionally performs operations that do **not create ongoing AWS resources**.

```text
Push / Pull Request
        ↓
Checkout Repository
        ↓
Set Up Python 3.12
        ↓
Install Dependencies
        ↓
Run pytest
        ↓
Build Docker Image
```

This keeps the repository continuously testable without automatically publishing a new ECR image on every commit.

### AWS Deployment Workflow Demonstrated

During the AWS deployment-validation phase, the CI/CD pipeline was extended to:

```text
Git Push
    ↓
GitHub Actions
    ↓
pytest + Docker Build
    ↓
GitHub OIDC Token
    ↓
AWS IAM Role
    ↓
Temporary AWS Credentials
    ↓
Amazon ECR
    ↓
Commit-SHA-Tagged Docker Image
    ↓
Amazon SageMaker Serverless
```

GitHub Actions authenticated to AWS using **IAM OpenID Connect (OIDC)**.

This avoided storing long-lived AWS access keys in GitHub.

After successful deployment validation, automatic ECR publication was disabled as part of the project's cost-control strategy.

---

## ☁️ AWS Deployment & Inference

The custom Docker container was successfully published to **Amazon ECR** and deployed using **Amazon SageMaker AI Serverless Inference**.

### Deployment Flow

```text
GitHub Actions
      ↓
Amazon ECR
      ↓
SageMaker Model
      ↓
Serverless Endpoint Configuration
      ↓
SageMaker Serverless Endpoint
      ↓
Real-Time Inference
```

The trained Random Forest model and its metadata were packaged directly inside the custom inference image.

Therefore, this deployment design did not require a separate S3 model-artifact URL.

### Serverless Configuration

```text
Memory size              = 2 GB
Maximum concurrency      = 1
Provisioned concurrency  = Disabled
```

Serverless inference was appropriate for this portfolio validation because inference requests were intermittent rather than a continuously provisioned production workload.

### Real-Time Cloud Inference Validation

After the endpoint reached:

```text
InService
```

the model was invoked through the **SageMaker Runtime API**.

Validated cloud response:

```json
{
  "prediction": 1,
  "probability": 0.7557877114110082,
  "threshold": 0.3723840180220012
}
```

This validated the complete cloud inference path:

```text
Client
   ↓
SageMaker Runtime
   ↓
Serverless Endpoint
   ↓
Custom Docker Container
   ↓
FastAPI /invocations
   ↓
Random Forest Model
   ↓
Stored Classification Threshold
   ↓
Prediction Response
```

---

## 💰 AWS Cost-Control Strategy

After successful cloud deployment and inference validation:

- The **SageMaker endpoint was deleted**
- The **SageMaker endpoint configuration was deleted**
- The **SageMaker model resource was deleted**
- The **Amazon ECR repository and stored images were deleted**
- The project-specific **SageMaker endpoint CloudWatch log group was deleted**
- Automatic **GitHub Actions → ECR pushes were disabled**
- **Provisioned concurrency was never enabled**

The GitHub repository retains the source code, deployment configuration/history, architecture, and screenshots required to reproduce and discuss the deployment without keeping unnecessary billable serving or storage resources running.

This demonstrates an important cloud-engineering practice:

> **Provision resources for a defined workload, validate the deployment, preserve reproducible evidence, and remove unnecessary cloud resources afterward.**

---

## 📊 Monitoring & Data Drift

The inference API records structured prediction events containing:

- UTC timestamp
- Model version
- Input features
- Prediction
- Probability
- Classification threshold

The project compares production input distributions with reference distributions using **Population Stability Index (PSI)**.

> **Important:** PSI measures **input distribution shift**. It does not prove that model accuracy has degraded. Model-performance monitoring requires ground-truth outcomes when they become available.

### Monitoring Policy

```text
PSI drift threshold              = 0.20
Minimum production sample window = 500
```

These values are **project-specific monitoring choices**, not universal thresholds.

### Drift Validation

Two traffic scenarios were tested.

#### Normal Traffic

```text
Production observations = 602
Result                  = No monitored feature crossed the PSI threshold
```

#### Intentionally Shifted Traffic

A simulated production batch of **500 observations** introduced changes to:

- `Engine rpm`
- `Coolant temp`

Detected drift:

| Feature | PSI |
|---|---:|
| Engine rpm | ≈ 1.7458 |
| Coolant temp | ≈ 5.6507 |

The drift detector also generates a **structured JSON report** that could be consumed by downstream monitoring or alerting systems.

---

## 📸 AWS Deployment Evidence

The following screenshots document the completed AWS deployment before cost-control cleanup.

### SageMaker Serverless Endpoint

![SageMaker Endpoint InService](docs/images/sagemaker-endpoint-inservice.png)

### Serverless Inference Configuration

![SageMaker Serverless Configuration](docs/images/sagemaker-serverless-config.png)

### Successful Cloud Inference

![SageMaker Inference Response](docs/images/sagemaker-inference-response.png)

### Amazon ECR Container Image

![Amazon ECR Container Image](docs/images/ecr-container-image.png)

### Successful GitHub Actions Pipeline

![GitHub Actions Success](docs/images/github-actions-success.png)

---

## 📁 Project Structure

```text
predictive-maintenance-mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml
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
│   ├── Dockerfile
│   └── serve.py
│
├── docs/
│   └── images/
│       ├── mlops-architecture.png
│       ├── sagemaker-endpoint-inservice.png
│       ├── sagemaker-serverless-config.png
│       ├── sagemaker-inference-response.png
│       ├── ecr-container-image.png
│       └── github-actions-success.png
│
├── notebooks/
│   └── predictive-maintenance-eda.ipynb
│
├── scripts/
│   ├── run_baseline_experiments.py
│   ├── run_model_selection.py
│   ├── run_tuning_experiments.py
│   ├── simulate_production_traffic.py
│   ├── test_ingestion.py
│   └── test_prediction.py
│
├── src/
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── preprocessing.py
│   │   └── validation.py
│   │
│   ├── models/
│   │   ├── evaluate.py
│   │   ├── experiment.py
│   │   ├── predict.py
│   │   └── train.py
│   │
│   └── monitoring/
│       ├── __init__.py
│       ├── drift_detector.py
│       └── prediction_logger.py
│
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_monitoring.py
│
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

Runtime prediction logs and generated monitoring reports are intentionally excluded from Git version control.

---

## ▶️ Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/bhargavibalineni12/predictive-maintenance-mlops.git
cd predictive-maintenance-mlops
```

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Automated Tests

```bash
python -m pytest tests/ -v
```

Expected validated result:

```text
10 passed
```

### 5. Start the FastAPI Application

```bash
uvicorn app.app:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### 6. Run with Docker

Build:

```bash
docker build -f docker/Dockerfile -t predictive-maintenance-api .
```

Run:

```bash
docker run --rm -p 8080:8080 predictive-maintenance-api
```

Open:

```text
http://127.0.0.1:8080/docs
```

---

## 🔮 Future Enhancements

- [ ] S3-backed production model-artifact storage
- [ ] SageMaker Model Registry
- [ ] CloudWatch-based centralized prediction logging and monitoring
- [ ] Model-performance monitoring when delayed ground-truth labels are available
- [ ] Automated retraining pipeline
- [ ] Automated deployment gates
- [ ] Model promotion and versioning strategy
- [ ] Feature Store or feature-management layer where justified
- [ ] Additional production security and governance controls
- [ ] Nested cross-validation for more rigorous post-selection model assessment

---

## 💡 Key MLOps Concepts Demonstrated

This project demonstrates:

- Modular ML development
- Data validation
- Reproducible train/test splitting
- Multiple-model experimentation
- Cross-validation
- MLflow experiment tracking
- Hyperparameter tuning
- Leakage-aware threshold selection
- Untouched final test evaluation
- Model artifact packaging
- Metadata-driven inference
- REST API model serving
- Automated testing
- Docker containerization
- Git-based version control
- GitHub Actions CI/CD
- Git commit-to-container traceability
- IAM OIDC authentication
- Temporary AWS credentials for CI/CD
- Amazon ECR container publishing
- Custom SageMaker inference containers
- SageMaker Serverless Inference
- Real-time cloud inference validation
- Prediction logging
- PSI-based data-drift detection
- Cloud-resource cost control and cleanup

---

## 🎓 Key Learning Outcome

This project demonstrates that **MLOps extends beyond training a machine-learning model**.

```text
Data
 ↓
Validation
 ↓
Experimentation
 ↓
Model Selection
 ↓
Artifact Packaging
 ↓
API Serving
 ↓
Testing
 ↓
Containerization
 ↓
CI/CD
 ↓
Cloud Deployment
 ↓
Inference Validation
 ↓
Monitoring
 ↓
Operational Cleanup
```

The project combines **machine learning development, software engineering, automated testing, CI/CD, containerization, cloud infrastructure, model serving, and monitoring** into one end-to-end MLOps workflow.

---

## 👩‍💻 Author

**Bhargavi Balineni**

Aspiring MLOps Engineer | Cloud Engineering Experience