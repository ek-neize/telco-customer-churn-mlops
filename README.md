# Telco Customer Churn Prediction | Databricks, PySpark & MLflow

End-to-end customer churn modeling and MLOps portfolio project built with **Databricks, Apache Spark MLlib, Delta Lake, Unity Catalog, and MLflow**.

### Explore the Project

[Modeling Notebooks](notebooks/01_modeling/) | [Advanced Analysis](notebooks/02_advanced_analysis/) | [MLOps Examples](notebooks/03_mlops_examples/) | [Findings Reports](reports/)

> **Portfolio status:** Existing Databricks notebook source has been exported and organized for review. The code has **not** been rerun in a clean environment, and the advanced deployment examples are **not** a production release or a turnkey one-click deployment.

## Business problem

Telecom providers can use predicted churn risk to prioritize customer retention outreach. This project explores customer attributes, trains and compares classifiers, assesses probability thresholds, and demonstrates experiment tracking and guarded batch-scoring patterns. Predicted risk does not by itself establish that an intervention will prevent churn.

## Results

Previously recorded evaluation on a held-out **1,409-customer test set**, with a Logistic Regression model and decision threshold of **0.50**:

| Metric | Value |
|---|---:|
| Accuracy | 0.8006 |
| Precision (churn) | 0.6700 |
| Recall (churn) | 0.5248 |
| F1 (churn) | 0.5886 |
| ROC-AUC | 0.8385 |

At threshold **0.40**, the recorded recall increased to **0.6292**, while precision decreased to **0.6148** (F1 **0.6219**). This illustrates a business trade-off, not a universally optimal threshold.

The comparison notebooks explore **Logistic Regression, Gradient-Boosted Trees, and Random Forest**. Logistic Regression was selected in the recorded comparison. Metrics are historical results from the original workspace; they have not been independently reproduced from this export.

## Workflow

```text
Telco customer data
       |
       v
Exploration / cleaning -> Feature engineering / train-test split
       |                              |
       +------------------------------v
                            Spark ML classifiers
                                    |
                         Evaluation and optimization
                                    |
                         MLflow experiment tracking
                                    |
                  Batch inference / validation / monitoring
```

## Repository guide

| Directory | Contents |
|---|---|
| `notebooks/01_modeling/` | Exploration, preprocessing, feature engineering, model training, evaluation, optimization, comparison, MLflow |
| `notebooks/02_advanced_analysis/` | Cost-sensitive thresholds, lift/gains, feature importance, calibration |
| `notebooks/03_mlops_examples/` | Deployment, batch scoring, monitoring, guarded Job examples and draft production deployment notebooks |
| `reports/` | Notebook-formatted findings and supporting analysis |

The `.py` files are **Databricks source-format notebooks**, not ordinary standalone Python scripts. They include `# COMMAND ----------` and `# MAGIC` markers and are intended for import into Databricks.

## Running the project

1. Set up a Databricks workspace with Spark, MLflow, Delta Lake and Unity Catalog permissions.
2. Import the source notebooks from `notebooks/01_modeling/` into your Databricks workspace.
3. Supply the Telco Customer Churn dataset yourself, and update the workspace-specific data locations, catalog/schema names, volumes and experiment paths in the notebooks.
4. Run notebooks **01 through 08** in sequence, inspecting checkpoints and required permissions.
5. Review the advanced analysis notebooks separately. **Do not run deployment/Job notebooks blindly**: they may register models, append registry rows, write Delta tables, or submit other side effects.

The export does not include source data, trained model artifacts, MLflow tracking state, or Unity Catalog tables. This is therefore a documented **Databricks workspace project**, not a fully portable executable package.

## Modeling and validation notes

- The dataset contained **7,043 customers**; the saved split was **5,634 train / 1,409 test**.
- The fitted Spark preprocessing pipeline has **20 stages** and a recorded **60-feature** output.
- The exported notebook documentation notes that preprocessing was fitted before the later validation split used in the model-comparison exercise; that validation comparison should **not** be presented as fully leakage-free.
- Results from the test set are descriptive of this experiment; production generalization, calibration, fairness, drift, and intervention effectiveness require further validation.
- Some notebooks contain fixed Databricks workspace paths (`workspace.default...`, `/Volumes/workspace/...`) and require adaptation.

## MLOps scope

Examples cover MLflow experiment tracking and model registry, snapshot-based batch inference, quality gates, idempotency-oriented publishing patterns, and monitoring. These demonstrate engineering considerations, **not a claim of production deployment**. The original project also explored a staging release plan, which was not completed or deployed.

## Publication safety

- The exported notebook files were checked for common credential patterns, and a personal Databricks username/email was replaced with `YOUR_DATABRICKS_USERNAME`.
- Before pushing, manually review the files for any remaining personal information, secrets, private URLs, workspace IDs, proprietary data, and generated outputs.
- Do not commit Databricks access tokens, private datasets, MLflow model artifacts, or credentials.

## Technology

Python · PySpark · Spark MLlib · Databricks · Delta Lake · Unity Catalog · MLflow · SQL · Classification · MLOps

## License and data

No dataset is included in this repository. Confirm the dataset's license and attribution before redistributing it. No software license is asserted by this package; add one only if you choose to grant reuse rights.
