# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Telco MLflow Experiment Tracking and Model Registry
# MAGIC
# MAGIC **Goal:** Track a reproducible Logistic Regression training run, log metrics and artifacts, and optionally register the trained model in Unity Catalog.
# MAGIC
# MAGIC **Prerequisites:** Notebooks 01–07 completed; access to `workspace.default.telco_train_raw`, `workspace.default.telco_test_raw`, and the fitted preprocessing pipeline in `/Volumes/workspace/default/telco_data/telco_preprocessing_pipeline`.
# MAGIC
# MAGIC **Important:** This notebook logs a **new** model training run. The previously reported test metrics are historical reference values, not metrics to assign to a new run. We will calculate fresh results. The saved preprocessing pipeline was fitted before the Notebook 07 validation split; this is a learning exercise rather than a fully leakage-free benchmark. Do not use the test set for model selection.
# MAGIC
# MAGIC **Notebook format:** Each numbered Markdown section is followed by its Python code cell. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Verify MLflow and Databricks environment
# MAGIC
# MAGIC **Why:** MLflow tracks experiments, runs, parameters, metrics, and model artifacts. We first confirm that the package and Spark session are available.
# MAGIC
# MAGIC **Checkpoint:** An MLflow version and Spark version appear.

# COMMAND ----------

import mlflow
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

print("MLflow version:", mlflow.__version__)
print("Spark version:", spark.version)
print("MLflow tracking URI:", mlflow.get_tracking_uri())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define project paths and create an MLflow experiment
# MAGIC
# MAGIC **Why:** An experiment groups related runs. Databricks workspace experiments use an absolute workspace path. Replace `YOUR_USERNAME` with your Databricks workspace username (typically an email address). If your workspace uses a different user-home convention, edit `experiment_path` accordingly.
# MAGIC
# MAGIC **Checkpoint:** The experiment ID prints. If permissions prevent creating it, choose a workspace folder you can write to.

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "default"
VOLUME = "/Volumes/workspace/default/telco_data"
TRAIN_TABLE = f"{CATALOG}.{SCHEMA}.telco_train_raw"
TEST_TABLE = f"{CATALOG}.{SCHEMA}.telco_test_raw"
PIPELINE_PATH = f"{VOLUME}/telco_preprocessing_pipeline"

# Edit this value before running.
YOUR_USERNAME = "YOUR_DATABRICKS_USERNAME"
if YOUR_USERNAME == "YOUR_USERNAME":
    raise ValueError("Replace YOUR_USERNAME with your Databricks workspace username before running Section 2.")

experiment_path = f"/Users/{YOUR_USERNAME}/telco_churn_mlflow"
mlflow.set_experiment(experiment_path)
experiment = mlflow.get_experiment_by_name(experiment_path)
print("Experiment:", experiment.name)
print("Experiment ID:", experiment.experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load the original raw train/test tables
# MAGIC
# MAGIC **Why:** Reuse the same split established in Notebook 03. The test set stays separate until final evaluation.
# MAGIC
# MAGIC **Checkpoint:** Expect approximately **5,634 training** and **1,409 test** records.

# COMMAND ----------

train_raw = spark.table(TRAIN_TABLE)
test_raw = spark.table(TEST_TABLE)
print("Train records:", train_raw.count())
print("Test records:", test_raw.count())
print("Churn distribution in training data:")
train_raw.groupBy("Churn").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load and apply the saved preprocessing pipeline
# MAGIC
# MAGIC **Why:** The fitted pipeline converts raw columns into the same 60-element feature vector used previously. Transform both datasets, then create binary labels (Yes = 1.0, No = 0.0).
# MAGIC
# MAGIC **Checkpoint:** Confirm 60 features and no null labels.
# MAGIC
# MAGIC **Methodology note:** For production-grade validation, fit a new preprocessing pipeline inside each training fold to prevent leakage.

# COMMAND ----------

preprocessor = PipelineModel.load(PIPELINE_PATH)
train_features = preprocessor.transform(train_raw).withColumn(
    "label", F.when(F.col("Churn") == "Yes", F.lit(1.0)).otherwise(F.lit(0.0))
)
test_features = preprocessor.transform(test_raw).withColumn(
    "label", F.when(F.col("Churn") == "Yes", F.lit(1.0)).otherwise(F.lit(0.0))
)
feature_count = train_features.select("features").first()["features"].size
print("Preprocessing stages:", len(preprocessor.stages))
print("Feature count:", feature_count)
print("Train labels:")
train_features.groupBy("label").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Choose training parameters and define evaluation helpers
# MAGIC
# MAGIC **Why:** MLflow parameters make runs reproducible. The evaluation helper measures churn-class precision, recall, F1, accuracy and ROC-AUC consistently.
# MAGIC
# MAGIC **Checkpoint:** The parameter dictionary appears.

# COMMAND ----------

PARAMS = {"maxIter": 100, "regParam": 0.0, "elasticNetParam": 0.0, "threshold": 0.5}

def evaluate_binary_predictions(predictions):
    multi = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    precision = multi.evaluate(predictions, {multi.metricName: "precisionByLabel", multi.metricLabel: 1.0})
    recall = multi.evaluate(predictions, {multi.metricName: "recallByLabel", multi.metricLabel: 1.0})
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": float(multi.evaluate(predictions, {multi.metricName: "accuracy"})),
            "precision_churn": float(precision), "recall_churn": float(recall),
            "f1_churn": float(f1), "roc_auc": float(auc_eval.evaluate(predictions))}

print("Training parameters:", PARAMS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Start an MLflow run, train the model, and log parameters
# MAGIC
# MAGIC **Why:** A run is the record of one training attempt. We train Logistic Regression and log the exact settings used.
# MAGIC
# MAGIC **Checkpoint:** A run ID prints. We deliberately close the run after logging its metrics in Section 7; if this section fails, end the active run with `mlflow.end_run()` before retrying.

# COMMAND ----------

# End any leftover run from an earlier partial execution.
if mlflow.active_run() is not None:
    mlflow.end_run()

run = mlflow.start_run(run_name="telco_logistic_regression_baseline")
mlflow.set_tags({"project": "Telco Customer Churn", "notebook": "08", "model_family": "LogisticRegression"})
mlflow.log_params(PARAMS)
mlflow.log_param("num_features", feature_count)
mlflow.log_param("train_records", train_raw.count())
model = LogisticRegression(featuresCol="features", labelCol="label", **PARAMS).fit(train_features)
print("Run ID:", run.info.run_id)
print("Model coefficients:", len(model.coefficients))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Evaluate the new run and log metrics
# MAGIC
# MAGIC **Why:** We measure performance on the held-out test set, log the actual results, and close the run. These scores may match earlier notebooks, but are computed anew.
# MAGIC
# MAGIC **Checkpoint:** Five metrics print and the MLflow run ends. Avoid comparing alternative configurations on this test set; use a validation split for future tuning.

# COMMAND ----------

predictions = model.transform(test_features)
metrics = evaluate_binary_predictions(predictions)
mlflow.log_metrics(metrics)
for name, value in metrics.items():
    print(f"{name}: {value:.6f}")
mlflow.end_run()
print("Run complete:", run.info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Inspect the run and compare with historical findings
# MAGIC
# MAGIC **Why:** The MLflow tracking API can retrieve logged runs. The historical values from Notebook 07 are shown only as reference, not as results of this new run.
# MAGIC
# MAGIC **Checkpoint:** A run summary table appears.

# COMMAND ----------

runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=20)
columns = [c for c in ["run_id", "status", "params.maxIter", "params.threshold", "metrics.accuracy", "metrics.recall_churn", "metrics.f1_churn", "metrics.roc_auc"] if c in runs.columns]
display(runs[columns])
print("Historical Notebook 07 test ROC-AUC (reference only):", 0.8384534733991994)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Log a confusion-matrix CSV as an artifact
# MAGIC
# MAGIC **Why:** Metrics summarize performance; artifacts preserve supporting evidence. A confusion matrix makes false negatives and false positives visible.
# MAGIC
# MAGIC **Checkpoint:** Four count combinations display, and the CSV is attached to the MLflow run.

# COMMAND ----------

import csv, tempfile, os
cm_rows = predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").collect()
for r in cm_rows:
    print(f"actual={int(r['label'])}, predicted={int(r['prediction'])}, count={r['count']}")
with tempfile.TemporaryDirectory() as tmp:
    artifact_path = os.path.join(tmp, "confusion_matrix.csv")
    with open(artifact_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "prediction", "count"])
        for r in cm_rows:
            writer.writerow([int(r["label"]), int(r["prediction"]), r["count"]])
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_artifact(artifact_path, artifact_path="evaluation")
print("Logged confusion matrix artifact to run:", run.info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Log the Spark ML model to MLflow
# MAGIC
# MAGIC **Why:** Model logging packages the trained model for later loading or registration. Databricks runtimes vary in their support for Spark Connect and `mlflow.spark.log_model`; if this cell reports an unsupported operation, use the fallback in the next section or consult your workspace's supported MLflow/Spark combination.
# MAGIC
# MAGIC **Checkpoint:** A model artifact is logged and its run URI is printed. This step may require cluster-specific permissions and model logging support.

# COMMAND ----------

# DBTITLE 1,untitled
import pandas as pd
from mlflow.models import infer_signature

MODEL_ARTIFACT_NAME = "telco_logistic_regression"
model_uri = f"runs:/{run.info.run_id}/{MODEL_ARTIFACT_NAME}"

# Input example from real preprocessed rows. Signature inference through the model fails
# on Spark Connect (array vs VectorUDT mismatch), so infer it explicitly from outputs.
example_rows = train_features.select("features").limit(3).collect()
input_example = pd.DataFrame({"features": [list(r["features"].toArray()) for r in example_rows]})
pred_rows = model.transform(train_features.limit(3)).select("prediction").collect()
signature = infer_signature(input_example, pd.DataFrame({"prediction": [float(r["prediction"]) for r in pred_rows]}))

# pyspark 4.x imports its Spark Connect module whenever a SparkSession is created, so a
# fresh serving environment needs these declared or the model cannot even be loaded.
EXTRA_PIP_REQUIREMENTS = ["grpcio==1.76.0", "grpcio-status==1.76.0", "zstandard==0.25.0"]

try:
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.spark.log_model(
            spark_model=model,
            artifact_path=MODEL_ARTIFACT_NAME,
            dfs_tmpdir=f"{VOLUME}/mlflow_tmp",
            signature=signature,
            input_example=input_example,
            extra_pip_requirements=EXTRA_PIP_REQUIREMENTS,
        )
    print("Model logged to MLflow:", model_uri)
    model_logged = True
except Exception as exc:
    model_logged = False
    print("MLflow Spark model logging was not supported in this environment.")
    print(type(exc).__name__ + ":", str(exc)[:700])
    print("Proceed to Section 11 to save a native Spark model backup.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Save a native Spark ML backup
# MAGIC
# MAGIC **Why:** A native Spark ML model saved to your Unity Catalog volume provides a fallback if MLflow Spark model logging is unavailable. It is separate from the original Notebook 04 model.
# MAGIC
# MAGIC **Checkpoint:** The model saves and reloads successfully.

# COMMAND ----------

BACKUP_MODEL_PATH = f"{VOLUME}/telco_08_logistic_regression_backup"
model.write().overwrite().save(BACKUP_MODEL_PATH)
reloaded = type(model).load(BACKUP_MODEL_PATH)
print("Saved native Spark model:", BACKUP_MODEL_PATH)
print("Reloaded coefficient count:", len(reloaded.coefficients))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Optional: Register the logged model in Unity Catalog
# MAGIC
# MAGIC **Why:** The Unity Catalog Model Registry stores governed model versions. The three-part name below is `catalog.schema.model`. Registration requires suitable permissions, and `workspace` must support registered models in your environment.
# MAGIC
# MAGIC **Safety:** This cell **does not register anything by default**. Set `REGISTER_MODEL = True` only after verifying the target name and permissions. The model must have been logged successfully in Section 10.
# MAGIC
# MAGIC **Checkpoint:** If enabled, the registered model name and version print. Otherwise the notebook safely skips registration.

# COMMAND ----------

# DBTITLE 1,untitled
REGISTER_MODEL = True  # Enabled after isolated serving validation of the logged artifact passed.
REGISTERED_MODEL_NAME = "workspace.default.telco_churn_logistic_regression"

if not REGISTER_MODEL:
    print("Registration skipped. Set REGISTER_MODEL=True when ready.")
elif not model_logged:
    print("Cannot register: MLflow model logging did not succeed in Section 10.")
else:
    mlflow.set_registry_uri("databricks-uc")
    registered = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME, await_registration_for=300)
    print("Registered model:", registered.name)
    print("Version:", registered.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Final summary and checkpoint
# MAGIC
# MAGIC **What you learned:** MLflow experiment tracking, run IDs, parameters, metrics, confusion-matrix artifacts, model logging, native backup, and optional Unity Catalog registration.
# MAGIC
# MAGIC **Report these outputs:** experiment ID, run ID, evaluation metrics, model-logging status, backup path, and (if used) registered model version.
# MAGIC
# MAGIC **Next:** Notebook 08A — MLflow experiment findings, or Notebook 09 — leakage-safe cross-validation and tuning. **Do not claim production readiness** based on this notebook alone.

# COMMAND ----------

print("Experiment:", experiment_path)
print("Run ID:", run.info.run_id)
print("Model logged to MLflow:", model_logged)
print("Native backup:", BACKUP_MODEL_PATH)
print("Registered in Unity Catalog:", bool(REGISTER_MODEL and model_logged))
print("Test ROC-AUC:", round(metrics["roc_auc"], 6))
print("Test churn recall:", round(metrics["recall_churn"], 6))