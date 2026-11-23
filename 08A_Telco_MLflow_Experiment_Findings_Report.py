# Databricks notebook source
# DBTITLE 1,08A title
# MAGIC %md
# MAGIC # 08A — Telco MLflow Experiment Findings Report
# MAGIC
# MAGIC **Purpose:** Document the actual Notebook 08 MLflow tracking results, the model's Unity Catalog registration, and the serving-validation findings. This report **does not retrain or re-log** any model; it reads the recorded experiment.
# MAGIC
# MAGIC **Sources:** Live MLflow experiment `/Users/YOUR_DATABRICKS_USERNAME/telco_churn_mlflow` (run `b45de5f13b7b4aa1aaa2318045b4a03f`) and recorded Notebook 08 outputs.
# MAGIC
# MAGIC **Important limitation:** The saved preprocessing pipeline predates the Notebook 07 validation split, and the test set has been examined in earlier notebooks — the test metrics below are a learning exercise, not an unbiased production estimate.

# COMMAND ----------

# DBTITLE 1,Section 1 intro
# MAGIC %md
# MAGIC ## 1. Experiment checkpoint
# MAGIC
# MAGIC Confirm the experiment exists and locate the tracked run. The rest of the report pulls its values from MLflow rather than hardcoding them where possible.

# COMMAND ----------

# DBTITLE 1,Section 1 experiment checkpoint
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

EXPERIMENT_PATH = "/Users/YOUR_DATABRICKS_USERNAME/telco_churn_mlflow"
REPORTED_RUN_ID = "b45de5f13b7b4aa1aaa2318045b4a03f"

experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=20)
print("Experiment:", experiment.name)
print("Experiment ID:", experiment.experiment_id)
print("Total runs recorded:", len(runs))
print("Reported run present:", REPORTED_RUN_ID in set(runs["run_id"]))

# COMMAND ----------

# DBTITLE 1,Section 2 intro
# MAGIC %md
# MAGIC ## 2. Executive summary
# MAGIC
# MAGIC What Notebook 08 tracked and produced, at a glance.

# COMMAND ----------

# DBTITLE 1,Section 2 summary table
summary = pd.DataFrame([
    ("Experiment path", EXPERIMENT_PATH),
    ("Run name", "telco_logistic_regression_baseline"),
    ("Run ID", REPORTED_RUN_ID),
    ("Algorithm", "LogisticRegression (60 features)"),
    ("Training records", 5634),
    ("Test records", 1409),
    ("Model logged with signature + input example", "Yes"),
    ("Native Spark backup", "/Volumes/workspace/default/telco_data/telco_08_logistic_regression_backup"),
    ("Registered model", "workspace.default.telco_churn_logistic_regression"),
    ("Registered version", "1"),
], columns=["Item", "Value"])
display(summary)

# COMMAND ----------

# DBTITLE 1,Section 3 intro
# MAGIC %md
# MAGIC ## 3. Tracked run parameters and test metrics
# MAGIC
# MAGIC Parameters come from the run record; metrics are the fresh Notebook 08 test-set results. The Notebook 07 value is shown only as a historical reference.

# COMMAND ----------

# DBTITLE 1,Section 3 params and metrics
row = runs.set_index("run_id").loc[REPORTED_RUN_ID]
print("Parameters:")
for c in sorted(col for col in runs.columns if col.startswith("params.")):
    print("  ", c.replace("params.", ""), "=", row[c])

metric_names = ["accuracy", "precision_churn", "recall_churn", "f1_churn", "roc_auc"]
metrics = pd.DataFrame([(m, float(row[f"metrics.{m}"])) for m in metric_names], columns=["Metric", "Value"])
display(metrics)
print("Historical Notebook 07 test ROC-AUC (reference only):", 0.8384534733991994)

# COMMAND ----------

# DBTITLE 1,Section 4 intro
# MAGIC %md
# MAGIC ## 4. Test metric chart
# MAGIC
# MAGIC Visualize the five logged test metrics. All are churn-class scores (precision/recall/F1 target label 1) or overall (accuracy, ROC-AUC).

# COMMAND ----------

# DBTITLE 1,Section 4 metric chart
ax = metrics.set_index("Metric")["Value"].mul(100).plot(kind="bar", figsize=(10, 4), rot=0, legend=False)
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 100)
ax.set_title("Notebook 08 run: test-set metrics")
for p in ax.patches:
    ax.annotate(f"{p.get_height()/100:.3f}", (p.get_x() + p.get_width()/2, p.get_height()), ha="center", va="bottom")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Section 5 intro
# MAGIC %md
# MAGIC ## 5. Confusion-matrix artifact
# MAGIC
# MAGIC The counts logged as the run's `evaluation/confusion_matrix.csv` artifact. False negatives (actual churn = 1, predicted 0) are the costly errors for a retention program.

# COMMAND ----------

# DBTITLE 1,Section 5 confusion matrix
local_cm = mlflow.artifacts.download_artifacts(f"runs:/{REPORTED_RUN_ID}/evaluation/confusion_matrix.csv")
cm = pd.read_csv(local_cm)
display(cm)
fn = cm[(cm.label == 1) & (cm.prediction == 0)]["count"].sum()
fp = cm[(cm.label == 0) & (cm.prediction == 1)]["count"].sum()
print("False negatives (missed churners):", int(fn), "| False positives (unnecessary offers):", int(fp))

# COMMAND ----------

# DBTITLE 1,Section 6 intro
# MAGIC %md
# MAGIC ## 6. Model artifact and Unity Catalog registration status
# MAGIC
# MAGIC Inspect what was actually logged: signature, input example, declared dependencies, and the registered UC model version.

# COMMAND ----------

# DBTITLE 1,Section 6 artifact and registration
from mlflow import MlflowClient

model_uri = f"runs:/{REPORTED_RUN_ID}/telco_logistic_regression"
meta = mlflow.models.Model.load(model_uri)
print("Signature:", meta.signature)
print("Input example saved:", meta.saved_input_example_info is not None)
local_req = mlflow.artifacts.download_artifacts(f"{model_uri}/requirements.txt")
print("Declared requirements:", open(local_req).read().split())

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")
mv = client.get_model_version("workspace.default.telco_churn_logistic_regression", "1")
print("Registered model:", mv.name, "| version:", mv.version, "| status:", mv.status)

# COMMAND ----------

# DBTITLE 1,Section 7 intro
# MAGIC %md
# MAGIC ## 7. Serving-validation findings (serverless)
# MAGIC
# MAGIC Notebook 08's model-logging and validation path surfaced three environment facts worth keeping for the project record.
# MAGIC
# MAGIC * **UC volume staging requirement:** on serverless compute, `mlflow.spark.log_model` fails unless `dfs_tmpdir` (or `MLFLOW_DFS_TMP`) points at a UC volume path.
# MAGIC * **Dependency completeness:** pyspark 4.x imports its Spark Connect modules whenever a session is created, so a fresh serving environment needs `grpcio`, `grpcio-status`, and `zstandard` declared — they were added via `extra_pip_requirements`.
# MAGIC * **Validation environment fidelity:** the first isolated check failed because the notebook session's `SPARK_REMOTE` (Spark Connect) configuration leaked into the validation subprocess, which then could not create the local Spark session Spark ML loading requires (`PySparkRuntimeError: CANNOT_CONFIGURE_SPARK_CONNECT_MASTER`). With those variables removed and `MLFLOW_DFS_TMP` set, the same artifact **passed** validation in a fresh uv environment (predictions `[0.0, 0.0, 1.0]`).

# COMMAND ----------

# DBTITLE 1,Section 7 findings table
findings = pd.DataFrame([
    ("mlflow.spark.log_model (initial attempt)", "Failed: SparkML needs a UC volume staging path on serverless", "Fixed with dfs_tmpdir=/Volumes/workspace/default/telco_data/mlflow_tmp"),
    ("Isolated validation (default notebook env)", "Failed: SPARK_REMOTE leaked into the subprocess (CANNOT_CONFIGURE_SPARK_CONNECT_MASTER)", "Root cause confirmed by re-running the exact failing command"),
    ("Artifact requirements", "Missing grpcio / grpcio-status / zstandard needed by pyspark 4.x", "Re-logged with extra_pip_requirements"),
    ("Isolated validation (serving-faithful env)", "Passed: fresh uv venv installed declared requirements; predictions [0.0, 0.0, 1.0]", "Artifact validated for serving-style environments"),
], columns=["Check", "Outcome", "Resolution"])
display(findings)

# COMMAND ----------

# DBTITLE 1,Section 8 conclusions
# MAGIC %md
# MAGIC ## 8. Conclusions and next steps
# MAGIC
# MAGIC **What was achieved:** a fully tracked MLflow run (parameters, five test metrics, confusion-matrix artifact), a signed model artifact with complete declared dependencies, a native Spark ML backup in the project volume, and a registered Unity Catalog model version (`workspace.default.telco_churn_logistic_regression` v1).
# MAGIC
# MAGIC **Caveats:** spark-flavor models are not pyfunc-servable from serverless compute with default session configuration; deploying this version would require classic Spark compute or a re-packaged flavor. The test set has been reused across notebooks, so these metrics are not an unbiased production estimate.
# MAGIC
# MAGIC **Next:** Notebook 09 — leakage-safe cross-validation and tuning.