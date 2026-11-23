# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 10 — Telco Batch Scoring Automation, Monitoring & Data Drift
# MAGIC
# MAGIC **Project:** Telco Customer Churn ML / MLOps  
# MAGIC **Environment:** Databricks, Unity Catalog, Spark ML, MLflow, Delta Lake  
# MAGIC **Prerequisite:** Notebooks 03, 08, 09 completed.
# MAGIC
# MAGIC ### Learning objectives
# MAGIC 1. Turn batch scoring into a repeatable, parameterized process.
# MAGIC 2. Keep historical prediction batches instead of overwriting prior results.
# MAGIC 3. Record operational and data-quality monitoring metrics.
# MAGIC 4. Measure **feature distribution drift** against a reference population.
# MAGIC 5. Visualize results and prepare for Databricks Workflows scheduling.
# MAGIC
# MAGIC **Important:** Drift alerts do **not** establish model performance degradation. Without newly observed churn labels, this notebook cannot measure current accuracy, recall, or concept drift. The test dataset is used here as a **demonstration batch**; production jobs should read newly arrived, unlabeled customers.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration and runtime parameters
# MAGIC
# MAGIC We use widgets so Databricks Workflows can supply parameters. The default input is the previously saved raw test table. A unique `batch_id` identifies each scoring run. Set a new `batch_id` when you want a separate historical batch; rerunning the same ID replaces only that batch's rows.
# MAGIC

# COMMAND ----------

# Remove the old widget and its saved value
try:
    dbutils.widgets.remove("batch_id")
    print("Old batch_id widget removed.")
except Exception as e:
    print("Widget removal:", str(e))

# COMMAND ----------

import os
from datetime import datetime, timezone
from uuid import uuid4

import mlflow
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

# --------------------------------------------------
# Unity Catalog configuration
# --------------------------------------------------

CATALOG = "workspace"
SCHEMA = "default"

MODEL_NAME = "workspace.default.telco_churn_logistic_regression"
MODEL_VERSION = "1"

PREPROCESSING_PATH = (
    "/Volumes/workspace/default/telco_data/"
    "telco_preprocessing_pipeline"
)

MLFLOW_TMP_DIR = (
    "/Volumes/workspace/default/telco_data/mlflow_tmp"
)

DEFAULT_INPUT = "workspace.default.telco_test_raw"
REFERENCE_TABLE = "workspace.default.telco_train_raw"

PREDICTIONS_HISTORY = "workspace.default.telco_10_prediction_history"
MONITORING_TABLE = "workspace.default.telco_10_batch_monitoring"
DRIFT_TABLE = "workspace.default.telco_10_feature_drift"

# --------------------------------------------------
# Databricks Job parameters
# --------------------------------------------------

try:
    dbutils.widgets.text("input_table", DEFAULT_INPUT, "Raw input table")
    dbutils.widgets.text("batch_id", "", "Batch ID (optional)")
    dbutils.widgets.text("model_version", MODEL_VERSION, "Registered model version")

    INPUT_TABLE = dbutils.widgets.get("input_table").strip()
    REQUESTED_BATCH_ID = dbutils.widgets.get("batch_id").strip()
    MODEL_VERSION = dbutils.widgets.get("model_version").strip()

except NameError:
    # Fallback for environments without dbutils
    INPUT_TABLE = DEFAULT_INPUT
    REQUESTED_BATCH_ID = ""

# --------------------------------------------------
# Generate or reuse a batch identifier
# --------------------------------------------------

if REQUESTED_BATCH_ID:
    # Explicit ID supplied: useful for retrying a batch
    BATCH_ID = REQUESTED_BATCH_ID
else:
    # Generate a unique ID for a new batch
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid4().hex[:8]

    BATCH_ID = f"telco_{timestamp}_{unique_suffix}"

# --------------------------------------------------
# Registered model URI
# --------------------------------------------------

MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

assert INPUT_TABLE and BATCH_ID and MODEL_VERSION, (
    "All required parameters must be nonempty"
)

print("Batch ID:", BATCH_ID)
print("Input table:", INPUT_TABLE)
print("Model URI:", MODEL_URI)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load registered model and fitted preprocessing
# MAGIC
# MAGIC The registered MLflow model is a **one-stage `PipelineModel`** containing Logistic Regression only. We must apply the separately saved 20-stage preprocessing pipeline **once** before scoring. Shared/serverless Databricks compute needs a Unity Catalog Volume for MLflow temporary files.
# MAGIC

# COMMAND ----------

os.makedirs(MLFLOW_TMP_DIR, exist_ok=True)
os.environ["MLFLOW_DFS_TMP"] = MLFLOW_TMP_DIR
mlflow.set_registry_uri("databricks-uc")
preprocessing_model = PipelineModel.load(PREPROCESSING_PATH)
registered_model = mlflow.spark.load_model(MODEL_URI, dfs_tmpdir=MLFLOW_TMP_DIR)
print("Preprocessing stages:", len(preprocessing_model.stages))
print("Registered model stages:", len(registered_model.stages))
print("Registered model type:", type(registered_model).__name__)
assert len(preprocessing_model.stages) == 20
assert len(registered_model.stages) == 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load and validate raw input
# MAGIC
# MAGIC The raw table must contain `customerID` and the same predictor fields used during training. Drop the ground-truth `Churn` and any `label` column **before inference**. Never use the target as a model feature. Fail fast on duplicate IDs or missing IDs to avoid silently corrupting downstream history.
# MAGIC

# COMMAND ----------

raw_df = spark.table(INPUT_TABLE)
required = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService"]
missing = [c for c in required if c not in raw_df.columns]
assert not missing, f"Missing required input columns: {missing}"
input_count = raw_df.count()
assert input_count > 0, "Input batch is empty"
assert raw_df.filter(F.col("customerID").isNull()).count() == 0, "Null customerID detected"
assert raw_df.select("customerID").distinct().count() == input_count, "Duplicate customer IDs detected"
inference_df = raw_df.drop("Churn", "label")
print("Input customers:", input_count)
print("Churn present in model input?", "Churn" in inference_df.columns)
display(inference_df.select("customerID", "tenure", "Contract", "MonthlyCharges").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run repeatable batch scoring
# MAGIC
# MAGIC Extract class-1 churn probability, apply the validation-selected 0.40 decision threshold, and assign **illustrative** low/medium/high risk tiers. The 0.70 high-risk cutoff is a business reporting choice, not a separately validated threshold.
# MAGIC

# COMMAND ----------

prepared_df = preprocessing_model.transform(inference_df)
scored_raw_df = registered_model.transform(prepared_df)
scored_df = (
    scored_raw_df
    .withColumn("churn_probability", vector_to_array(F.col("probability"))[1].cast("double"))
    .withColumn("optimized_prediction", F.when(F.col("churn_probability") >= 0.40, F.lit(1)).otherwise(F.lit(0)))
    .withColumn("risk_level", F.when(F.col("churn_probability") >= 0.70, "High Risk")
                .when(F.col("churn_probability") >= 0.40, "Medium Risk")
                .otherwise("Low Risk"))
)
output_df = (scored_df.select("customerID", "churn_probability", "optimized_prediction", "risk_level")
             .withColumn("batch_id", F.lit(BATCH_ID))
             .withColumn("source_table", F.lit(INPUT_TABLE))
             .withColumn("model_name", F.lit(MODEL_NAME))
             .withColumn("model_version", F.lit(MODEL_VERSION))
             .withColumn("scored_at", F.current_timestamp()))
assert output_df.count() == input_count
assert output_df.filter(F.col("churn_probability").isNull() | (F.col("churn_probability") < 0) | (F.col("churn_probability") > 1)).count() == 0
print("Scored customers:", output_df.count())
display(output_df.orderBy(F.desc("churn_probability")).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Persist historical batches in Delta
# MAGIC
# MAGIC Unlike Notebook 09's demonstration `overwrite`, this notebook preserves historical runs. A Delta `replaceWhere` write replaces only rows with the **same batch ID**, making retries idempotent while retaining other batches. The table is created automatically on the first run.
# MAGIC

# COMMAND ----------

# Delta overwrite with replaceWhere only replaces the matching batch partition.
# Partitioning by batch_id makes each scoring batch easy to inspect and retry.
(output_df.write.format("delta")
 .mode("overwrite")
 .option("replaceWhere", f"batch_id = '{BATCH_ID.replace(chr(39), chr(39)+chr(39))}'")
 .partitionBy("batch_id")
 .saveAsTable(PREDICTIONS_HISTORY))
saved_batch_df = spark.table(PREDICTIONS_HISTORY).filter(F.col("batch_id") == BATCH_ID)
assert saved_batch_df.count() == input_count
print("Saved batch:", BATCH_ID, "| Records:", saved_batch_df.count())
print("History table:", PREDICTIONS_HISTORY)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Operational monitoring and data-quality checks
# MAGIC
# MAGIC Monitor total records, distinct customers, missing probabilities, mean score, number flagged at 0.40, and risk-tier counts. Write one row per batch using the same batch-replacement pattern.
# MAGIC

# COMMAND ----------

monitor_df = (saved_batch_df.agg(
    F.count("*").alias("records"),
    F.countDistinct("customerID").alias("unique_customers"),
    F.sum(F.when(F.col("churn_probability").isNull(), 1).otherwise(0)).cast("long").alias("null_probabilities"),
    F.avg("churn_probability").alias("mean_churn_probability"),
    F.sum(F.when(F.col("optimized_prediction") == 1, 1).otherwise(0)).cast("long").alias("flagged_customers"),
    F.sum(F.when(F.col("risk_level") == "Low Risk", 1).otherwise(0)).cast("long").alias("low_risk"),
    F.sum(F.when(F.col("risk_level") == "Medium Risk", 1).otherwise(0)).cast("long").alias("medium_risk"),
    F.sum(F.when(F.col("risk_level") == "High Risk", 1).otherwise(0)).cast("long").alias("high_risk")
).withColumn("batch_id", F.lit(BATCH_ID))
 .withColumn("model_version", F.lit(MODEL_VERSION))
 .withColumn("monitored_at", F.current_timestamp()))
(monitor_df.write.format("delta").mode("overwrite")
 .option("replaceWhere", f"batch_id = '{BATCH_ID.replace(chr(39), chr(39)+chr(39))}'")
 .partitionBy("batch_id").saveAsTable(MONITORING_TABLE))
display(monitor_df)
assert monitor_df.first()["records"] == monitor_df.first()["unique_customers"]
assert monitor_df.first()["null_probabilities"] == 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Prepare a reference population for drift checks
# MAGIC
# MAGIC We compare raw incoming customer features with the saved **raw training split** from Notebook 03. The training split is a reference distribution, not necessarily an unbiased representation of future production traffic. The current demonstration input is the held-out test split, so drift values are illustrative.
# MAGIC
# MAGIC We check numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) and categorical columns (`Contract`, `InternetService`, `PaymentMethod`). Blank or malformed `TotalCharges` are converted to null and monitored explicitly.
# MAGIC

# COMMAND ----------

reference_df = spark.table(REFERENCE_TABLE)
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = ["Contract", "InternetService", "PaymentMethod"]
for c in NUMERIC_COLS + CATEGORICAL_COLS:
    assert c in raw_df.columns and c in reference_df.columns, f"Missing drift feature: {c}"
print("Reference records:", reference_df.count())
print("Incoming records:", raw_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Numeric feature drift — Population Stability Index (PSI)
# MAGIC
# MAGIC PSI compares the proportions of observations falling into the **same bins** for reference and incoming data. Here, reference quantiles define bins; duplicate quantiles are removed. Small smoothing avoids division by zero. Heuristic thresholds of 0.10 and 0.25 are often used for triage, but are **not universal statistical significance tests**.
# MAGIC
# MAGIC We collect only aggregated bucket counts and quantiles, not full customer records, to the driver.
# MAGIC

# COMMAND ----------

import math

def numeric_psi(ref_sdf, cur_sdf, feature, bins=10, epsilon=1e-6):
    ref = ref_sdf.select(F.col(feature).cast("double").alias("v")).filter(F.col("v").isNotNull() & ~F.isnan("v"))
    cur = cur_sdf.select(F.col(feature).cast("double").alias("v")).filter(F.col("v").isNotNull() & ~F.isnan("v"))
    nr, nc = ref.count(), cur.count()
    if nr == 0 or nc == 0:
        return None, nr, nc
    cuts = sorted(set(ref.approxQuantile("v", [i/bins for i in range(1,bins)], 0.01)))
    # A constant reference column has no meaningful quantile bins.
    if not cuts:
        return None, nr, nc
    def bucket_expr():
        expr = F.lit(len(cuts))
        for idx in range(len(cuts)-1, -1, -1):
            expr = F.when(F.col("v") <= F.lit(cuts[idx]), F.lit(idx)).otherwise(expr)
        return expr.alias("bucket")
    ref_counts = {int(r["bucket"]): r["count"] for r in ref.select(bucket_expr()).groupBy("bucket").count().collect()}
    cur_counts = {int(r["bucket"]): r["count"] for r in cur.select(bucket_expr()).groupBy("bucket").count().collect()}
    k = len(cuts)+1
    psi = 0.0
    for b in range(k):
        p = (ref_counts.get(b,0) + epsilon) / (nr + epsilon*k)
        q = (cur_counts.get(b,0) + epsilon) / (nc + epsilon*k)
        psi += (q-p)*math.log(q/p)
    return float(psi), nr, nc

numeric_results = []
for col in NUMERIC_COLS:
    score, nr, nc = numeric_psi(reference_df, raw_df, col)
    numeric_results.append((col, "numeric_psi", score, nr, nc))
print("Numeric drift results:")
for r in numeric_results: print(r)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Categorical feature drift — Total Variation Distance (TVD)
# MAGIC
# MAGIC TVD measures how different two categorical distributions are. It ranges from 0 (identical proportions) to 1 (no overlap). We treat missing values as an explicit category. The 0.10 alert threshold below is **an illustrative monitoring policy**, not a validated universal standard.
# MAGIC

# COMMAND ----------

def categorical_tvd(ref_sdf, cur_sdf, feature):
    def proportions(df):
        counts = (df.select(F.coalesce(F.col(feature).cast("string"), F.lit("<MISSING>")).alias("category"))
                  .groupBy("category").count().collect())
        total = sum(r["count"] for r in counts)
        return {r["category"]: r["count"]/total for r in counts} if total else {}
    a, b = proportions(ref_sdf), proportions(cur_sdf)
    if not a or not b: return None
    return 0.5 * sum(abs(a.get(k,0)-b.get(k,0)) for k in set(a)|set(b))

categorical_results = [(c, "categorical_tvd", categorical_tvd(reference_df, raw_df, c),
                        reference_df.count(), raw_df.count()) for c in CATEGORICAL_COLS]
for r in categorical_results: print(r)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Persist drift metrics and classify alerts
# MAGIC
# MAGIC Each drift record has a metric, a score, and a status. PSI: below 0.10 = normal, 0.10–0.25 = review, at least 0.25 = alert. TVD: below 0.10 = normal, at least 0.10 = review. These are **starting policies** to tune using historical batch behavior and business risk.
# MAGIC

# COMMAND ----------

drift_rows = []
for feature, metric, score, ref_n, cur_n in numeric_results + categorical_results:
    if score is None: status = "INSUFFICIENT_DATA"
    elif metric == "numeric_psi": status = "ALERT" if score >= 0.25 else ("REVIEW" if score >= 0.10 else "NORMAL")
    else: status = "REVIEW" if score >= 0.10 else "NORMAL"
    drift_rows.append((BATCH_ID, feature, metric, score, status, int(ref_n), int(cur_n), MODEL_VERSION))

schema = T.StructType([
    T.StructField("batch_id", T.StringType(), False),
    T.StructField("feature", T.StringType(), False),
    T.StructField("metric", T.StringType(), False),
    T.StructField("drift_score", T.DoubleType(), True),
    T.StructField("status", T.StringType(), False),
    T.StructField("reference_non_null_count", T.LongType(), False),
    T.StructField("incoming_non_null_count", T.LongType(), False),
    T.StructField("model_version", T.StringType(), False)
])
drift_df = spark.createDataFrame(drift_rows, schema).withColumn("checked_at", F.current_timestamp())
(drift_df.write.format("delta").mode("overwrite")
 .option("replaceWhere", f"batch_id = '{BATCH_ID.replace(chr(39), chr(39)+chr(39))}'")
 .partitionBy("batch_id").saveAsTable(DRIFT_TABLE))
display(drift_df.orderBy("feature"))
print("Drift features checked:", drift_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Report visuals and monitoring dashboard
# MAGIC
# MAGIC Visualize the latest risk mix and feature drift. Databricks `display()` lets you switch tables to bar charts in the notebook UI. The Python plot below provides a static chart without extra plotting dependencies beyond Matplotlib.
# MAGIC

# COMMAND ----------

display(saved_batch_df.groupBy("risk_level").count().orderBy("risk_level"))
display(drift_df.select("feature", "metric", "drift_score", "status").orderBy("feature"))

import matplotlib.pyplot as plt
risk_counts = {r["risk_level"]: r["count"] for r in saved_batch_df.groupBy("risk_level").count().collect()}
labels = ["Low Risk", "Medium Risk", "High Risk"]
fig, ax = plt.subplots(figsize=(8,4))
ax.bar(labels, [risk_counts.get(x,0) for x in labels])
ax.set_title(f"Customer Churn Risk — {BATCH_ID}")
ax.set_ylabel("Customers")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Final checkpoint and operational alert summary
# MAGIC
# MAGIC The notebook fails on core data-integrity violations but only **reports** drift alerts; drift alone should not automatically trigger retraining. Review the report and confirm data quality, then investigate meaningful distribution shifts.
# MAGIC

# COMMAND ----------

n = saved_batch_df.count()
unique_n = saved_batch_df.select("customerID").distinct().count()
null_n = saved_batch_df.filter(F.col("churn_probability").isNull()).count()
invalid_n = saved_batch_df.filter((F.col("churn_probability") < 0) | (F.col("churn_probability") > 1)).count()
assert n == input_count == unique_n and null_n == 0 and invalid_n == 0
alerts = drift_df.filter(F.col("status").isin("ALERT", "REVIEW", "INSUFFICIENT_DATA"))
print("BATCH VALIDATION PASSED")
print("Batch ID:", BATCH_ID)
print("Scored / unique customers:", n, "/", unique_n)
print("Null / invalid probabilities:", null_n, "/", invalid_n)
print("Drift features needing review:", alerts.count())
display(alerts.orderBy("feature"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Schedule with Databricks Workflows (manual setup)
# MAGIC
# MAGIC This notebook **does not create or activate a scheduled job**. After the manual checkpoints pass:
# MAGIC
# MAGIC 1. In Databricks, open **Workflows → Jobs & Pipelines → Create Job** (wording can vary by workspace).
# MAGIC 2. Add a **Notebook** task pointing to this Notebook 10.
# MAGIC 3. Choose supported compute and grant the job identity permissions to read source tables/model/volume and write the three monitoring tables.
# MAGIC 4. Configure task parameters: `input_table`, `batch_id`, `model_version`.
# MAGIC 5. Set a schedule appropriate for the arrival of **new unlabeled customer batches**.
# MAGIC 6. **Important:** Supply a distinct `batch_id` for each new batch. A static `batch_id` will intentionally replace the prior batch with that ID. Use an upstream orchestration task or job parameterization to generate IDs from the source delivery date/batch identifier.
# MAGIC 7. Configure job failure notifications and review the monitoring tables for drift warnings.
# MAGIC
# MAGIC ### Production hardening checklist
# MAGIC - Make input schema and categorical handling contracts explicit.
# MAGIC - Version preprocessing **together with** the model (ideally register one end-to-end pipeline) to avoid incompatible releases.
# MAGIC - Use source ingestion identifiers and deduplication rules.
# MAGIC - Add lineage, access controls, and retention for customer-level outputs.
# MAGIC - Calibrate and periodically revisit decision thresholds using newly labeled outcomes.
# MAGIC - Add true **performance monitoring** only when delayed actual churn labels become available.
# MAGIC - Tune PSI/TVD thresholds against normal historical variability; investigate changes before retraining.
# MAGIC
# MAGIC ### Expected demonstration checkpoint
# MAGIC With the original Notebook 09 test input: **1,409 scored customers**, **1,409 unique IDs**, **0 null probabilities**, and risk tiers close to **1,009 Low / 311 Medium / 89 High** (assuming the registered model and data are unchanged).
# MAGIC
# MAGIC ### Next notebook
# MAGIC **10A — Batch Scoring & Drift Monitoring Findings Report**, followed by a later notebook for production-grade scheduling, versioned feature pipelines, and delayed-label performance monitoring.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify the persisted batch records - your latest Job run

# COMMAND ----------

from pyspark.sql import functions as F

# Verify the batch created by the successful Job
TEST_BATCH_ID = "telco_788040999362398"

tables = {
    "Prediction history": "workspace.default.telco_10_prediction_history",
    "Batch monitoring": "workspace.default.telco_10_batch_monitoring",
    "Feature drift": "workspace.default.telco_10_feature_drift"
}

for name, table_name in tables.items():
    df = spark.table(table_name).filter(
        F.col("batch_id") == TEST_BATCH_ID
    )

    print(f"{name}: {df.count()} records")

# COMMAND ----------

from pyspark.sql import functions as F

batch_id = "telco_788040999362398"

for table in [
    "telco_10_prediction_history",
    "telco_10_batch_monitoring",
    "telco_10_feature_drift"
]:
    df = spark.table(f"workspace.default.{table}")
    count = df.filter(F.col("batch_id") == batch_id).count()
    print(f"{table}: {count} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify retry idempotency

# COMMAND ----------

from pyspark.sql import functions as F

TEST_BATCH_ID = "telco_788040999362398"

tables = {
    "Prediction history": "workspace.default.telco_10_prediction_history",
    "Batch monitoring": "workspace.default.telco_10_batch_monitoring",
    "Feature drift": "workspace.default.telco_10_feature_drift"
}

expected_counts = {
    "Prediction history": 1409,
    "Batch monitoring": 1,
    "Feature drift": 6
}

for name, table_name in tables.items():
    count = (
        spark.table(table_name)
        .filter(F.col("batch_id") == TEST_BATCH_ID)
        .count()
    )

    status = "PASS" if count == expected_counts[name] else "FAIL"
    print(f"{name}: {count} records — {status}")

# COMMAND ----------

# MAGIC %md
# MAGIC