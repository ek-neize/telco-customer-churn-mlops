# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 09A — Telco Model Deployment and Batch Inference Findings Report
# MAGIC
# MAGIC **Project:** Telco Customer Churn ML / MLOps  
# MAGIC **Status:** Batch inference and Delta-table validation completed  
# MAGIC **Model:** `models:/workspace.default.telco_churn_logistic_regression/1`  
# MAGIC **Output:** `workspace.default.telco_09_batch_predictions`
# MAGIC
# MAGIC This report summarizes the observed results from Notebook 09 and provides reproducible queries and charts. Run code cells in Databricks to refresh the figures from the saved Delta table.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Executive summary
# MAGIC
# MAGIC A registered MLflow model was loaded from Unity Catalog on Databricks, combined with a separately saved fitted preprocessing pipeline, and used to score **1,409** customer records. The scored output was saved to a managed Delta table. Validation confirmed **1,409 unique customers**, **zero null churn probabilities**, and all validation assertions passed.
# MAGIC
# MAGIC The segmentation uses an **operational classification threshold of 0.40**, selected during earlier validation. The high-risk cutoff of **0.70** is an illustrative business rule, not a separately validated threshold. Predicted probabilities represent model estimates, not observed future churn.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Deployment architecture
# MAGIC
# MAGIC ```text
# MAGIC Raw test customers (workspace.default.telco_test_raw)
# MAGIC                   |
# MAGIC                   v
# MAGIC Remove Churn and label from inference input
# MAGIC                   |
# MAGIC                   v
# MAGIC Saved fitted preprocessing PipelineModel (20 stages)
# MAGIC /Volumes/workspace/default/telco_data/telco_preprocessing_pipeline
# MAGIC                   |
# MAGIC                   v
# MAGIC 60-dimensional feature vector
# MAGIC                   |
# MAGIC                   v
# MAGIC MLflow / Unity Catalog registered PipelineModel
# MAGIC models:/workspace.default.telco_churn_logistic_regression/1
# MAGIC (one LogisticRegressionModel stage)
# MAGIC                   |
# MAGIC                   v
# MAGIC Probability of churn + 0.40 decision threshold + risk tier
# MAGIC                   |
# MAGIC                   v
# MAGIC workspace.default.telco_09_batch_predictions
# MAGIC                   |
# MAGIC                   v
# MAGIC Validation / reporting / retention prioritization
# MAGIC ```
# MAGIC
# MAGIC On shared/serverless compute, MLflow Spark model loading required a temporary Unity Catalog Volume path: `/Volumes/workspace/default/telco_data/mlflow_tmp`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Deployment configuration and reproducibility
# MAGIC
# MAGIC | Component | Value |
# MAGIC |---|---|
# MAGIC | Registered model | `workspace.default.telco_churn_logistic_regression` |
# MAGIC | Version | `1` |
# MAGIC | Loaded MLflow flavor | Spark `PipelineModel` |
# MAGIC | Registered model stages | 1 — `LogisticRegressionModel` |
# MAGIC | Separate preprocessing pipeline | 20 fitted stages |
# MAGIC | Input features | 60 |
# MAGIC | Batch source | `workspace.default.telco_test_raw` |
# MAGIC | Delta destination | `workspace.default.telco_09_batch_predictions` |
# MAGIC | Classification threshold | 0.40 |
# MAGIC | Risk tiers | Low: <0.40; Medium: 0.40–<0.70; High: ≥0.70 |
# MAGIC
# MAGIC **Important:** The registered model does not include preprocessing. The separately saved fitted preprocessing pipeline must be loaded and applied before scoring. Do not fit preprocessing again on inference data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verified validation results
# MAGIC
# MAGIC | Validation check | Observed result | Status |
# MAGIC |---|---:|---|
# MAGIC | Saved prediction rows | 1,409 | Passed |
# MAGIC | Unique customer IDs | 1,409 | Passed |
# MAGIC | Null churn probabilities | 0 | Passed |
# MAGIC | Row-count, uniqueness, null, and probability-range assertions | All passed | Passed |
# MAGIC
# MAGIC The Delta output was saved successfully to `workspace.default.telco_09_batch_predictions`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Churn risk distribution
# MAGIC
# MAGIC | Risk tier | Customers | Percentage of batch | Average estimated churn probability |
# MAGIC |---|---:|---:|---:|
# MAGIC | Low Risk | 1,009 | 71.61% | 12.42% |
# MAGIC | Medium Risk | 311 | 22.07% | 56.05% |
# MAGIC | High Risk | 89 | 6.32% | 74.74% |
# MAGIC | **Total** | **1,409** | **100%** | — |
# MAGIC
# MAGIC **Operational interpretation:** 400 customers (28.39%) are classified as potential churners at threshold 0.40. Of these, 89 are in the highest illustrative risk tier.

# COMMAND ----------


from pyspark.sql import functions as F
PREDICTIONS_TABLE = "workspace.default.telco_09_batch_predictions"
report_df = spark.table(PREDICTIONS_TABLE)
risk_summary = (report_df.groupBy("risk_level")
    .agg(F.count("*").alias("customers"),
         F.round(F.avg("churn_probability"), 4).alias("average_churn_probability")))
display(risk_summary.orderBy("risk_level"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Risk distribution visualization
# MAGIC
# MAGIC This chart uses the saved Delta table and can be refreshed after subsequent batch runs.

# COMMAND ----------

import matplotlib.pyplot as plt
risk_order = ["Low Risk", "Medium Risk", "High Risk"]
rows = {r["risk_level"]: r["customers"] for r in risk_summary.collect()}
counts = [rows.get(k, 0) for k in risk_order]
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(risk_order, counts)
ax.set(title="Telco batch inference — customer risk distribution", ylabel="Customers")
for bar, value in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, value, str(value), ha="center", va="bottom")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Top 20 highest-scoring customers
# MAGIC
# MAGIC The following query retrieves the top-ranked customers from the saved table. **Customer identifiers should be handled as confidential operational data in real deployments.**

# COMMAND ----------

display(report_df.select("customerID", "churn_probability", "risk_level", "optimized_prediction", "model_version")
    .orderBy(F.desc("churn_probability")).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Observed top five (Notebook 09 output)
# MAGIC
# MAGIC | Customer ID | Estimated churn probability |
# MAGIC |---|---:|
# MAGIC | `5150-ITWWB` | 84.69% |
# MAGIC | `6630-UJZMY` | 82.30% |
# MAGIC | `4750-ZRXIU` | 81.81% |
# MAGIC | `7851-WZEKY` | 81.65% |
# MAGIC | `0295-PPHDO` | 81.49% |
# MAGIC
# MAGIC These are *estimated* probabilities, not actual future outcomes.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Quality assurance — rerunnable checks

# COMMAND ----------

from pyspark.sql import functions as F
n = report_df.count()
unique_n = report_df.select("customerID").distinct().count()
null_n = report_df.filter(F.col("churn_probability").isNull()).count()
invalid_n = report_df.filter((F.col("churn_probability") < 0) | (F.col("churn_probability") > 1)).count()
assert n == 1409, f"Expected 1409 rows, found {n}"
assert unique_n == n, "Duplicate customer IDs"
assert null_n == 0, "Null probabilities"
assert invalid_n == 0, "Probabilities outside [0, 1]"
print(f"PASS: rows={n}, unique_customers={unique_n}, null_probabilities={null_n}, invalid_probabilities={invalid_n}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Business recommendations
# MAGIC
# MAGIC 1. **Prioritize the 89 high-risk customers** for review and potential retention interventions, considering eligibility and outreach capacity.
# MAGIC 2. **Evaluate the remaining 311 medium-risk customers** using campaign budget, expected customer value, and contact costs.
# MAGIC 3. **Run a controlled retention experiment** to estimate incremental impact; a high churn score does not imply an offer will prevent churn.
# MAGIC 4. **Monitor batch quality** (schema, row counts, nulls, feature distributions, score drift) and periodically evaluate model performance against later observed churn.
# MAGIC 5. **Protect customer data** through Unity Catalog access controls, approved retention policies, and auditability.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Limitations and caveats
# MAGIC
# MAGIC - This is a **batch inference** workflow, not a continuously running serving endpoint.
# MAGIC - The batch uses a held-out raw test dataset to *simulate* new customers. It is not evidence of live production traffic.
# MAGIC - The 0.40 threshold was chosen during earlier validation; the 0.70 high-risk tier is illustrative and should be reviewed against business objectives.
# MAGIC - The fitted preprocessing pipeline was created earlier in the project. Earlier model-comparison validation may not have been fully leakage-free; do not claim independently confirmed generalization from that comparison.
# MAGIC - Probabilities may require calibration before being interpreted as reliable real-world risk estimates.
# MAGIC - No causal impact of retention outreach has been measured.
# MAGIC - The saved table was written with `overwrite` for reproducibility. Production pipelines should use appropriate batch IDs, idempotent writes, lineage, and monitoring.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Conclusion and next steps
# MAGIC
# MAGIC Notebook 09 successfully demonstrated **MLflow Unity Catalog model loading**, **reuse of fitted feature engineering**, **batch inference**, **business-oriented churn risk segmentation**, **Delta table persistence**, and **data quality validation**.
# MAGIC
# MAGIC Suggested next stage: **Notebook 10 — Batch Scoring Automation, Monitoring, and Model Drift**, including scheduled Databricks Jobs, operational checks, data/score drift, and model-version tracking.

# COMMAND ----------

# MAGIC %md
# MAGIC
