# Databricks notebook source
# MAGIC %md
# MAGIC # 09 — Telco Model Deployment and Batch Inference
# MAGIC
# MAGIC **Goal:** Load a Unity Catalog registered MLflow Spark model, transform customer records with the saved preprocessing pipeline, generate churn predictions, and write a Delta batch scoring table.
# MAGIC
# MAGIC **Prerequisites:** Notebook 08 registered a Spark ML Logistic Regression model. The existing fitted preprocessing pipeline is stored in `/Volumes/workspace/default/telco_data/telco_preprocessing_pipeline`. The test dataset is used **only as a demonstration batch**, not as evidence of real deployment. For real inference, replace it with new unlabeled customers.
# MAGIC
# MAGIC **Important:** This notebook uses `models:/catalog.schema.model/version` and requires a registered model name and version copied from Unity Catalog. Do not assume that a model was registered just because a native Spark backup exists.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Verify environment
# MAGIC
# MAGIC Check that MLflow and Spark are available. This notebook is intended to run inside Databricks with access to Unity Catalog.

# COMMAND ----------

import mlflow
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

print("MLflow version:", mlflow.__version__)
print("Spark version:", spark.version)
print("Catalog:", spark.sql("SELECT current_catalog()").first()[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configure the registered model
# MAGIC
# MAGIC Open **Catalog → workspace → default → Models**, select the model registered in Notebook 08, and copy its **full three-part name** and **version number**. Enter those below. The notebook will stop with a clear message until you do.

# COMMAND ----------

# DBTITLE 1,untitled
# # SHOW MODELS is not valid SQL on Unity Catalog; use the Databricks SDK instead.
# from databricks.sdk import WorkspaceClient

# w = WorkspaceClient()
# models = list(w.registered_models.list(catalog_name="workspace", schema_name="default"))
# if not models:
#     print("No registered models in workspace.default")
# for m in models:
#     print(m.full_name)

# COMMAND ----------

REGISTERED_MODEL_NAME = "workspace.default.telco_churn_logistic_regression"  # Example format ONLY: workspace.default.some_model_name
MODEL_VERSION = "1"          # Copy the real version, such as "1"

if not REGISTERED_MODEL_NAME or not MODEL_VERSION:
    raise ValueError("Enter REGISTERED_MODEL_NAME and MODEL_VERSION from Unity Catalog before continuing.")
if len(REGISTERED_MODEL_NAME.split(".")) != 3:
    raise ValueError("Use a full three-part name: catalog.schema.model")

MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_VERSION}"
print("Model URI:", MODEL_URI)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure MLflow Temporary Storage
# MAGIC
# MAGIC Databricks shared and serverless compute requires MLflow to use a Unity Catalog Volume for temporary Spark ML model files.
# MAGIC
# MAGIC We will configure MLflow to use our existing `telco_data` volume.

# COMMAND ----------

import os
import mlflow

# Configure a temporary directory in our Unity Catalog Volume
MLFLOW_TMP_DIR = "/Volumes/workspace/default/telco_data/mlflow_tmp"

# Create the directory if it does not already exist
os.makedirs(MLFLOW_TMP_DIR, exist_ok=True)

# Tell MLflow to use this directory
os.environ["MLFLOW_DFS_TMP"] = MLFLOW_TMP_DIR

# Use the Unity Catalog model registry
mlflow.set_registry_uri("databricks-uc")

print("MLflow temporary directory:", MLFLOW_TMP_DIR)
print("MLflow registry:", mlflow.get_registry_uri())

# COMMAND ----------

# Our registered model from Notebook 08
MODEL_URI = "models:/workspace.default.telco_churn_logistic_regression/1"

# Load the model using the Unity Catalog Volume
registered_model = mlflow.spark.load_model(
    model_uri=MODEL_URI,
    dfs_tmpdir=MLFLOW_TMP_DIR
)

print("Registered model loaded successfully!")
print("Model type:", type(registered_model).__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Check the registered model version
# MAGIC
# MAGIC Use the MLflow client to confirm that the requested version exists and display its source run. This avoids silently loading the wrong model.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()
version_info = client.get_model_version(REGISTERED_MODEL_NAME, MODEL_VERSION)
print("Registered model:", version_info.name)
print("Version:", version_info.version)
print("Source run ID:", version_info.run_id)
print("Status:", version_info.status)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect the Registered MLflow Pipeline
# MAGIC
# MAGIC Before performing batch inference, we will inspect the stages of our
# MAGIC registered model.
# MAGIC
# MAGIC This helps us determine whether preprocessing is already included
# MAGIC or whether we need to apply our saved preprocessing pipeline separately.

# COMMAND ----------

# Inspect the registered PipelineModel
print("Registered model stages:", len(registered_model.stages))

for i, stage in enumerate(registered_model.stages):
    print(i, type(stage).__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Load the Saved Preprocessing Pipeline
# MAGIC
# MAGIC Our registered model contains only Logistic Regression.
# MAGIC
# MAGIC We must load the fitted preprocessing pipeline from Notebook 03
# MAGIC to transform raw customer records into the 60 numerical features
# MAGIC expected by the trained model.
# MAGIC
# MAGIC We will use `PipelineModel.load()`, not fit a new pipeline.

# COMMAND ----------

from pyspark.ml import PipelineModel

# Saved preprocessing pipeline from Notebook 03
PREPROCESSING_PATH = (
    "/Volumes/workspace/default/telco_data/"
    "telco_preprocessing_pipeline"
)

# Load the already-fitted preprocessing pipeline
preprocessing_model = PipelineModel.load(
    PREPROCESSING_PATH
)

print("Preprocessing pipeline loaded successfully!")
print("Number of stages:", len(preprocessing_model.stages))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Load Raw Customer Data
# MAGIC
# MAGIC We will use the existing raw test dataset to simulate a batch
# MAGIC of customers requiring churn predictions.
# MAGIC
# MAGIC Although the dataset contains the actual churn outcome,
# MAGIC we will exclude that label from the model input.
# MAGIC
# MAGIC This allows us to simulate inference on unlabeled customers.

# COMMAND ----------

# Load raw customer records
batch_df = spark.table(
    "workspace.default.telco_test_raw"
)

print("Batch customer count:", batch_df.count())

# Inspect the schema
batch_df.printSchema()

# Show a sample
display(batch_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Apply the Saved Preprocessing Pipeline
# MAGIC
# MAGIC The preprocessing pipeline converts raw customer attributes
# MAGIC into numerical features suitable for Logistic Regression.
# MAGIC
# MAGIC The fitted transformations are reused exactly as saved
# MAGIC during model development.

# COMMAND ----------

# Remove the target label for inference
inference_df = batch_df.drop("Churn", "label")

# Apply the fitted preprocessing pipeline
prepared_df = preprocessing_model.transform(inference_df)

print("Preprocessing completed!")

# Inspect the generated feature columns
print("Feature columns:")
print(prepared_df.columns)

# Show a sample of feature vectors
display(
    prepared_df.select("features").limit(5)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Generate Batch Churn Predictions
# MAGIC
# MAGIC We will use our registered MLflow Logistic Regression model
# MAGIC to predict whether customers are likely to churn.
# MAGIC
# MAGIC The model generates:
# MAGIC - prediction: 0 = No Churn, 1 = Churn
# MAGIC - probability: probability of each class
# MAGIC - rawPrediction: raw model scores
# MAGIC
# MAGIC We will retain customerID to identify each customer.

# COMMAND ----------

# Apply the registered Logistic Regression PipelineModel
predictions_df = registered_model.transform(prepared_df)

# Inspect prediction results
display(
    predictions_df.select(
        "customerID",
        "prediction",
        "probability"
    ).limit(10)
)

print("Batch prediction completed!")
print("Customers scored:", predictions_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Calculate Churn Probability and Risk Levels
# MAGIC
# MAGIC We will extract the probability that each customer will churn.
# MAGIC
# MAGIC For this project, we will classify customers into three
# MAGIC illustrative risk categories:
# MAGIC
# MAGIC - Low Risk: churn probability below 0.40
# MAGIC - Medium Risk: churn probability from 0.40 to below 0.70
# MAGIC - High Risk: churn probability 0.70 or higher
# MAGIC
# MAGIC We will also apply our validation-selected classification
# MAGIC threshold of 0.40 from Notebook 06.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

# Extract probability of churn (class 1)
scored_df = predictions_df.withColumn(
    "churn_probability",
    vector_to_array(F.col("probability"))[1]
)

# Apply optimized threshold from Notebook 06
scored_df = scored_df.withColumn(
    "optimized_prediction",
    F.when(F.col("churn_probability") >= 0.40, 1).otherwise(0)
)

# Assign risk categories
scored_df = scored_df.withColumn(
    "risk_level",
    F.when(F.col("churn_probability") >= 0.70, "High Risk")
     .when(F.col("churn_probability") >= 0.40, "Medium Risk")
     .otherwise("Low Risk")
)

# Display customer predictions
display(
    scored_df.select(
        "customerID",
        "churn_probability",
        "prediction",
        "optimized_prediction",
        "risk_level"
    ).limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Analyze Customer Churn Risk
# MAGIC
# MAGIC We will summarize how many customers fall into each
# MAGIC risk category.
# MAGIC
# MAGIC This information could help a telecom company prioritize
# MAGIC customer retention campaigns.

# COMMAND ----------

# Calculate number of customers by risk category
risk_summary_df = (
    scored_df
    .groupBy("risk_level")
    .agg(
        F.count("*").alias("customer_count"),
        F.round(
            F.avg("churn_probability"), 4
        ).alias("average_churn_probability")
    )
    .orderBy("risk_level")
)

display(risk_summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Save Batch Predictions to a Delta Table
# MAGIC
# MAGIC We will save customer churn predictions to a managed Delta table
# MAGIC in Unity Catalog.
# MAGIC
# MAGIC The output includes customer identifiers, predicted churn
# MAGIC probabilities, classification results, risk categories,
# MAGIC model version, and prediction timestamp.

# COMMAND ----------

from pyspark.sql import functions as F

# Destination Delta table
PREDICTIONS_TABLE = "workspace.default.telco_09_batch_predictions"

# Select business-ready prediction columns
batch_output_df = scored_df.select(
    "customerID",
    "churn_probability",
    "prediction",
    "optimized_prediction",
    "risk_level"
).withColumn(
    "model_name",
    F.lit("workspace.default.telco_churn_logistic_regression")
).withColumn(
    "model_version",
    F.lit("1")
).withColumn(
    "prediction_timestamp",
    F.current_timestamp()
)

# Save predictions as a managed Delta table
(
    batch_output_df.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(PREDICTIONS_TABLE)
)

print("Batch predictions saved successfully!")
print("Destination:", PREDICTIONS_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Validate Saved Batch Predictions
# MAGIC
# MAGIC We will verify that the Delta table contains all scored customers,
# MAGIC check for missing predictions, and confirm the distribution of
# MAGIC risk categories.

# COMMAND ----------

# Read the saved Delta table
saved_df = spark.table(PREDICTIONS_TABLE)

# Check total rows
saved_count = saved_df.count()

# Check for missing probabilities
null_count = saved_df.filter(
    F.col("churn_probability").isNull()
).count()

# Check unique customers
unique_customers = saved_df.select(
    "customerID"
).distinct().count()

print("Saved prediction records:", saved_count)
print("Unique customers:", unique_customers)
print("Null churn probabilities:", null_count)

# Validation checks
assert saved_count == scored_df.count(), "Record count mismatch!"
assert null_count == 0, "Missing churn probabilities!"
assert unique_customers == saved_count, "Duplicate customer IDs!"
assert saved_df.filter(
    (F.col("churn_probability") < 0) |
    (F.col("churn_probability") > 1)
).count() == 0, "Invalid probabilities!"

print("All batch prediction validation checks passed!")

# Display saved risk distribution
display(
    saved_df.groupBy("risk_level")
    .count()
    .orderBy("risk_level")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Query the highest-risk customers
# MAGIC Once validation passes, run this SQL cell to see the customers with the highest predicted churn probabilities.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     customerID,
# MAGIC     ROUND(churn_probability, 4) AS churn_probability,
# MAGIC     risk_level,
# MAGIC     optimized_prediction,
# MAGIC     model_version
# MAGIC FROM workspace.default.telco_09_batch_predictions
# MAGIC WHERE risk_level = 'High Risk'
# MAGIC ORDER BY churn_probability DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %md
# MAGIC