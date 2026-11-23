# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 15 — Telco Production Model Deployment
# MAGIC
# MAGIC ## Project: Telco Customer Churn Prediction
# MAGIC
# MAGIC ### Objective
# MAGIC
# MAGIC Develop a reproducible, production-style batch inference workflow
# MAGIC using a registered MLflow Logistic Regression model.
# MAGIC
# MAGIC ### Model Assets
# MAGIC
# MAGIC - Registered model:
# MAGIC   `workspace.default.telco_churn_logistic_regression`
# MAGIC - Model version: `1`
# MAGIC - Model registry: Databricks Unity Catalog
# MAGIC - Algorithm: Logistic Regression
# MAGIC - Preprocessing: Saved Spark ML PipelineModel
# MAGIC - Prediction target: Customer churn (Yes / No)
# MAGIC
# MAGIC ### Deployment Architecture
# MAGIC
# MAGIC 1. Load customer data.
# MAGIC 2. Validate input schema and data quality.
# MAGIC 3. Apply the saved preprocessing pipeline.
# MAGIC 4. Load the registered MLflow model.
# MAGIC 5. Generate churn predictions and probabilities.
# MAGIC 6. Validate prediction outputs.
# MAGIC 7. Persist results to a Delta table.
# MAGIC 8. Prepare monitoring and operational safeguards.
# MAGIC
# MAGIC ### Production Considerations
# MAGIC
# MAGIC - Reproducibility and model versioning
# MAGIC - Schema validation
# MAGIC - Data quality checks
# MAGIC - Idempotent batch processing
# MAGIC - Prediction traceability
# MAGIC - Error handling and logging
# MAGIC - Model monitoring
# MAGIC - Access controls and governance
# MAGIC
# MAGIC ### Important Scope
# MAGIC
# MAGIC This notebook implements a production-style batch inference workflow.
# MAGIC
# MAGIC It does not, by itself, establish that a production serving
# MAGIC endpoint, automated Databricks Job, or operational monitoring
# MAGIC service has been deployed.
# MAGIC

# COMMAND ----------


# Step 15.3: Configure production inference environment

import os
import mlflow
import mlflow.spark

from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

print("=" * 65)
print("PRODUCTION INFERENCE ENVIRONMENT CONFIGURATION")
print("=" * 65)

# --------------------------------------------------
# 1. Unity Catalog Model Registry
# --------------------------------------------------

mlflow.set_registry_uri("databricks-uc")

MODEL_NAME = (
    "workspace.default.telco_churn_logistic_regression"
)

MODEL_VERSION = "1"

MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

# --------------------------------------------------
# 2. Saved preprocessing pipeline
# --------------------------------------------------

PREPROCESSING_PIPELINE_PATH = (
    "/Volumes/workspace/default/telco_data/"
    "telco_preprocessing_pipeline"
)

# --------------------------------------------------
# 3. MLflow temporary storage
# --------------------------------------------------

MLFLOW_TMP_PATH = (
    "/Volumes/workspace/default/telco_data/"
    "mlflow_spark_tmp"
)

os.environ["MLFLOW_DFS_TMP"] = MLFLOW_TMP_PATH

# --------------------------------------------------
# 4. Output configuration
# --------------------------------------------------

OUTPUT_TABLE = (
    "workspace.default.telco_15_production_predictions"
)

# --------------------------------------------------
# 5. Verify Spark environment
# --------------------------------------------------

spark_version = spark.version

current_catalog = spark.sql(
    "SELECT current_catalog()"
).first()[0]

current_schema = spark.sql(
    "SELECT current_schema()"
).first()[0]

# --------------------------------------------------
# 6. Display configuration
# --------------------------------------------------

print("\nMODEL CONFIGURATION")
print("Registered model:", MODEL_NAME)
print("Model version:", MODEL_VERSION)
print("Model URI:", MODEL_URI)

print("\nSTORAGE CONFIGURATION")
print("Preprocessing pipeline:", PREPROCESSING_PIPELINE_PATH)
print("MLflow temporary path:", MLFLOW_TMP_PATH)
print("Prediction output table:", OUTPUT_TABLE)

print("\nDATABRICKS ENVIRONMENT")
print("Spark version:", spark_version)
print("Current catalog:", current_catalog)
print("Current schema:", current_schema)

# --------------------------------------------------
# 7. Validate configuration
# --------------------------------------------------

assert MODEL_VERSION == "1"
assert MODEL_URI.startswith("models:/")
assert PREPROCESSING_PIPELINE_PATH.startswith("/Volumes/")
assert OUTPUT_TABLE.count(".") == 2

print("\nENVIRONMENT CONFIGURATION: PASS")


# COMMAND ----------


# Step 15.4: Load and validate production model assets

import mlflow.spark
from pyspark.ml import PipelineModel

print("=" * 65)
print("PRODUCTION MODEL ASSET LOADING")
print("=" * 65)

# --------------------------------------------------
# 1. Load registered MLflow model
# --------------------------------------------------

print("\nLoading registered MLflow model...")

production_model = mlflow.spark.load_model(
    MODEL_URI,
    dfs_tmpdir=MLFLOW_TMP_PATH
)

print("Registered model loaded successfully.")

# --------------------------------------------------
# 2. Load saved preprocessing pipeline
# --------------------------------------------------

print("\nLoading preprocessing pipeline...")

production_preprocessing_pipeline = PipelineModel.load(
    PREPROCESSING_PIPELINE_PATH
)

print("Preprocessing pipeline loaded successfully.")

# --------------------------------------------------
# 3. Inspect model structure
# --------------------------------------------------

model_stage_count = len(production_model.stages)

preprocessing_stage_count = len(
    production_preprocessing_pipeline.stages
)

print("\nMODEL STRUCTURE")
print("Registered model stages:", model_stage_count)
print("Preprocessing stages:", preprocessing_stage_count)

# --------------------------------------------------
# 4. Extract Logistic Regression model
# --------------------------------------------------

production_lr_model = production_model.stages[0]

print("\nLOGISTIC REGRESSION DETAILS")
print("Model type:", type(production_lr_model).__name__)
print(
    "Number of coefficients:",
    len(production_lr_model.coefficients)
)
print(
    "Number of classes:",
    production_lr_model.numClasses
)
print(
    "Feature column:",
    production_lr_model.getFeaturesCol()
)
print(
    "Probability column:",
    production_lr_model.getProbabilityCol()
)

# --------------------------------------------------
# 5. Validate model compatibility
# --------------------------------------------------

assert model_stage_count == 1, (
    "Unexpected registered model structure."
)

assert preprocessing_stage_count == 20, (
    "Unexpected preprocessing pipeline structure."
)

assert len(production_lr_model.coefficients) == 60, (
    "Expected 60 model coefficients."
)

assert production_lr_model.numClasses == 2, (
    "Expected binary classification model."
)

assert production_lr_model.getFeaturesCol() == "features"

print("\nPRODUCTION MODEL ASSET VALIDATION: PASS")


# COMMAND ----------


# Step 15.5: Validate production input schema

from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

print("=" * 65)
print("PRODUCTION INPUT SCHEMA VALIDATION")
print("=" * 65)

# --------------------------------------------------
# 1. Load historical data to simulate an incoming batch
# --------------------------------------------------

SOURCE_TABLE = "workspace.default.telco_test_raw"

raw_batch_df = spark.table(SOURCE_TABLE)

print("\nSource table:", SOURCE_TABLE)

# --------------------------------------------------
# 2. Define required input columns
# --------------------------------------------------

categorical_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

numeric_columns = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen"
]

required_columns = (
    ["customerID"]
    + categorical_columns
    + numeric_columns
)

# --------------------------------------------------
# 3. Validate required columns
# --------------------------------------------------

missing_columns = [
    col for col in required_columns
    if col not in raw_batch_df.columns
]

print("\nRequired columns:", len(required_columns))
print("Missing columns:", missing_columns)

assert not missing_columns, (
    f"Missing required columns: {missing_columns}"
)

# --------------------------------------------------
# 4. Remove target labels from inference input
# --------------------------------------------------

inference_input_df = raw_batch_df.drop(
    "Churn",
    "label"
)

print(
    "Target columns removed:",
    "Churn" not in inference_input_df.columns
    and "label" not in inference_input_df.columns
)

# --------------------------------------------------
# 5. Validate numeric column types
# --------------------------------------------------

invalid_numeric_types = []

for col_name in numeric_columns:
    data_type = inference_input_df.schema[
        col_name
    ].dataType

    if not isinstance(data_type, NumericType):
        invalid_numeric_types.append(
            (col_name, str(data_type))
        )

print("Invalid numeric types:", invalid_numeric_types)

assert not invalid_numeric_types, (
    f"Invalid numeric types: {invalid_numeric_types}"
)

# --------------------------------------------------
# 6. Validate customer identifiers
# --------------------------------------------------

total_rows = inference_input_df.count()

null_customer_ids = inference_input_df.filter(
    F.col("customerID").isNull()
    | (F.trim(F.col("customerID")) == "")
).count()

unique_customer_ids = (
    inference_input_df
    .select("customerID")
    .distinct()
    .count()
)

duplicate_customer_ids = (
    total_rows - unique_customer_ids
)

print("\nCUSTOMER VALIDATION")
print("Total customers:", total_rows)
print("Unique customer IDs:", unique_customer_ids)
print("Null or blank customer IDs:", null_customer_ids)
print("Duplicate customer IDs:", duplicate_customer_ids)

# --------------------------------------------------
# 7. Final validation
# --------------------------------------------------

assert total_rows > 0, "Input dataset is empty."
assert null_customer_ids == 0, "Invalid customer IDs found."
assert duplicate_customer_ids == 0, "Duplicate IDs found."

assert "Churn" not in inference_input_df.columns
assert "label" not in inference_input_df.columns

print("\nPRODUCTION INPUT SCHEMA VALIDATION: PASS")


# COMMAND ----------


# Step 15.6: Apply production preprocessing pipeline

from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

print("=" * 65)
print("PRODUCTION PREPROCESSING PIPELINE VALIDATION")
print("=" * 65)

# --------------------------------------------------
# 1. Apply saved preprocessing pipeline
# --------------------------------------------------

print("\nApplying saved preprocessing pipeline...")

production_features_df = (
    production_preprocessing_pipeline.transform(
        inference_input_df
    )
)

print("Preprocessing transformation completed.")

# --------------------------------------------------
# 2. Validate transformed customer count
# --------------------------------------------------

input_count = inference_input_df.count()
transformed_count = production_features_df.count()

print("\nCUSTOMER COUNT VALIDATION")
print("Input customers:", input_count)
print("Transformed customers:", transformed_count)

assert transformed_count == input_count, (
    "Customer count changed during preprocessing."
)

# --------------------------------------------------
# 3. Validate feature-vector dimensions
# --------------------------------------------------

feature_validation_df = (
    production_features_df
    .select(
        "customerID",
        F.size(
            vector_to_array(F.col("features"))
        ).alias("feature_count")
    )
)

feature_count_summary = (
    feature_validation_df
    .groupBy("feature_count")
    .count()
    .orderBy("feature_count")
)

print("\nFEATURE VECTOR DIMENSION SUMMARY")
feature_count_summary.show()

# Validate against registered model
expected_features = len(
    production_lr_model.coefficients
)

invalid_feature_count = (
    feature_validation_df
    .filter(
        F.col("feature_count") != expected_features
    )
    .count()
)

assert invalid_feature_count == 0, (
    "Feature vector dimension mismatch."
)

# --------------------------------------------------
# 4. Validate feature values
# --------------------------------------------------

feature_array_df = (
    production_features_df
    .select(
        "customerID",
        vector_to_array(
            F.col("features")
        ).alias("feature_array")
    )
)

invalid_feature_rows = (
    feature_array_df
    .filter(
        F.exists(
            F.col("feature_array"),
            lambda x: (
                x.isNull()
                | F.isnan(x)
                | (F.abs(x) == float("inf"))
            )
        )
    )
    .count()
)

print("\nFEATURE QUALITY VALIDATION")
print("Expected feature count:", expected_features)
print("Invalid feature dimensions:", invalid_feature_count)
print("Rows containing NaN, null or infinity:", invalid_feature_rows)

assert invalid_feature_rows == 0, (
    "Invalid numerical values found in feature vectors."
)

# --------------------------------------------------
# 5. Final validation
# --------------------------------------------------

assert transformed_count == 1409
assert expected_features == 60

print("\nPRODUCTION PREPROCESSING VALIDATION: PASS")


# COMMAND ----------


# Step 15.7: Generate production-style churn predictions

from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

print("=" * 65)
print("PRODUCTION CHURN PREDICTION GENERATION")
print("=" * 65)

# --------------------------------------------------
# 1. Generate predictions using registered model
# --------------------------------------------------

print("\nGenerating churn predictions...")

production_scored_df = (
    production_model.transform(
        production_features_df
    )
)

print("Model inference completed.")

# --------------------------------------------------
# 2. Extract probability of churn (class 1)
# --------------------------------------------------

production_predictions_df = (
    production_scored_df
    .select(
        F.col("customerID"),

        vector_to_array(
            F.col("probability")
        )[1].alias("churn_probability"),

        F.col("prediction")
        .cast("int")
        .alias("predicted_class")
    )
    .withColumn(
        "predicted_churn",
        F.when(
            F.col("predicted_class") == 1,
            F.lit("Yes")
        ).otherwise(F.lit("No"))
    )
    .withColumn(
        "model_name",
        F.lit(MODEL_NAME)
    )
    .withColumn(
        "model_version",
        F.lit(MODEL_VERSION)
    )
)

# --------------------------------------------------
# 3. Validate prediction counts
# --------------------------------------------------

prediction_count = production_predictions_df.count()

unique_customers = (
    production_predictions_df
    .select("customerID")
    .distinct()
    .count()
)

print("\nPREDICTION COUNT VALIDATION")
print("Input customers:", input_count)
print("Prediction records:", prediction_count)
print("Unique customers:", unique_customers)

assert prediction_count == input_count
assert unique_customers == input_count

# --------------------------------------------------
# 4. Validate probabilities and predictions
# --------------------------------------------------

invalid_predictions = (
    production_predictions_df
    .filter(
        F.col("churn_probability").isNull()
        | F.isnan("churn_probability")
        | (F.col("churn_probability") < 0)
        | (F.col("churn_probability") > 1)
        | F.col("predicted_class").isNull()
        | (~F.col("predicted_class").isin(0, 1))
    )
    .count()
)

assert invalid_predictions == 0

# --------------------------------------------------
# 5. Display prediction summary
# --------------------------------------------------

print("\nPREDICTION SUMMARY")

production_predictions_df.agg(
    F.count("*").alias("customers"),
    F.sum("predicted_class").alias("predicted_churners"),
    F.avg("churn_probability").alias("mean_churn_probability"),
    F.min("churn_probability").alias("min_probability"),
    F.max("churn_probability").alias("max_probability")
).show(truncate=False)

print("\nPREDICTED CHURN DISTRIBUTION")

production_predictions_df.groupBy(
    "predicted_churn"
).count().orderBy("predicted_churn").show()

# --------------------------------------------------
# 6. Display sample predictions
# --------------------------------------------------

print("\nSAMPLE CUSTOMER PREDICTIONS")

production_predictions_df.orderBy(
    "customerID"
).show(10, truncate=False)

# --------------------------------------------------
# 7. Final validation
# --------------------------------------------------

print("\nInvalid predictions:", invalid_predictions)

print("\nPRODUCTION CHURN PREDICTION GENERATION: PASS")



# COMMAND ----------


# Step 15.8: Add prediction traceability and batch metadata

from datetime import datetime, timezone
from uuid import uuid4
from pyspark.sql import functions as F

print("=" * 65)
print("PRODUCTION PREDICTION TRACEABILITY")
print("=" * 65)

# --------------------------------------------------
# 1. Generate one identifier per inference batch
# --------------------------------------------------

BATCH_ID = str(uuid4())

BATCH_TIMESTAMP = datetime.now(timezone.utc)

CLASSIFICATION_THRESHOLD = 0.50

print("\nBATCH METADATA")
print("Batch ID:", BATCH_ID)
print("Batch timestamp (UTC):", BATCH_TIMESTAMP)
print("Classification threshold:", CLASSIFICATION_THRESHOLD)

# --------------------------------------------------
# 2. Add metadata to predictions
# --------------------------------------------------

production_predictions_with_metadata_df = (
    production_predictions_df
    .withColumn(
        "batch_id",
        F.lit(BATCH_ID)
    )
    .withColumn(
        "prediction_timestamp",
        F.lit(BATCH_TIMESTAMP).cast("timestamp")
    )
    .withColumn(
        "classification_threshold",
        F.lit(CLASSIFICATION_THRESHOLD)
    )
    .withColumn(
        "source_table",
        F.lit(SOURCE_TABLE)
    )
)

# --------------------------------------------------
# 3. Validate metadata completeness
# --------------------------------------------------

required_metadata_columns = [
    "batch_id",
    "prediction_timestamp",
    "model_name",
    "model_version",
    "classification_threshold",
    "source_table"
]

missing_metadata_columns = [
    col for col in required_metadata_columns
    if col not in production_predictions_with_metadata_df.columns
]

assert not missing_metadata_columns, (
    f"Missing metadata columns: {missing_metadata_columns}"
)

null_metadata_condition = None

for col_name in required_metadata_columns:
    condition = F.col(col_name).isNull()

    null_metadata_condition = (
        condition
        if null_metadata_condition is None
        else null_metadata_condition | condition
    )

null_metadata_rows = (
    production_predictions_with_metadata_df
    .filter(null_metadata_condition)
    .count()
)

assert null_metadata_rows == 0

# --------------------------------------------------
# 4. Validate batch consistency
# --------------------------------------------------

batch_count = (
    production_predictions_with_metadata_df
    .select("batch_id")
    .distinct()
    .count()
)

timestamp_count = (
    production_predictions_with_metadata_df
    .select("prediction_timestamp")
    .distinct()
    .count()
)

record_count = (
    production_predictions_with_metadata_df
    .count()
)

assert batch_count == 1
assert timestamp_count == 1
assert record_count == input_count

# Confirm recorded threshold agrees with model decisions
threshold_mismatches = (
    production_predictions_with_metadata_df
    .filter(
        F.col("predicted_class") !=
        F.when(
            F.col("churn_probability") >=
            F.col("classification_threshold"),
            F.lit(1)
        ).otherwise(F.lit(0))
    )
    .count()
)

assert threshold_mismatches == 0

# --------------------------------------------------
# 5. Display traceable prediction records
# --------------------------------------------------

print("\nTRACEABILITY VALIDATION")
print("Prediction records:", record_count)
print("Distinct batch IDs:", batch_count)
print("Distinct batch timestamps:", timestamp_count)
print("Missing metadata values:", null_metadata_rows)
print("Threshold mismatches:", threshold_mismatches)

print("\nSAMPLE TRACEABLE PREDICTIONS")

production_predictions_with_metadata_df.select(
    "customerID",
    "churn_probability",
    "predicted_churn",
    "model_version",
    "batch_id",
    "prediction_timestamp",
    "classification_threshold"
).orderBy("customerID").show(
    5,
    truncate=False
)

print("\nPRODUCTION PREDICTION TRACEABILITY: PASS")


# COMMAND ----------

# DBTITLE 1,Step 15.9: Persist production predictions to Delta

# Step 15.9: Persist production predictions to Delta

from delta.tables import DeltaTable
from pyspark.sql import functions as F

print("=" * 65)
print("PRODUCTION PREDICTION DELTA PERSISTENCE")
print("=" * 65)

print("\nDestination table:", OUTPUT_TABLE)
print("Current batch ID:", BATCH_ID)

# --------------------------------------------------
# 1. Validate batch records before writing
# --------------------------------------------------

batch_predictions_df = (
    production_predictions_with_metadata_df
)

batch_record_count = batch_predictions_df.count()

batch_unique_keys = (
    batch_predictions_df
    .select("batch_id", "customerID")
    .distinct()
    .count()
)

assert batch_record_count == batch_unique_keys, (
    "Duplicate batch/customer keys detected."
)

assert batch_record_count == input_count

print("\nBatch records:", batch_record_count)
print("Unique batch/customer keys:", batch_unique_keys)

# --------------------------------------------------
# 2. Create Delta table if it does not exist
# --------------------------------------------------

table_exists = spark.catalog.tableExists(
    OUTPUT_TABLE
)

if not table_exists:

    print("\nCreating Delta prediction table...")

    (
        batch_predictions_df
        .limit(0)
        .write
        .format("delta")
        .mode("error")
        .saveAsTable(OUTPUT_TABLE)
    )

    print("Delta prediction table created.")

else:
    print("\nDelta prediction table already exists.")

# --------------------------------------------------
# 3. Merge predictions using batch/customer keys
# --------------------------------------------------

print("\nMerging prediction records...")

target_table = DeltaTable.forName(
    spark,
    OUTPUT_TABLE
)

(
    target_table.alias("target")
    .merge(
        batch_predictions_df.alias("source"),
        """
        target.batch_id = source.batch_id
        AND target.customerID = source.customerID
        """
    )
    .whenNotMatchedInsertAll()
    .execute()
)

print("Delta MERGE completed.")

# --------------------------------------------------
# 4. Validate persisted batch
# --------------------------------------------------

saved_batch_df = (
    spark.table(OUTPUT_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

saved_record_count = saved_batch_df.count()

saved_unique_customers = (
    saved_batch_df
    .select("customerID")
    .distinct()
    .count()
)

print("\nDELTA PERSISTENCE VALIDATION")
print("Expected records:", batch_record_count)
print("Saved records:", saved_record_count)
print("Unique saved customers:", saved_unique_customers)

assert saved_record_count == batch_record_count
assert saved_unique_customers == batch_record_count

# --------------------------------------------------
# 5. Display saved prediction summary
# --------------------------------------------------

print("\nSAVED BATCH SUMMARY")

saved_batch_df.groupBy(
    "predicted_churn"
).count().orderBy(
    "predicted_churn"
).show()

print("\nSAMPLE SAVED RECORDS")

saved_batch_df.select(
    "customerID",
    "churn_probability",
    "predicted_churn",
    "model_version",
    "batch_id"
).orderBy("customerID").show(
    5,
    truncate=False
)

print("\nPRODUCTION DELTA PERSISTENCE: PASS")


# COMMAND ----------


# Step 15.10: Verify Delta table integrity and history

from pyspark.sql import functions as F

print("=" * 65)
print("PRODUCTION DELTA TABLE INTEGRITY AUDIT")
print("=" * 65)

# --------------------------------------------------
# 1. Read the persisted Delta table
# --------------------------------------------------

persisted_predictions_df = spark.table(OUTPUT_TABLE)

print("\nTABLE INFORMATION")
print("Table:", OUTPUT_TABLE)

table_details_df = spark.sql(
    f"DESCRIBE DETAIL {OUTPUT_TABLE}"
)

table_details_df.select(
    "format",
    "location",
    "numFiles"
).show(truncate=False)

table_format = (
    table_details_df
    .select("format")
    .first()[0]
)

assert table_format.lower() == "delta"

# --------------------------------------------------
# 2. Validate table-wide record uniqueness
# --------------------------------------------------

total_saved_records = persisted_predictions_df.count()

unique_batch_customer_keys = (
    persisted_predictions_df
    .select("batch_id", "customerID")
    .distinct()
    .count()
)

duplicate_records = (
    total_saved_records - unique_batch_customer_keys
)

print("\nTABLE INTEGRITY")
print("Total stored records:", total_saved_records)
print("Unique batch/customer keys:", unique_batch_customer_keys)
print("Duplicate records:", duplicate_records)

assert duplicate_records == 0

# --------------------------------------------------
# 3. Validate stored prediction values
# --------------------------------------------------

invalid_probabilities = (
    persisted_predictions_df
    .filter(
        F.col("churn_probability").isNull()
        | F.isnan("churn_probability")
        | (F.col("churn_probability") < 0)
        | (F.col("churn_probability") > 1)
    )
    .count()
)

invalid_classes = (
    persisted_predictions_df
    .filter(
        F.col("predicted_class").isNull()
        | (~F.col("predicted_class").isin(0, 1))
    )
    .count()
)

missing_metadata = (
    persisted_predictions_df
    .filter(
        F.col("batch_id").isNull()
        | F.col("prediction_timestamp").isNull()
        | F.col("model_name").isNull()
        | F.col("model_version").isNull()
        | F.col("classification_threshold").isNull()
        | F.col("source_table").isNull()
    )
    .count()
)

assert invalid_probabilities == 0
assert invalid_classes == 0
assert missing_metadata == 0

print("\nPREDICTION QUALITY")
print("Invalid probabilities:", invalid_probabilities)
print("Invalid classes:", invalid_classes)
print("Records with missing metadata:", missing_metadata)

# --------------------------------------------------
# 4. Verify current batch
# --------------------------------------------------

current_batch_df = (
    persisted_predictions_df
    .filter(F.col("batch_id") == BATCH_ID)
)

current_batch_count = current_batch_df.count()

assert current_batch_count == batch_record_count

print("\nCURRENT BATCH VALIDATION")
print("Batch ID:", BATCH_ID)
print("Expected records:", batch_record_count)
print("Stored records:", current_batch_count)

# --------------------------------------------------
# 5. Summarize stored prediction batches
# --------------------------------------------------

print("\nPREDICTION BATCH HISTORY")

persisted_predictions_df.groupBy(
    "batch_id",
    "model_version"
).agg(
    F.count("*").alias("prediction_count"),
    F.sum("predicted_class").alias("predicted_churners"),
    F.min("prediction_timestamp").alias("batch_timestamp")
).orderBy(
    F.col("batch_timestamp").desc()
).show(truncate=False)

# --------------------------------------------------
# 6. Inspect Delta transaction history
# --------------------------------------------------

print("\nDELTA TRANSACTION HISTORY")

spark.sql(
    f"DESCRIBE HISTORY {OUTPUT_TABLE}"
).select(
    "version",
    "timestamp",
    "operation"
).orderBy(
    F.col("version").desc()
).show(10, truncate=False)

# --------------------------------------------------
# 7. Final validation
# --------------------------------------------------

print("\nPRODUCTION DELTA TABLE INTEGRITY: PASS")


# COMMAND ----------


# Step 15.11: Validate idempotent batch processing

from delta.tables import DeltaTable
from pyspark.sql import functions as F

print("=" * 65)
print("PRODUCTION BATCH IDEMPOTENCY VALIDATION")
print("=" * 65)

# --------------------------------------------------
# 1. Capture table state before retry
# --------------------------------------------------

before_df = spark.table(OUTPUT_TABLE)

records_before = before_df.count()

batch_records_before = (
    before_df
    .filter(F.col("batch_id") == BATCH_ID)
    .count()
)

print("\nBEFORE RETRY")
print("Total table records:", records_before)
print("Current batch records:", batch_records_before)

# --------------------------------------------------
# 2. Repeat MERGE with the same batch ID
# --------------------------------------------------

print("\nRepeating Delta MERGE for the same batch...")

retry_target = DeltaTable.forName(
    spark,
    OUTPUT_TABLE
)

(
    retry_target.alias("target")
    .merge(
        batch_predictions_df.alias("source"),
        """
        target.batch_id = source.batch_id
        AND target.customerID = source.customerID
        """
    )
    .whenNotMatchedInsertAll()
    .execute()
)

print("Retry MERGE completed.")

# --------------------------------------------------
# 3. Capture table state after retry
# --------------------------------------------------

after_df = spark.table(OUTPUT_TABLE)

records_after = after_df.count()

batch_records_after = (
    after_df
    .filter(F.col("batch_id") == BATCH_ID)
    .count()
)

# --------------------------------------------------
# 4. Check for duplicate batch/customer keys
# --------------------------------------------------

unique_keys_after = (
    after_df
    .select("batch_id", "customerID")
    .distinct()
    .count()
)

duplicate_keys_after = (
    records_after - unique_keys_after
)

# --------------------------------------------------
# 5. Validate idempotency
# --------------------------------------------------

print("\nIDEMPOTENCY VALIDATION")
print("Records before retry:", records_before)
print("Records after retry:", records_after)
print("Batch records before:", batch_records_before)
print("Batch records after:", batch_records_after)
print("Duplicate batch/customer keys:", duplicate_keys_after)

assert records_after == records_before, (
    "Retry changed the total record count."
)

assert batch_records_after == batch_records_before, (
    "Retry changed the batch record count."
)

assert duplicate_keys_after == 0, (
    "Duplicate records detected after retry."
)

print("\nPRODUCTION BATCH IDEMPOTENCY: PASS")


# COMMAND ----------


# Step 15.12: Build production inference batch summary

from pyspark.sql import functions as F

print("=" * 65)
print("PRODUCTION INFERENCE BATCH MONITORING SUMMARY")
print("=" * 65)

# --------------------------------------------------
# 1. Read the current persisted inference batch
# --------------------------------------------------

monitoring_source_df = (
    spark.table(OUTPUT_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

# --------------------------------------------------
# 2. Generate batch-level monitoring metrics
# --------------------------------------------------

production_batch_summary_df = (
    monitoring_source_df
    .groupBy(
        "batch_id",
        "model_name",
        "model_version",
        "prediction_timestamp",
        "classification_threshold",
        "source_table"
    )
    .agg(
        F.count("*").alias("total_customers"),

        F.sum(
            F.when(
                F.col("predicted_class") == 1, 1
            ).otherwise(0)
        ).alias("predicted_churners"),

        F.sum(
            F.when(
                F.col("predicted_class") == 0, 1
            ).otherwise(0)
        ).alias("predicted_non_churners"),

        F.avg("churn_probability").alias(
            "average_churn_probability"
        ),

        F.min("churn_probability").alias(
            "minimum_churn_probability"
        ),

        F.max("churn_probability").alias(
            "maximum_churn_probability"
        )
    )
    .withColumn(
        "predicted_churn_rate",
        F.col("predicted_churners")
        / F.col("total_customers")
    )
)

# --------------------------------------------------
# 3. Display operational metrics
# --------------------------------------------------

print("\nBATCH MONITORING METRICS")

production_batch_summary_df.select(
    "total_customers",
    "predicted_churners",
    "predicted_non_churners",
    F.round(
        F.col("predicted_churn_rate") * 100, 2
    ).alias("predicted_churn_rate_pct"),
    F.round(
        "average_churn_probability", 4
    ).alias("average_churn_probability"),
    F.round(
        "minimum_churn_probability", 4
    ).alias("minimum_churn_probability"),
    F.round(
        "maximum_churn_probability", 4
    ).alias("maximum_churn_probability")
).show(truncate=False)

# --------------------------------------------------
# 4. Validate batch summary
# --------------------------------------------------

summary_rows = production_batch_summary_df.count()

summary = production_batch_summary_df.first()

assert summary_rows == 1, (
    "Expected exactly one batch summary."
)

assert summary["total_customers"] == 1409

assert (
    summary["predicted_churners"]
    + summary["predicted_non_churners"]
    == summary["total_customers"]
)

assert 0 <= summary["predicted_churn_rate"] <= 1

assert 0 <= summary["average_churn_probability"] <= 1

# --------------------------------------------------
# 5. Display batch metadata
# --------------------------------------------------

print("\nBATCH METADATA")

production_batch_summary_df.select(
    "batch_id",
    "model_name",
    "model_version",
    "prediction_timestamp",
    "classification_threshold",
    "source_table"
).show(truncate=False)

print("\nBATCH SUMMARY VALIDATION")
print("Summary records:", summary_rows)
print("Total customers:", summary["total_customers"])
print("Predicted churners:", summary["predicted_churners"])
print(
    "Predicted churn rate:",
    round(summary["predicted_churn_rate"] * 100, 2),
    "%"
)

print("\nPRODUCTION BATCH MONITORING SUMMARY: PASS")


# COMMAND ----------

# DBTITLE 1,Step 15.13: Persist batch monitoring summary to Delta

# Step 15.13: Persist batch monitoring summary to Delta

from delta.tables import DeltaTable
from pyspark.sql import functions as F

print("=" * 65)
print("PRODUCTION BATCH MONITORING DELTA PERSISTENCE")
print("=" * 65)

# --------------------------------------------------
# 1. Define monitoring destination
# --------------------------------------------------

MONITORING_TABLE = (
    "workspace.default.telco_15_batch_monitoring"
)

print("\nMonitoring table:", MONITORING_TABLE)
print("Current batch ID:", BATCH_ID)

# --------------------------------------------------
# 2. Validate monitoring summary before writing
# --------------------------------------------------

monitoring_record_count = (
    production_batch_summary_df.count()
)

unique_monitoring_batches = (
    production_batch_summary_df
    .select("batch_id")
    .distinct()
    .count()
)

assert monitoring_record_count == 1
assert unique_monitoring_batches == 1

print("\nMONITORING INPUT VALIDATION")
print("Summary records:", monitoring_record_count)
print("Unique batch IDs:", unique_monitoring_batches)

# --------------------------------------------------
# 3. Create monitoring table if necessary
# --------------------------------------------------

monitoring_table_exists = (
    spark.catalog.tableExists(MONITORING_TABLE)
)

if not monitoring_table_exists:

    print("\nCreating batch monitoring Delta table...")

    (
        production_batch_summary_df
        .limit(0)
        .write
        .format("delta")
        .mode("error")
        .saveAsTable(MONITORING_TABLE)
    )

    print("Monitoring Delta table created.")

else:
    print("\nMonitoring Delta table already exists.")

# --------------------------------------------------
# 4. Merge monitoring summary by batch ID
# --------------------------------------------------

print("\nMerging batch monitoring summary...")

monitoring_target = DeltaTable.forName(
    spark,
    MONITORING_TABLE
)

(
    monitoring_target.alias("target")
    .merge(
        production_batch_summary_df.alias("source"),
        "target.batch_id = source.batch_id"
    )
    .whenNotMatchedInsertAll()
    .execute()
)

print("Monitoring Delta MERGE completed.")

# --------------------------------------------------
# 5. Validate saved monitoring record
# --------------------------------------------------

saved_monitoring_df = (
    spark.table(MONITORING_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

saved_monitoring_count = saved_monitoring_df.count()

assert saved_monitoring_count == 1

saved_summary = saved_monitoring_df.first()

assert (
    saved_summary["total_customers"]
    == batch_record_count
)

assert saved_summary["predicted_churners"] == 305

print("\nMONITORING PERSISTENCE VALIDATION")
print("Saved batch summary records:", saved_monitoring_count)
print("Customers scored:", saved_summary["total_customers"])
print("Predicted churners:", saved_summary["predicted_churners"])
print(
    "Predicted churn rate:",
    round(
        saved_summary["predicted_churn_rate"] * 100,
        2
    ),
    "%"
)

# --------------------------------------------------
# 6. Display persisted monitoring summary
# --------------------------------------------------

print("\nSAVED BATCH MONITORING SUMMARY")

saved_monitoring_df.select(
    "batch_id",
    "model_version",
    "total_customers",
    "predicted_churners",
    "predicted_non_churners",
    F.round(
        F.col("predicted_churn_rate") * 100,
        2
    ).alias("predicted_churn_rate_pct")
).show(truncate=False)

print("\nPRODUCTION BATCH MONITORING PERSISTENCE: PASS")


# COMMAND ----------


# Step 15.14: Validate prediction and monitoring consistency

from pyspark.sql import functions as F

print("=" * 65)
print("PRODUCTION CROSS-TABLE CONSISTENCY VALIDATION")
print("=" * 65)

# --------------------------------------------------
# 1. Read current batch from both Delta tables
# --------------------------------------------------

predictions_check_df = (
    spark.table(OUTPUT_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

monitoring_check_df = (
    spark.table(MONITORING_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

# --------------------------------------------------
# 2. Independently calculate prediction metrics
# --------------------------------------------------

actual_metrics_df = (
    predictions_check_df
    .groupBy("batch_id")
    .agg(
        F.count("*").alias("actual_customers"),

        F.sum(
            F.when(
                F.col("predicted_class") == 1, 1
            ).otherwise(0)
        ).alias("actual_churners"),

        F.sum(
            F.when(
                F.col("predicted_class") == 0, 1
            ).otherwise(0)
        ).alias("actual_non_churners"),

        F.avg("churn_probability").alias(
            "actual_avg_probability"
        ),

        F.min("churn_probability").alias(
            "actual_min_probability"
        ),

        F.max("churn_probability").alias(
            "actual_max_probability"
        )
    )
    .withColumn(
        "actual_churn_rate",
        F.col("actual_churners") /
        F.col("actual_customers")
    )
)

# --------------------------------------------------
# 3. Join actual metrics to stored monitoring metrics
# --------------------------------------------------

consistency_df = (
    actual_metrics_df.alias("a")
    .join(
        monitoring_check_df.alias("m"),
        on="batch_id",
        how="inner"
    )
)

# --------------------------------------------------
# 4. Validate both tables contain exactly one batch
# --------------------------------------------------

assert actual_metrics_df.count() == 1
assert monitoring_check_df.count() == 1
assert consistency_df.count() == 1

# --------------------------------------------------
# 5. Compare prediction counts and statistics
# --------------------------------------------------

result = consistency_df.first()

assert (
    result["actual_customers"]
    == result["total_customers"]
)

assert (
    result["actual_churners"]
    == result["predicted_churners"]
)

assert (
    result["actual_non_churners"]
    == result["predicted_non_churners"]
)

# Floating-point comparisons use a small tolerance
TOLERANCE = 1e-10

metric_pairs = [
    ("actual_churn_rate", "predicted_churn_rate"),
    ("actual_avg_probability", "average_churn_probability"),
    ("actual_min_probability", "minimum_churn_probability"),
    ("actual_max_probability", "maximum_churn_probability")
]

for actual_col, stored_col in metric_pairs:
    assert abs(
        result[actual_col] - result[stored_col]
    ) < TOLERANCE, (
        f"Mismatch: {actual_col} vs {stored_col}"
    )

# --------------------------------------------------
# 6. Display cross-table comparison
# --------------------------------------------------

print("\nCROSS-TABLE METRIC COMPARISON")

comparison_df = consistency_df.select(
    "batch_id",
    F.col("actual_customers").alias("prediction_count"),
    F.col("total_customers").alias("monitoring_count"),
    F.col("actual_churners").alias("prediction_churners"),
    F.col("predicted_churners").alias("monitoring_churners"),
    F.round(
        F.col("actual_churn_rate") * 100, 2
    ).alias("prediction_churn_rate_pct"),
    F.round(
        F.col("predicted_churn_rate") * 100, 2
    ).alias("monitoring_churn_rate_pct")
)

comparison_df.show(truncate=False)

print("\nCONSISTENCY VALIDATION")
print("Customer counts match: True")
print("Predicted churn counts match: True")
print("Predicted non-churn counts match: True")
print("Average probabilities match: True")
print("Minimum probabilities match: True")
print("Maximum probabilities match: True")
print("Predicted churn rates match: True")

print("\nPRODUCTION CROSS-TABLE CONSISTENCY: PASS")


# COMMAND ----------


# Step 15.15: Implement production inference quality gates

from pyspark.sql import functions as F

print("=" * 65)
print("PRODUCTION INFERENCE QUALITY GATES")
print("=" * 65)

# --------------------------------------------------
# 1. Load persisted current batch
# --------------------------------------------------

quality_df = (
    spark.table(OUTPUT_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

monitoring_quality_df = (
    spark.table(MONITORING_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

# --------------------------------------------------
# 2. Calculate quality metrics
# --------------------------------------------------

quality_record_count = quality_df.count()

quality_unique_customers = (
    quality_df
    .select("customerID")
    .distinct()
    .count()
)

invalid_probability_count = (
    quality_df
    .filter(
        F.col("churn_probability").isNull()
        | F.isnan("churn_probability")
        | (F.col("churn_probability") < 0)
        | (F.col("churn_probability") > 1)
    )
    .count()
)

missing_metadata_count = (
    quality_df
    .filter(
        F.col("batch_id").isNull()
        | F.col("prediction_timestamp").isNull()
        | F.col("model_name").isNull()
        | F.col("model_version").isNull()
        | F.col("classification_threshold").isNull()
        | F.col("source_table").isNull()
    )
    .count()
)

invalid_classification_count = (
    quality_df
    .filter(
        F.col("predicted_class").isNull()
        | (~F.col("predicted_class").isin(0, 1))
        | F.col("classification_threshold").isNull()
        | F.col("churn_probability").isNull()
        | (
            F.col("predicted_class") !=
            F.when(
                F.col("churn_probability") >=
                F.col("classification_threshold"),
                F.lit(1)
            ).otherwise(F.lit(0))
        )
    )
    .count()
)

monitoring_rows = monitoring_quality_df.count()

if monitoring_rows == 1:
    monitoring_row = monitoring_quality_df.first()

    actual_churners = (
        quality_df
        .filter(F.col("predicted_class") == 1)
        .count()
    )

    actual_non_churners = (
        quality_df
        .filter(F.col("predicted_class") == 0)
        .count()
    )

    monitoring_matches = (
        monitoring_row["total_customers"]
        == quality_record_count
        and monitoring_row["predicted_churners"]
        == actual_churners
        and monitoring_row["predicted_non_churners"]
        == actual_non_churners
    )
else:
    monitoring_matches = False

# --------------------------------------------------
# 3. Define quality gate results
# --------------------------------------------------

quality_gates = {
    "Record completeness": (
        quality_record_count == input_count
    ),

    "Customer uniqueness": (
        quality_unique_customers == quality_record_count
    ),

    "Probability validity": (
        invalid_probability_count == 0
    ),

    "Metadata completeness": (
        missing_metadata_count == 0
    ),

    "Classification consistency": (
        invalid_classification_count == 0
    ),

    "Monitoring reconciliation": (
        monitoring_matches
    )
}

# --------------------------------------------------
# 4. Display quality gate report
# --------------------------------------------------

print("\nQUALITY GATE RESULTS")

for gate_name, passed in quality_gates.items():
    status = "PASS" if passed else "FAIL"
    print(f"{gate_name}: {status}")

# --------------------------------------------------
# 5. Calculate overall batch quality status
# --------------------------------------------------

passed_gates = sum(quality_gates.values())
total_gates = len(quality_gates)

overall_quality_status = (
    "PASS"
    if all(quality_gates.values())
    else "FAIL"
)

print("\nOVERALL QUALITY SUMMARY")
print("Total quality gates:", total_gates)
print("Passed quality gates:", passed_gates)
print("Failed quality gates:", total_gates - passed_gates)
print("Overall batch status:", overall_quality_status)

# --------------------------------------------------
# 6. Enforce quality gate validation
# --------------------------------------------------

assert overall_quality_status == "PASS", (
    "Production inference quality gates failed."
)

print("\nPRODUCTION INFERENCE QUALITY GATES: PASS")


# COMMAND ----------

# DBTITLE 1,Step 15.16: Persist batch quality audit results

# Step 15.16: Persist batch quality audit results

from datetime import datetime, timezone
from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    TimestampType
)

print("=" * 65)
print("PRODUCTION BATCH QUALITY AUDIT")
print("=" * 65)

# --------------------------------------------------
# 1. Define audit table
# --------------------------------------------------

QUALITY_AUDIT_TABLE = (
    "workspace.default.telco_15_batch_quality_audit"
)

AUDIT_TIMESTAMP = datetime.now(timezone.utc)

print("\nQuality audit table:", QUALITY_AUDIT_TABLE)
print("Batch ID:", BATCH_ID)

# --------------------------------------------------
# 2. Prepare audit record
# --------------------------------------------------

audit_schema = StructType([
    StructField("batch_id", StringType(), False),
    StructField("model_name", StringType(), False),
    StructField("model_version", StringType(), False),
    StructField("quality_status", StringType(), False),
    StructField("total_gates", IntegerType(), False),
    StructField("passed_gates", IntegerType(), False),
    StructField("failed_gates", IntegerType(), False),
    StructField("audit_timestamp", TimestampType(), False)
])

audit_data = [(
    BATCH_ID,
    MODEL_NAME,
    MODEL_VERSION,
    overall_quality_status,
    int(total_gates),
    int(passed_gates),
    int(total_gates - passed_gates),
    AUDIT_TIMESTAMP
)]

quality_audit_df = spark.createDataFrame(
    audit_data,
    schema=audit_schema
)

print("\nQUALITY AUDIT RECORD")

quality_audit_df.show(truncate=False)

# --------------------------------------------------
# 3. Validate audit record
# --------------------------------------------------

assert quality_audit_df.count() == 1

assert overall_quality_status == "PASS"

assert passed_gates == total_gates

# --------------------------------------------------
# 4. Create Delta audit table if necessary
# --------------------------------------------------

if not spark.catalog.tableExists(QUALITY_AUDIT_TABLE):

    print("\nCreating quality audit Delta table...")

    (
        quality_audit_df
        .limit(0)
        .write
        .format("delta")
        .mode("error")
        .saveAsTable(QUALITY_AUDIT_TABLE)
    )

    print("Quality audit Delta table created.")

else:
    print("\nQuality audit Delta table already exists.")

# --------------------------------------------------
# 5. Merge audit record by batch ID
# --------------------------------------------------

print("\nSaving quality audit record...")

audit_target = DeltaTable.forName(
    spark,
    QUALITY_AUDIT_TABLE
)

(
    audit_target.alias("target")
    .merge(
        quality_audit_df.alias("source"),
        "target.batch_id = source.batch_id"
    )
    .whenNotMatchedInsertAll()
    .execute()
)

print("Quality audit MERGE completed.")

# --------------------------------------------------
# 6. Validate persisted audit record
# --------------------------------------------------

saved_audit_df = (
    spark.table(QUALITY_AUDIT_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

saved_audit_count = saved_audit_df.count()

assert saved_audit_count == 1

saved_audit = saved_audit_df.first()

assert saved_audit["quality_status"] == "PASS"
assert saved_audit["total_gates"] == 6
assert saved_audit["passed_gates"] == 6
assert saved_audit["failed_gates"] == 0

print("\nPERSISTED QUALITY AUDIT")
print("Saved audit records:", saved_audit_count)
print("Quality status:", saved_audit["quality_status"])
print("Total gates:", saved_audit["total_gates"])
print("Passed gates:", saved_audit["passed_gates"])
print("Failed gates:", saved_audit["failed_gates"])

print("\nPRODUCTION BATCH QUALITY AUDIT: PASS")


# COMMAND ----------


# Step 15.17: Validate complete production inference audit trail

from pyspark.sql import functions as F

print("=" * 65)
print("PRODUCTION INFERENCE AUDIT TRAIL VALIDATION")
print("=" * 65)

# --------------------------------------------------
# 1. Load current batch from all three Delta tables
# --------------------------------------------------

audit_predictions_df = (
    spark.table(OUTPUT_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

audit_monitoring_df = (
    spark.table(MONITORING_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

audit_quality_df = (
    spark.table(QUALITY_AUDIT_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

# --------------------------------------------------
# 2. Validate batch existence
# --------------------------------------------------

prediction_count = audit_predictions_df.count()
monitoring_count = audit_monitoring_df.count()
quality_audit_count = audit_quality_df.count()

print("\nBATCH RECORD AVAILABILITY")
print("Customer predictions:", prediction_count)
print("Monitoring summaries:", monitoring_count)
print("Quality audit records:", quality_audit_count)

assert prediction_count > 0
assert monitoring_count == 1
assert quality_audit_count == 1

# --------------------------------------------------
# 3. Read monitoring and audit records
# --------------------------------------------------

monitoring_record = audit_monitoring_df.first()
quality_record = audit_quality_df.first()

# --------------------------------------------------
# 4. Validate record counts
# --------------------------------------------------

assert (
    prediction_count
    == monitoring_record["total_customers"]
)

# --------------------------------------------------
# 5. Validate model traceability
# --------------------------------------------------

prediction_model_metadata = (
    audit_predictions_df
    .select("model_name", "model_version")
    .distinct()
    .collect()
)

assert len(prediction_model_metadata) == 1

prediction_model = prediction_model_metadata[0]

assert (
    prediction_model["model_name"]
    == monitoring_record["model_name"]
    == quality_record["model_name"]
    == MODEL_NAME
)

assert (
    prediction_model["model_version"]
    == monitoring_record["model_version"]
    == quality_record["model_version"]
    == MODEL_VERSION
)

# --------------------------------------------------
# 6. Validate quality audit status
# --------------------------------------------------

assert quality_record["quality_status"] == "PASS"

assert quality_record["total_gates"] == 6
assert quality_record["passed_gates"] == 6
assert quality_record["failed_gates"] == 0

# --------------------------------------------------
# 7. Validate unique customer predictions
# --------------------------------------------------

unique_prediction_keys = (
    audit_predictions_df
    .select("batch_id", "customerID")
    .distinct()
    .count()
)

duplicate_prediction_keys = (
    prediction_count - unique_prediction_keys
)

assert duplicate_prediction_keys == 0

# --------------------------------------------------
# 8. Validate prediction counts against monitoring
# --------------------------------------------------

class_counts = (
    audit_predictions_df
    .groupBy("predicted_class")
    .count()
    .collect()
)

class_count_map = {
    row["predicted_class"]: row["count"]
    for row in class_counts
}

actual_churners = class_count_map.get(1, 0)
actual_non_churners = class_count_map.get(0, 0)

assert (
    actual_churners
    == monitoring_record["predicted_churners"]
)

assert (
    actual_non_churners
    == monitoring_record["predicted_non_churners"]
)

assert (
    actual_churners + actual_non_churners
    == prediction_count
)

# --------------------------------------------------
# 9. Display consolidated audit trail
# --------------------------------------------------

print("\nCOMPLETE INFERENCE AUDIT TRAIL")
print("Batch ID:", BATCH_ID)
print("Model:", MODEL_NAME)
print("Model version:", MODEL_VERSION)
print("Customers scored:", prediction_count)
print("Predicted churners:", actual_churners)
print("Predicted non-churners:", actual_non_churners)
print(
    "Predicted churn rate:",
    round(
        monitoring_record["predicted_churn_rate"] * 100,
        2
    ),
    "%"
)
print("Duplicate prediction keys:", duplicate_prediction_keys)
print("Quality status:", quality_record["quality_status"])
print("Quality gates passed:", quality_record["passed_gates"])

# --------------------------------------------------
# 10. Final validation
# --------------------------------------------------

print("\nAUDIT TRAIL VALIDATION RESULTS")
print("Batch exists in all three tables: PASS")
print("Prediction and monitoring counts: PASS")
print("Model name consistency: PASS")
print("Model version consistency: PASS")
print("Quality audit status: PASS")
print("Quality gate results: PASS")
print("Customer prediction uniqueness: PASS")
print("Prediction class reconciliation: PASS")

print("\nPRODUCTION INFERENCE AUDIT TRAIL: PASS")


# COMMAND ----------


# Step 15.18: Prepare quality-gated predictions
# for downstream consumption

from pyspark.sql import functions as F

print("=" * 65)
print("PRODUCTION QUALITY-GATED PREDICTION RELEASE")
print("=" * 65)

# --------------------------------------------------
# 1. Load persisted predictions
# --------------------------------------------------

release_predictions_df = (
    spark.table(OUTPUT_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

# --------------------------------------------------
# 2. Load approved quality audit records
# --------------------------------------------------

approved_batches_df = (
    spark.table(QUALITY_AUDIT_TABLE)
    .filter(
        (F.col("batch_id") == BATCH_ID)
        & (F.col("quality_status") == "PASS")
        & (F.col("total_gates") == 6)
        & (F.col("passed_gates") == 6)
        & (F.col("failed_gates") == 0)
    )
    .select(
        "batch_id",
        "model_name",
        "model_version"
    )
    .distinct()
)

# --------------------------------------------------
# 3. Verify approval status
# --------------------------------------------------

approved_batch_count = approved_batches_df.count()

print("\nQUALITY APPROVAL CHECK")
print("Approved batch records:", approved_batch_count)

assert approved_batch_count == 1, (
    "Batch is not approved for downstream release."
)

# --------------------------------------------------
# 4. Join predictions with approved batch metadata
# --------------------------------------------------

approved_predictions_df = (
    release_predictions_df.alias("p")
    .join(
        approved_batches_df.alias("a"),
        on=[
            "batch_id",
            "model_name",
            "model_version"
        ],
        how="inner"
    )
)

# --------------------------------------------------
# 5. Validate released prediction count
# --------------------------------------------------

source_prediction_count = release_predictions_df.count()

approved_prediction_count = (
    approved_predictions_df.count()
)

unique_approved_customers = (
    approved_predictions_df
    .select("customerID")
    .distinct()
    .count()
)

print("\nRELEASE VALIDATION")
print("Source predictions:", source_prediction_count)
print("Approved predictions:", approved_prediction_count)
print("Unique approved customers:", unique_approved_customers)

assert approved_prediction_count == source_prediction_count

assert unique_approved_customers == approved_prediction_count

# --------------------------------------------------
# 6. Validate approved prediction classes
# --------------------------------------------------

approved_churners = (
    approved_predictions_df
    .filter(F.col("predicted_class") == 1)
    .count()
)

approved_non_churners = (
    approved_predictions_df
    .filter(F.col("predicted_class") == 0)
    .count()
)

assert (
    approved_churners + approved_non_churners
    == approved_prediction_count
)

print("\nAPPROVED PREDICTION SUMMARY")
print("Approved customers:", approved_prediction_count)
print("Predicted churners:", approved_churners)
print("Predicted non-churners:", approved_non_churners)

# --------------------------------------------------
# 7. Preview approved predictions
# --------------------------------------------------

print("\nAPPROVED CUSTOMER PREDICTIONS")

approved_predictions_df.select(
    "customerID",
    "churn_probability",
    "predicted_class",
    "predicted_churn",
    "model_version",
    "batch_id"
).orderBy(
    F.desc("churn_probability")
).show(10, truncate=False)

# --------------------------------------------------
# 8. Final release eligibility validation
# --------------------------------------------------

print("\nQUALITY-GATED RELEASE STATUS: PASS")
print("Batch is eligible for downstream consumption.")


# COMMAND ----------

# DBTITLE 1,Step 15.19: Publish approved predictions to Delta

# Step 15.19: Publish approved predictions to Delta

from delta.tables import DeltaTable
from pyspark.sql import functions as F

print("=" * 65)
print("PRODUCTION APPROVED PREDICTION PUBLISHING")
print("=" * 65)

# --------------------------------------------------
# 1. Define approved prediction destination
# --------------------------------------------------

APPROVED_OUTPUT_TABLE = (
    "workspace.default.telco_15_approved_predictions"
)

print("\nApproved output table:", APPROVED_OUTPUT_TABLE)
print("Batch ID:", BATCH_ID)

# --------------------------------------------------
# 2. Revalidate batch approval before publishing
# --------------------------------------------------

current_approval_df = (
    spark.table(QUALITY_AUDIT_TABLE)
    .filter(
        (F.col("batch_id") == BATCH_ID)
        & (F.col("quality_status") == "PASS")
        & (F.col("total_gates") == 6)
        & (F.col("passed_gates") == 6)
        & (F.col("failed_gates") == 0)
        & (F.col("model_name") == MODEL_NAME)
        & (F.col("model_version") == MODEL_VERSION)
    )
)

assert current_approval_df.count() == 1, (
    "Batch does not have valid quality approval."
)

# --------------------------------------------------
# 3. Prepare approved predictions for publication
# --------------------------------------------------

publish_df = (
    approved_predictions_df
    .withColumn(
        "publication_timestamp",
        F.current_timestamp()
    )
)

publish_count = publish_df.count()

unique_publish_keys = (
    publish_df
    .select("batch_id", "customerID")
    .distinct()
    .count()
)

assert publish_count == unique_publish_keys

assert publish_count == source_prediction_count

print("\nPRE-PUBLICATION VALIDATION")
print("Approved batch status: PASS")
print("Predictions to publish:", publish_count)
print("Unique batch/customer keys:", unique_publish_keys)

# --------------------------------------------------
# 4. Create approved Delta table if needed
# --------------------------------------------------

if not spark.catalog.tableExists(APPROVED_OUTPUT_TABLE):

    print("\nCreating approved prediction Delta table...")

    (
        publish_df
        .limit(0)
        .write
        .format("delta")
        .mode("error")
        .saveAsTable(APPROVED_OUTPUT_TABLE)
    )

    print("Approved prediction table created.")

else:
    print("\nApproved prediction table already exists.")

# --------------------------------------------------
# 5. Publish using idempotent Delta MERGE
# --------------------------------------------------

print("\nPublishing approved predictions...")

approved_target = DeltaTable.forName(
    spark,
    APPROVED_OUTPUT_TABLE
)

(
    approved_target.alias("target")
    .merge(
        publish_df.alias("source"),
        """
        target.batch_id = source.batch_id
        AND target.customerID = source.customerID
        """
    )
    .whenNotMatchedInsertAll()
    .execute()
)

print("Approved prediction MERGE completed.")

# --------------------------------------------------
# 6. Validate persisted approved predictions
# --------------------------------------------------

published_batch_df = (
    spark.table(APPROVED_OUTPUT_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

published_count = published_batch_df.count()

published_unique_customers = (
    published_batch_df
    .select("customerID")
    .distinct()
    .count()
)

published_churners = (
    published_batch_df
    .filter(F.col("predicted_class") == 1)
    .count()
)

published_non_churners = (
    published_batch_df
    .filter(F.col("predicted_class") == 0)
    .count()
)

assert published_count == publish_count

assert published_unique_customers == published_count

assert published_churners == approved_churners

assert published_non_churners == approved_non_churners

# --------------------------------------------------
# 7. Display publication results
# --------------------------------------------------

print("\nPUBLISHED PREDICTION VALIDATION")
print("Published records:", published_count)
print("Unique customers:", published_unique_customers)
print("Predicted churners:", published_churners)
print("Predicted non-churners:", published_non_churners)

print("\nPUBLISHED PREDICTION SAMPLE")

published_batch_df.select(
    "customerID",
    "churn_probability",
    "predicted_churn",
    "model_version",
    "batch_id",
    "publication_timestamp"
).orderBy(
    F.desc("churn_probability")
).show(10, truncate=False)

print("\nPRODUCTION APPROVED PREDICTION PUBLISHING: PASS")


# COMMAND ----------


# Step 15.20: Verify published prediction integrity
# and idempotent publication

from delta.tables import DeltaTable
from pyspark.sql import functions as F

print("=" * 65)
print("PRODUCTION PUBLISHED PREDICTION INTEGRITY")
print("=" * 65)

# --------------------------------------------------
# 1. Load original and published predictions
# --------------------------------------------------

original_df = (
    spark.table(OUTPUT_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

published_df = (
    spark.table(APPROVED_OUTPUT_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

original_count = original_df.count()
published_count_before = published_df.count()

print("\nBEFORE PUBLICATION RETRY")
print("Original predictions:", original_count)
print("Published predictions:", published_count_before)

assert original_count == published_count_before

# --------------------------------------------------
# 2. Compare original and published prediction data
# --------------------------------------------------

comparison_columns = [
    "batch_id",
    "customerID",
    "churn_probability",
    "predicted_class",
    "predicted_churn",
    "model_name",
    "model_version",
    "classification_threshold",
    "prediction_timestamp",
    "source_table"
]

original_comparison_df = original_df.select(
    *comparison_columns
)

published_comparison_df = published_df.select(
    *comparison_columns
)

# exceptAll compares complete rows and preserves duplicates
missing_published_records = (
    original_comparison_df
    .exceptAll(published_comparison_df)
    .count()
)

unexpected_published_records = (
    published_comparison_df
    .exceptAll(original_comparison_df)
    .count()
)

print("\nPUBLISHED DATA INTEGRITY")
print("Missing published records:", missing_published_records)
print("Unexpected published records:", unexpected_published_records)

assert missing_published_records == 0
assert unexpected_published_records == 0

# --------------------------------------------------
# 3. Retry publication for the same batch
# --------------------------------------------------

print("\nRepeating publication MERGE...")

retry_publish_df = (
    approved_predictions_df
    .withColumn(
        "publication_timestamp",
        F.current_timestamp()
    )
)

retry_target = DeltaTable.forName(
    spark,
    APPROVED_OUTPUT_TABLE
)

(
    retry_target.alias("target")
    .merge(
        retry_publish_df.alias("source"),
        """
        target.batch_id = source.batch_id
        AND target.customerID = source.customerID
        """
    )
    .whenNotMatchedInsertAll()
    .execute()
)

print("Publication retry completed.")

# --------------------------------------------------
# 4. Validate record counts after retry
# --------------------------------------------------

published_after_df = (
    spark.table(APPROVED_OUTPUT_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

published_count_after = published_after_df.count()

unique_published_keys = (
    published_after_df
    .select("batch_id", "customerID")
    .distinct()
    .count()
)

duplicate_published_keys = (
    published_count_after - unique_published_keys
)

assert published_count_after == published_count_before
assert duplicate_published_keys == 0

# --------------------------------------------------
# 5. Verify original publication timestamps
# remain unchanged
# --------------------------------------------------

timestamp_changes = (
    published_df
    .select(
        "batch_id",
        "customerID",
        F.col("publication_timestamp").alias(
            "timestamp_before"
        )
    )
    .join(
        published_after_df.select(
            "batch_id",
            "customerID",
            F.col("publication_timestamp").alias(
                "timestamp_after"
            )
        ),
        on=["batch_id", "customerID"],
        how="inner"
    )
    .filter(
        ~F.col("timestamp_before").eqNullSafe(
            F.col("timestamp_after")
        )
    )
    .count()
)

assert timestamp_changes == 0

# --------------------------------------------------
# 6. Display validation results
# --------------------------------------------------

print("\nPUBLICATION IDEMPOTENCY VALIDATION")
print("Published records before retry:", published_count_before)
print("Published records after retry:", published_count_after)
print("Duplicate published keys:", duplicate_published_keys)
print("Changed publication timestamps:", timestamp_changes)

print("\nPUBLISHED PREDICTION INTEGRITY: PASS")
print("PUBLICATION IDEMPOTENCY: PASS")


# COMMAND ----------


# Step 15.21: Production deployment readiness assessment

from pyspark.sql import functions as F

print("=" * 65)
print("TELCO CHURN ML - DEPLOYMENT READINESS ASSESSMENT")
print("=" * 65)

# --------------------------------------------------
# 1. Define required production artifacts
# --------------------------------------------------

required_tables = {
    "Customer predictions": OUTPUT_TABLE,
    "Batch monitoring": MONITORING_TABLE,
    "Quality audit": QUALITY_AUDIT_TABLE,
    "Approved predictions": APPROVED_OUTPUT_TABLE
}

# --------------------------------------------------
# 2. Verify all required Delta tables
# --------------------------------------------------

table_checks = {}

print("\nPRODUCTION ARTIFACT VALIDATION")

for artifact_name, table_name in required_tables.items():

    exists = spark.catalog.tableExists(table_name)

    if exists:
        table_details = (
            spark.sql(f"DESCRIBE DETAIL {table_name}")
            .select("format")
            .first()
        )
        is_delta = (
            table_details is not None
            and table_details["format"].lower() == "delta"
        )
    else:
        is_delta = False

    table_checks[artifact_name] = exists and is_delta

    status = "PASS" if table_checks[artifact_name] else "FAIL"

    print(f"{artifact_name}: {status}")

# --------------------------------------------------
# 3. Load persisted current-batch records
# --------------------------------------------------

readiness_predictions_df = (
    spark.table(OUTPUT_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

readiness_monitoring_df = (
    spark.table(MONITORING_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

readiness_audit_df = (
    spark.table(QUALITY_AUDIT_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

readiness_approved_df = (
    spark.table(APPROVED_OUTPUT_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

prediction_total = readiness_predictions_df.count()
monitoring_total = readiness_monitoring_df.count()
audit_total = readiness_audit_df.count()
approved_total = readiness_approved_df.count()

# --------------------------------------------------
# 4. Validate batch records and quality approval
# --------------------------------------------------

batch_records_valid = (
    prediction_total > 0
    and prediction_total == input_count
    and monitoring_total == 1
    and audit_total == 1
    and approved_total == prediction_total
)

audit_record = (
    readiness_audit_df.first()
    if audit_total == 1 else None
)

quality_approved = (
    audit_record is not None
    and audit_record["quality_status"] == "PASS"
    and audit_record["total_gates"] == 6
    and audit_record["passed_gates"] == 6
    and audit_record["failed_gates"] == 0
)

# --------------------------------------------------
# 5. Revalidate published prediction integrity
# --------------------------------------------------

comparison_columns = [
    "batch_id",
    "customerID",
    "churn_probability",
    "predicted_class",
    "predicted_churn",
    "model_name",
    "model_version",
    "classification_threshold",
    "prediction_timestamp",
    "source_table"
]

missing_records = (
    readiness_predictions_df
    .select(*comparison_columns)
    .exceptAll(
        readiness_approved_df.select(*comparison_columns)
    )
    .count()
)

unexpected_records = (
    readiness_approved_df
    .select(*comparison_columns)
    .exceptAll(
        readiness_predictions_df.select(*comparison_columns)
    )
    .count()
)

publication_integrity_valid = (
    missing_records == 0
    and unexpected_records == 0
)

# --------------------------------------------------
# 6. Build readiness checklist
# --------------------------------------------------

readiness_checks = {
    "All required Delta tables exist": all(
        table_checks.values()
    ),
    "Batch record counts reconcile": batch_records_valid,
    "Quality audit approved": quality_approved,
    "Published predictions match source": (
        publication_integrity_valid
    ),
    "Publication retry validation passed": (
        published_count_after == published_count_before
        and duplicate_published_keys == 0
        and timestamp_changes == 0
    )
}

print("\nBATCH PIPELINE READINESS CHECKLIST")

for check_name, passed in readiness_checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"{check_name}: {status}")

# --------------------------------------------------
# 7. Calculate readiness status
# --------------------------------------------------

total_checks = len(readiness_checks)
passed_checks = sum(readiness_checks.values())
failed_checks = total_checks - passed_checks

batch_pipeline_ready = all(readiness_checks.values())

print("\nREADINESS SUMMARY")
print("Total readiness checks:", total_checks)
print("Passed checks:", passed_checks)
print("Failed checks:", failed_checks)

print(
    "Batch pipeline readiness:",
    "READY" if batch_pipeline_ready else "NOT READY"
)

# --------------------------------------------------
# 8. Report operational deployment limitations
# --------------------------------------------------

print("\nOPERATIONAL DEPLOYMENT STATUS")
print("Automated job scheduling: NOT YET VERIFIED")
print("Production data integration: NOT YET IMPLEMENTED")
print("Unity Catalog access controls: NOT YET VERIFIED")
print("Automated alerting: NOT YET IMPLEMENTED")
print("Multi-batch operational monitoring: NOT YET VALIDATED")

print("\nOperational deployment readiness: INCOMPLETE")

# --------------------------------------------------
# 9. Final validation
# --------------------------------------------------

assert batch_pipeline_ready, (
    "Batch pipeline readiness validation failed."
)

print("\nBATCH PIPELINE READINESS ASSESSMENT: PASS")


# COMMAND ----------


# Step 15.22: Define Databricks job parameters
# Preparation for Lakeflow Jobs orchestration

from datetime import datetime, timezone
from uuid import uuid4

print("=" * 65)
print("DATABRICKS JOB ORCHESTRATION CONFIGURATION")
print("=" * 65)

# --------------------------------------------------
# 1. Define notebook parameters
# --------------------------------------------------

dbutils.widgets.text(
    "source_table",
    "workspace.default.telco_test_raw"
)

dbutils.widgets.text(
    "model_name",
    "workspace.default.telco_churn_logistic_regression"
)

dbutils.widgets.text(
    "model_version",
    "1"
)

dbutils.widgets.text(
    "classification_threshold",
    "0.50"
)

dbutils.widgets.text(
    "batch_id",
    ""
)

# --------------------------------------------------
# 2. Read parameter values
# --------------------------------------------------

JOB_SOURCE_TABLE = (
    dbutils.widgets.get("source_table").strip()
)

JOB_MODEL_NAME = (
    dbutils.widgets.get("model_name").strip()
)

JOB_MODEL_VERSION = (
    dbutils.widgets.get("model_version").strip()
)

JOB_THRESHOLD = float(
    dbutils.widgets.get("classification_threshold")
)

JOB_BATCH_ID = (
    dbutils.widgets.get("batch_id").strip()
)

# --------------------------------------------------
# 3. Validate parameter values
# --------------------------------------------------

assert JOB_SOURCE_TABLE, "Source table cannot be empty."

assert JOB_MODEL_NAME, "Model name cannot be empty."

assert JOB_MODEL_VERSION, "Model version cannot be empty."

assert 0 <= JOB_THRESHOLD <= 1, (
    "Classification threshold must be between 0 and 1."
)

# --------------------------------------------------
# 4. Generate a batch ID if not supplied
# --------------------------------------------------

if not JOB_BATCH_ID:
    JOB_BATCH_ID = str(uuid4())

JOB_CONFIG_TIMESTAMP = datetime.now(timezone.utc)

# --------------------------------------------------
# 5. Build registered model URI
# --------------------------------------------------

JOB_MODEL_URI = (
    f"models:/{JOB_MODEL_NAME}/{JOB_MODEL_VERSION}"
)

# --------------------------------------------------
# 6. Display job configuration
# --------------------------------------------------

print("\nJOB CONFIGURATION")

print("Source table:", JOB_SOURCE_TABLE)
print("Model name:", JOB_MODEL_NAME)
print("Model version:", JOB_MODEL_VERSION)
print("Model URI:", JOB_MODEL_URI)
print("Classification threshold:", JOB_THRESHOLD)
print("Batch ID:", JOB_BATCH_ID)
print("Configuration timestamp:", JOB_CONFIG_TIMESTAMP)

print("\nJOB PARAMETER VALIDATION: PASS")
print("Notebook configuration is ready for orchestration.")


# COMMAND ----------


# Step 15.23: Validate Databricks job configuration
# against registered model and source assets

import mlflow
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

print("=" * 65)
print("DATABRICKS JOB CONFIGURATION ASSET VALIDATION")
print("=" * 65)

# --------------------------------------------------
# 1. Validate source table
# --------------------------------------------------

print("\nSOURCE TABLE VALIDATION")

source_exists = spark.catalog.tableExists(
    JOB_SOURCE_TABLE
)

assert source_exists, (
    f"Source table does not exist: {JOB_SOURCE_TABLE}"
)

job_source_df = spark.table(JOB_SOURCE_TABLE)

job_source_count = job_source_df.count()

assert job_source_count > 0, (
    "Configured source table contains no records."
)

assert "customerID" in job_source_df.columns, (
    "Source table must contain customerID."
)

print("Source table:", JOB_SOURCE_TABLE)
print("Source exists:", source_exists)
print("Source records:", job_source_count)
print("Customer ID column: PASS")

# --------------------------------------------------
# 2. Validate registered model version
# --------------------------------------------------

print("\nREGISTERED MODEL VALIDATION")

mlflow.set_registry_uri("databricks-uc")

job_model_info = mlflow.models.get_model_info(
    JOB_MODEL_URI
)

assert job_model_info is not None

print("Registered model:", JOB_MODEL_NAME)
print("Model version:", JOB_MODEL_VERSION)
print("Model URI:", JOB_MODEL_URI)
print("Registered model metadata: PASS")

# --------------------------------------------------
# 3. Validate preprocessing pipeline
# --------------------------------------------------

print("\nPREPROCESSING PIPELINE VALIDATION")

job_preprocessing_pipeline = PipelineModel.load(
    PREPROCESSING_PIPELINE_PATH
)

preprocessing_stage_count = len(
    job_preprocessing_pipeline.stages
)

assert preprocessing_stage_count > 0

print(
    "Preprocessing pipeline:",
    PREPROCESSING_PIPELINE_PATH
)

print(
    "Preprocessing stages:",
    preprocessing_stage_count
)

print("Preprocessing pipeline: PASS")

# --------------------------------------------------
# 4. Validate classification threshold
# --------------------------------------------------

print("\nCLASSIFICATION THRESHOLD VALIDATION")

assert 0 <= JOB_THRESHOLD <= 1

print("Configured threshold:", JOB_THRESHOLD)
print("Classification threshold: PASS")

# --------------------------------------------------
# 5. Validate batch identifier
# --------------------------------------------------

print("\nBATCH IDENTIFIER VALIDATION")

assert JOB_BATCH_ID.strip(), (
    "Job batch ID cannot be empty."
)

print("Configured batch ID:", JOB_BATCH_ID)
print("Batch identifier: PASS")

# --------------------------------------------------
# 6. Display validation summary
# --------------------------------------------------

print("\nJOB ASSET VALIDATION SUMMARY")
print("Source table: PASS")
print("Registered model metadata: PASS")
print("Preprocessing pipeline: PASS")
print("Classification threshold: PASS")
print("Batch identifier: PASS")

print("\nDATABRICKS JOB CONFIGURATION ASSETS: PASS")


# COMMAND ----------

job_model_info = mlflow.models.get_model_info(
    JOB_MODEL_URI
)

# COMMAND ----------


# Step 15.24: Validate job input schema and model compatibility

import mlflow
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

print("=" * 65)
print("DATABRICKS JOB INPUT AND MODEL COMPATIBILITY")
print("=" * 65)

# --------------------------------------------------
# 1. Prepare a small sample of source records
# --------------------------------------------------

print("\nSOURCE SAMPLE VALIDATION")

job_input_sample_df = (
    spark.table(JOB_SOURCE_TABLE)
    .limit(10)
    .cache()
)

sample_count = job_input_sample_df.count()

assert sample_count == 10, (
    "Expected 10 records for compatibility testing."
)

print("Sample records:", sample_count)
print("Source columns:", len(job_input_sample_df.columns))

# --------------------------------------------------
# 2. Apply saved preprocessing pipeline
# --------------------------------------------------

print("\nPREPROCESSING COMPATIBILITY")

job_transformed_sample_df = (
    job_preprocessing_pipeline.transform(
        job_input_sample_df
    )
)

assert "features" in job_transformed_sample_df.columns, (
    "Preprocessing did not produce a features column."
)

feature_lengths_df = (
    job_transformed_sample_df
    .select(
        F.size(
            vector_to_array("features")
        ).alias("feature_length")
    )
)

feature_lengths = [
    row["feature_length"]
    for row in feature_lengths_df.distinct().collect()
]

print("Feature vector lengths:", feature_lengths)

assert feature_lengths == [60], (
    "Unexpected feature vector dimension."
)

print("Preprocessing compatibility: PASS")

# --------------------------------------------------
# 3. Load registered Spark MLflow model
# --------------------------------------------------

print("\nREGISTERED MODEL LOADING")

job_registered_model = (
    mlflow.spark.load_model(JOB_MODEL_URI)
)

print("Model URI:", JOB_MODEL_URI)
print("Registered model loading: PASS")

# --------------------------------------------------
# 4. Test inference on transformed sample
# --------------------------------------------------

print("\nMODEL INFERENCE COMPATIBILITY")

job_sample_predictions_df = (
    job_registered_model.transform(
        job_transformed_sample_df
    )
)

required_prediction_columns = [
    "prediction",
    "probability"
]

for column_name in required_prediction_columns:
    assert column_name in job_sample_predictions_df.columns, (
        f"Missing model output column: {column_name}"
    )

job_prediction_results_df = (
    job_sample_predictions_df
    .select(
        "customerID",
        F.col("prediction").alias("predicted_class"),
        vector_to_array("probability")[1].alias(
            "churn_probability"
        )
    )
    .cache()
)

prediction_count = job_prediction_results_df.count()

assert prediction_count == sample_count

invalid_probability_count = (
    job_prediction_results_df
    .filter(
        F.col("churn_probability").isNull()
        | F.isnan("churn_probability")
        | (F.col("churn_probability") < 0)
        | (F.col("churn_probability") > 1)
    )
    .count()
)

assert invalid_probability_count == 0

invalid_class_count = (
    job_prediction_results_df
    .filter(
        ~F.col("predicted_class").isin(0.0, 1.0)
        | F.col("predicted_class").isNull()
    )
    .count()
)

assert invalid_class_count == 0

print("Prediction records:", prediction_count)
print("Invalid probabilities:", invalid_probability_count)
print("Invalid prediction classes:", invalid_class_count)

print("\nSAMPLE PREDICTIONS")

job_prediction_results_df.show(
    10,
    truncate=False
)

# --------------------------------------------------
# 5. Final validation
# --------------------------------------------------

print("\nINPUT SCHEMA COMPATIBILITY: PASS")
print("PREPROCESSING COMPATIBILITY: PASS")
print("REGISTERED MODEL INFERENCE: PASS")

print("\nDATABRICKS JOB MODEL COMPATIBILITY: PASS")

# --------------------------------------------------
# 6. Release cached sample data
# --------------------------------------------------

job_prediction_results_df.unpersist()
job_input_sample_df.unpersist()


# COMMAND ----------

job_registered_model = mlflow.spark.load_model(
    JOB_MODEL_URI
)

# COMMAND ----------

job_prediction_results_df.unpersist()
job_input_sample_df.unpersist()

# COMMAND ----------


# Step 15.24A: Verify Spark Connect recovery
# Rebuild DataFrame directly from the source table

print("=" * 60)
print("SPARK CONNECT CACHE RECOVERY VALIDATION")
print("=" * 60)

# Create a fresh DataFrame from the original source
fresh_source_df = spark.table(JOB_SOURCE_TABLE)

# Read a small sample without caching
fresh_sample_df = fresh_source_df.limit(10)

fresh_count = fresh_sample_df.count()

print("\nFresh source sample count:", fresh_count)

assert fresh_count == 10, (
    "Spark source DataFrame recovery failed."
)

print("Fresh DataFrame access: PASS")
print("Spark Connect recovery: PASS")


# COMMAND ----------


# Step 15.25: Reusable parameterized batch inference

from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

print("=" * 65)
print("REUSABLE PARAMETERIZED BATCH INFERENCE")
print("=" * 65)

# --------------------------------------------------
# 1. Define reusable inference function
# --------------------------------------------------

def run_batch_inference(
    source_df,
    preprocessing_pipeline,
    registered_model,
    threshold=0.50
):
    """
    Apply preprocessing and a registered Spark ML model
    to customer records.

    Returns:
        DataFrame containing customer predictions.
    """

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Classification threshold must be between 0 and 1."
        )

    # Apply the fitted preprocessing pipeline
    transformed_df = preprocessing_pipeline.transform(
        source_df
    )

    # Generate model predictions and probabilities
    model_predictions_df = registered_model.transform(
        transformed_df
    )

    # Extract positive-class probability and apply
    # the configured business classification threshold
    results_df = (
        model_predictions_df
        .withColumn(
            "churn_probability",
            vector_to_array("probability")[1]
        )
        .withColumn(
            "predicted_class",
            F.when(
                F.col("churn_probability") >= F.lit(threshold),
                F.lit(1)
            ).otherwise(F.lit(0))
        )
        .withColumn(
            "predicted_churn",
            F.when(
                F.col("predicted_class") == 1,
                F.lit("Yes")
            ).otherwise(F.lit("No"))
        )
        .select(
            "customerID",
            "churn_probability",
            "predicted_class",
            "predicted_churn"
        )
    )

    return results_df


print("\nReusable inference function: DEFINED")

# --------------------------------------------------
# 2. Load fresh source sample
# --------------------------------------------------

test_source_df = (
    spark.table(JOB_SOURCE_TABLE)
    .limit(10)
)

# --------------------------------------------------
# 3. Execute reusable inference function
# --------------------------------------------------

test_predictions_df = run_batch_inference(
    source_df=test_source_df,
    preprocessing_pipeline=job_preprocessing_pipeline,
    registered_model=job_registered_model,
    threshold=JOB_THRESHOLD
)

# --------------------------------------------------
# 4. Validate inference results
# --------------------------------------------------

test_prediction_count = test_predictions_df.count()

assert test_prediction_count == 10

invalid_probability_count = (
    test_predictions_df
    .filter(
        F.col("churn_probability").isNull()
        | F.isnan("churn_probability")
        | (F.col("churn_probability") < 0)
        | (F.col("churn_probability") > 1)
    )
    .count()
)

assert invalid_probability_count == 0

invalid_class_count = (
    test_predictions_df
    .filter(
        F.col("predicted_class").isNull()
        | ~F.col("predicted_class").isin(0, 1)
    )
    .count()
)

assert invalid_class_count == 0

# --------------------------------------------------
# 5. Validate configured threshold consistency
# --------------------------------------------------

threshold_mismatch_count = (
    test_predictions_df
    .filter(
        (
            (F.col("churn_probability") >= JOB_THRESHOLD)
            & (F.col("predicted_class") != 1)
        )
        |
        (
            (F.col("churn_probability") < JOB_THRESHOLD)
            & (F.col("predicted_class") != 0)
        )
    )
    .count()
)

assert threshold_mismatch_count == 0

# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\nPARAMETERIZED INFERENCE VALIDATION")
print("Prediction records:", test_prediction_count)
print("Invalid probabilities:", invalid_probability_count)
print("Invalid classes:", invalid_class_count)
print("Threshold mismatches:", threshold_mismatch_count)
print("Configured threshold:", JOB_THRESHOLD)

print("\nSAMPLE CUSTOMER PREDICTIONS")

test_predictions_df.orderBy(
    F.desc("churn_probability")
).show(10, truncate=False)

print("\nREUSABLE BATCH INFERENCE FUNCTION: PASS")


# COMMAND ----------


# Step 15.26: Add production metadata to inference output

from pyspark.sql import functions as F

print("=" * 65)
print("PARAMETERIZED BATCH INFERENCE METADATA")
print("=" * 65)

# --------------------------------------------------
# 1. Define reusable metadata function
# --------------------------------------------------

def add_batch_metadata(
    predictions_df,
    batch_id,
    model_name,
    model_version,
    threshold,
    source_table
):
    """
    Attach batch-level metadata to model predictions.

    Returns:
        Spark DataFrame with predictions and metadata.
    """

    if not batch_id or not batch_id.strip():
        raise ValueError("Batch ID cannot be empty.")

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Classification threshold must be between 0 and 1."
        )

    enriched_df = (
        predictions_df
        .withColumn(
            "batch_id",
            F.lit(batch_id)
        )
        .withColumn(
            "prediction_timestamp",
            F.current_timestamp()
        )
        .withColumn(
            "model_name",
            F.lit(model_name)
        )
        .withColumn(
            "model_version",
            F.lit(model_version)
        )
        .withColumn(
            "classification_threshold",
            F.lit(float(threshold))
        )
        .withColumn(
            "source_table",
            F.lit(source_table)
        )
    )

    return enriched_df


print("\nBatch metadata function: DEFINED")

# --------------------------------------------------
# 2. Generate a fresh sample prediction DataFrame
# --------------------------------------------------

metadata_source_df = (
    spark.table(JOB_SOURCE_TABLE)
    .limit(10)
)

metadata_predictions_df = run_batch_inference(
    source_df=metadata_source_df,
    preprocessing_pipeline=job_preprocessing_pipeline,
    registered_model=job_registered_model,
    threshold=JOB_THRESHOLD
)

# --------------------------------------------------
# 3. Attach configured job metadata
# --------------------------------------------------

metadata_enriched_df = add_batch_metadata(
    predictions_df=metadata_predictions_df,
    batch_id=JOB_BATCH_ID,
    model_name=JOB_MODEL_NAME,
    model_version=JOB_MODEL_VERSION,
    threshold=JOB_THRESHOLD,
    source_table=JOB_SOURCE_TABLE
)

# --------------------------------------------------
# 4. Validate required metadata columns
# --------------------------------------------------

required_metadata_columns = [
    "batch_id",
    "prediction_timestamp",
    "model_name",
    "model_version",
    "classification_threshold",
    "source_table"
]

for column_name in required_metadata_columns:
    assert column_name in metadata_enriched_df.columns, (
        f"Missing metadata column: {column_name}"
    )

# --------------------------------------------------
# 5. Validate metadata completeness
# --------------------------------------------------

metadata_null_condition = None

for column_name in required_metadata_columns:

    condition = F.col(column_name).isNull()

    metadata_null_condition = (
        condition
        if metadata_null_condition is None
        else metadata_null_condition | condition
    )

missing_metadata_count = (
    metadata_enriched_df
    .filter(metadata_null_condition)
    .count()
)

assert missing_metadata_count == 0

# --------------------------------------------------
# 6. Validate configured metadata values
# --------------------------------------------------

metadata_mismatch_count = (
    metadata_enriched_df
    .filter(
        (F.col("batch_id") != JOB_BATCH_ID)
        | (F.col("model_name") != JOB_MODEL_NAME)
        | (F.col("model_version") != JOB_MODEL_VERSION)
        | (
            F.col("classification_threshold")
            != JOB_THRESHOLD
        )
        | (F.col("source_table") != JOB_SOURCE_TABLE)
    )
    .count()
)

assert metadata_mismatch_count == 0

# --------------------------------------------------
# 7. Validate record count and uniqueness
# --------------------------------------------------

metadata_record_count = metadata_enriched_df.count()

unique_metadata_customers = (
    metadata_enriched_df
    .select("customerID")
    .distinct()
    .count()
)

assert metadata_record_count == 10
assert unique_metadata_customers == metadata_record_count

# --------------------------------------------------
# 8. Display validation results
# --------------------------------------------------

print("\nBATCH METADATA VALIDATION")
print("Prediction records:", metadata_record_count)
print("Unique customers:", unique_metadata_customers)
print("Missing metadata records:", missing_metadata_count)
print("Metadata mismatches:", metadata_mismatch_count)
print("Metadata columns:", len(required_metadata_columns))

print("\nENRICHED PREDICTION SAMPLE")

metadata_enriched_df.select(
    "customerID",
    "churn_probability",
    "predicted_churn",
    "batch_id",
    "model_version",
    "classification_threshold",
    "prediction_timestamp"
).orderBy(
    F.desc("churn_probability")
).show(10, truncate=False)

print("\nPARAMETERIZED BATCH METADATA: PASS")


# COMMAND ----------


# Step 15.27: Reusable batch prediction quality validation

from pyspark.sql import functions as F

print("=" * 65)
print("REUSABLE BATCH PREDICTION QUALITY VALIDATION")
print("=" * 65)


# --------------------------------------------------
# 1. Define reusable quality validation function
# --------------------------------------------------

def validate_batch_predictions(
    predictions_df,
    expected_count,
    batch_id,
    model_name,
    model_version,
    threshold,
    source_table
):
    """
    Validate a batch of customer churn predictions.

    Returns:
        Dictionary containing quality gate results.
    """

    required_columns = [
        "customerID",
        "churn_probability",
        "predicted_class",
        "predicted_churn",
        "batch_id",
        "prediction_timestamp",
        "model_name",
        "model_version",
        "classification_threshold",
        "source_table"
    ]

    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in predictions_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ----------------------------------------------
    # Gate 1: Record completeness
    # ----------------------------------------------

    record_count = predictions_df.count()

    record_completeness = (
        record_count == expected_count
        and record_count > 0
    )

    # ----------------------------------------------
    # Gate 2: Customer uniqueness
    # ----------------------------------------------

    unique_customers = (
        predictions_df
        .select("customerID")
        .distinct()
        .count()
    )

    customer_uniqueness = (
        unique_customers == record_count
    )

    # ----------------------------------------------
    # Gate 3: Probability validity
    # ----------------------------------------------

    invalid_probabilities = (
        predictions_df
        .filter(
            F.col("churn_probability").isNull()
            | F.isnan("churn_probability")
            | (F.col("churn_probability") < 0)
            | (F.col("churn_probability") > 1)
        )
        .count()
    )

    probability_validity = (
        invalid_probabilities == 0
    )

    # ----------------------------------------------
    # Gate 4: Metadata completeness
    # ----------------------------------------------

    metadata_columns = [
        "customerID",
        "batch_id",
        "prediction_timestamp",
        "model_name",
        "model_version",
        "classification_threshold",
        "source_table"
    ]

    missing_condition = None

    for column_name in metadata_columns:

        condition = F.col(column_name).isNull()

        missing_condition = (
            condition
            if missing_condition is None
            else missing_condition | condition
        )

    missing_metadata_records = (
        predictions_df
        .filter(missing_condition)
        .count()
    )

    metadata_completeness = (
        missing_metadata_records == 0
    )

    # ----------------------------------------------
    # Gate 5: Classification consistency
    # ----------------------------------------------

    classification_mismatches = (
        predictions_df
        .filter(
            F.col("predicted_class").isNull()
            | F.col("predicted_churn").isNull()
            | ~F.col("predicted_class").isin(0, 1)
            | (
                (F.col("churn_probability") >= threshold)
                & (F.col("predicted_class") != 1)
            )
            | (
                (F.col("churn_probability") < threshold)
                & (F.col("predicted_class") != 0)
            )
            | (
                (F.col("predicted_class") == 1)
                & (F.col("predicted_churn") != "Yes")
            )
            | (
                (F.col("predicted_class") == 0)
                & (F.col("predicted_churn") != "No")
            )
        )
        .count()
    )

    classification_consistency = (
        classification_mismatches == 0
    )

    # ----------------------------------------------
    # Gate 6: Batch metadata consistency
    # ----------------------------------------------

    metadata_mismatches = (
        predictions_df
        .filter(
            (F.col("batch_id") != batch_id)
            | (F.col("model_name") != model_name)
            | (F.col("model_version") != model_version)
            | (
                F.col("classification_threshold")
                != threshold
            )
            | (F.col("source_table") != source_table)
        )
        .count()
    )

    batch_metadata_consistency = (
        metadata_mismatches == 0
    )

    # ----------------------------------------------
    # Assemble quality gate results
    # ----------------------------------------------

    quality_gates = {
        "Record completeness": record_completeness,
        "Customer uniqueness": customer_uniqueness,
        "Probability validity": probability_validity,
        "Metadata completeness": metadata_completeness,
        "Classification consistency": classification_consistency,
        "Batch metadata consistency": batch_metadata_consistency
    }

    passed_gates = sum(quality_gates.values())
    total_gates = len(quality_gates)
    failed_gates = total_gates - passed_gates

    overall_status = (
        "PASS" if failed_gates == 0 else "FAIL"
    )

    return {
        "batch_id": batch_id,
        "record_count": record_count,
        "unique_customers": unique_customers,
        "quality_gates": quality_gates,
        "total_gates": total_gates,
        "passed_gates": passed_gates,
        "failed_gates": failed_gates,
        "overall_status": overall_status
    }


print("\nReusable quality validation function: DEFINED")


# --------------------------------------------------
# 2. Validate current sample batch
# --------------------------------------------------

job_quality_results = validate_batch_predictions(
    predictions_df=metadata_enriched_df,
    expected_count=10,
    batch_id=JOB_BATCH_ID,
    model_name=JOB_MODEL_NAME,
    model_version=JOB_MODEL_VERSION,
    threshold=JOB_THRESHOLD,
    source_table=JOB_SOURCE_TABLE
)


# --------------------------------------------------
# 3. Display quality gate results
# --------------------------------------------------

print("\nQUALITY GATE RESULTS")

for gate_name, passed in (
    job_quality_results["quality_gates"].items()
):

    status = "PASS" if passed else "FAIL"

    print(f"{gate_name}: {status}")


# --------------------------------------------------
# 4. Display validation summary
# --------------------------------------------------

print("\nBATCH QUALITY SUMMARY")

print(
    "Batch ID:",
    job_quality_results["batch_id"]
)

print(
    "Prediction records:",
    job_quality_results["record_count"]
)

print(
    "Unique customers:",
    job_quality_results["unique_customers"]
)

print(
    "Total quality gates:",
    job_quality_results["total_gates"]
)

print(
    "Passed gates:",
    job_quality_results["passed_gates"]
)

print(
    "Failed gates:",
    job_quality_results["failed_gates"]
)

print(
    "Overall quality status:",
    job_quality_results["overall_status"]
)


# --------------------------------------------------
# 5. Enforce quality gate approval
# --------------------------------------------------

assert job_quality_results["overall_status"] == "PASS", (
    "Batch prediction quality validation failed."
)

print("\nREUSABLE BATCH QUALITY VALIDATION: PASS")


# COMMAND ----------

# DBTITLE 1,Step 15.28: Reusable quality-gated Delta publication

# Step 15.28: Reusable quality-gated Delta publication

from delta.tables import DeltaTable
from pyspark.sql import functions as F
from datetime import datetime, timezone

print("=" * 65)
print("REUSABLE QUALITY-GATED DELTA PUBLICATION")
print("=" * 65)

# --------------------------------------------------
# 1. Define development publication destination
# --------------------------------------------------

JOB_DEV_OUTPUT_TABLE = (
    "workspace.default.telco_15_job_dev_predictions"
)

print("\nDevelopment output table:", JOB_DEV_OUTPUT_TABLE)


# --------------------------------------------------
# 2. Define reusable publication function
# --------------------------------------------------

def publish_quality_approved_batch(
    predictions_df,
    quality_results,
    destination_table
):
    """
    Publish an approved prediction batch to Delta.

    Quality validation must pass before publication.

    Uses an idempotent MERGE on batch_id + customerID.
    """

    # ----------------------------------------------
    # A. Enforce quality approval
    # ----------------------------------------------

    if quality_results["overall_status"] != "PASS":
        raise ValueError(
            "Publication blocked: batch quality validation failed."
        )

    if quality_results["failed_gates"] != 0:
        raise ValueError(
            "Publication blocked: one or more quality gates failed."
        )

    if quality_results["passed_gates"] != quality_results["total_gates"]:
        raise ValueError(
            "Publication blocked: incomplete quality approval."
        )

    # Confirm approval applies to this DataFrame's batch
    batch_ids = [
        row["batch_id"]
        for row in predictions_df
        .select("batch_id")
        .distinct()
        .collect()
    ]

    if batch_ids != [quality_results["batch_id"]]:
        raise ValueError(
            "Publication blocked: batch ID does not match approval."
        )

    current_count = predictions_df.count()

    if current_count != quality_results["record_count"]:
        raise ValueError(
            "Publication blocked: record count changed after validation."
        )

    # ----------------------------------------------
    # B. Add publication timestamp
    # ----------------------------------------------

    publication_time = datetime.now(timezone.utc)

    publish_df = predictions_df.withColumn(
        "publication_timestamp",
        F.lit(publication_time).cast("timestamp")
    )

    # ----------------------------------------------
    # C. Create destination table if necessary
    # ----------------------------------------------

    if not spark.catalog.tableExists(destination_table):

        (
            publish_df
            .limit(0)
            .write
            .format("delta")
            .mode("error")
            .saveAsTable(destination_table)
        )

        print("Development Delta table created.")

    else:
        print("Development Delta table already exists.")

    # ----------------------------------------------
    # D. Publish with idempotent Delta MERGE
    # ----------------------------------------------

    delta_target = DeltaTable.forName(
        spark,
        destination_table
    )

    (
        delta_target.alias("target")
        .merge(
            publish_df.alias("source"),
            """
            target.batch_id = source.batch_id
            AND target.customerID = source.customerID
            """
        )
        .whenNotMatchedInsertAll()
        .execute()
    )

    # ----------------------------------------------
    # E. Verify published records
    # ----------------------------------------------

    published_df = (
        spark.table(destination_table)
        .filter(
            F.col("batch_id") == quality_results["batch_id"]
        )
    )

    published_count = published_df.count()

    unique_keys = (
        published_df
        .select("batch_id", "customerID")
        .distinct()
        .count()
    )

    if published_count != current_count:
        raise RuntimeError(
            "Published record count does not match approved count."
        )

    if unique_keys != published_count:
        raise RuntimeError(
            "Duplicate prediction keys detected after publication."
        )

    return {
        "destination_table": destination_table,
        "batch_id": quality_results["batch_id"],
        "published_count": published_count,
        "unique_keys": unique_keys,
        "publication_status": "PASS"
    }


print("\nReusable publication function: DEFINED")


# --------------------------------------------------
# 3. Publish approved development sample
# --------------------------------------------------

job_publication_results = publish_quality_approved_batch(
    predictions_df=metadata_enriched_df,
    quality_results=job_quality_results,
    destination_table=JOB_DEV_OUTPUT_TABLE
)


# --------------------------------------------------
# 4. Display publication results
# --------------------------------------------------

print("\nQUALITY-GATED PUBLICATION RESULTS")

print(
    "Destination:",
    job_publication_results["destination_table"]
)

print(
    "Batch ID:",
    job_publication_results["batch_id"]
)

print(
    "Published records:",
    job_publication_results["published_count"]
)

print(
    "Unique prediction keys:",
    job_publication_results["unique_keys"]
)

print(
    "Publication status:",
    job_publication_results["publication_status"]
)


# --------------------------------------------------
# 5. Validate final publication
# --------------------------------------------------

assert job_publication_results["published_count"] == 10

assert job_publication_results["unique_keys"] == 10

assert job_publication_results["publication_status"] == "PASS"

print("\nREUSABLE QUALITY-GATED PUBLICATION: PASS")


# COMMAND ----------


# Step 15.29: Publication idempotency and failure protection

from pyspark.sql import functions as F

print("=" * 65)
print("PUBLICATION IDEMPOTENCY AND FAILURE PROTECTION")
print("=" * 65)

# --------------------------------------------------
# 1. Capture development table state before retry
# --------------------------------------------------

before_retry_df = (
    spark.table(JOB_DEV_OUTPUT_TABLE)
    .filter(F.col("batch_id") == JOB_BATCH_ID)
)

before_count = before_retry_df.count()

before_timestamps = {
    row["customerID"]: row["publication_timestamp"]
    for row in before_retry_df.select(
        "customerID",
        "publication_timestamp"
    ).collect()
}

assert before_count == 10

print("\nBEFORE RETRY")
print("Published records:", before_count)

# --------------------------------------------------
# 2. Retry publication using the same batch
# --------------------------------------------------

retry_results = publish_quality_approved_batch(
    predictions_df=metadata_enriched_df,
    quality_results=job_quality_results,
    destination_table=JOB_DEV_OUTPUT_TABLE
)

# --------------------------------------------------
# 3. Verify record count and uniqueness
# --------------------------------------------------

after_retry_df = (
    spark.table(JOB_DEV_OUTPUT_TABLE)
    .filter(F.col("batch_id") == JOB_BATCH_ID)
)

after_count = after_retry_df.count()

unique_keys = (
    after_retry_df
    .select("batch_id", "customerID")
    .distinct()
    .count()
)

duplicate_keys = after_count - unique_keys

assert after_count == before_count
assert duplicate_keys == 0

print("\nAFTER RETRY")
print("Published records:", after_count)
print("Duplicate prediction keys:", duplicate_keys)

# --------------------------------------------------
# 4. Verify publication timestamps are unchanged
# --------------------------------------------------

after_timestamps = {
    row["customerID"]: row["publication_timestamp"]
    for row in after_retry_df.select(
        "customerID",
        "publication_timestamp"
    ).collect()
}

timestamps_unchanged = (
    before_timestamps == after_timestamps
)

assert timestamps_unchanged

print(
    "Publication timestamps unchanged:",
    timestamps_unchanged
)

print("IDEMPOTENT RETRY: PASS")

# --------------------------------------------------
# 5. Simulate a failed quality approval
# --------------------------------------------------

failed_quality_results = {
    **job_quality_results,
    "overall_status": "FAIL",
    "failed_gates": 1,
    "passed_gates": 5
}

publication_blocked = False

try:

    publish_quality_approved_batch(
        predictions_df=metadata_enriched_df,
        quality_results=failed_quality_results,
        destination_table=JOB_DEV_OUTPUT_TABLE
    )

except ValueError as error:

    publication_blocked = True

    print("\nEXPECTED PUBLICATION REJECTION")
    print(str(error))

assert publication_blocked

print("FAILED QUALITY PUBLICATION BLOCKED: PASS")

# --------------------------------------------------
# 6. Verify failed publication caused no changes
# --------------------------------------------------

final_count = (
    spark.table(JOB_DEV_OUTPUT_TABLE)
    .filter(F.col("batch_id") == JOB_BATCH_ID)
    .count()
)

assert final_count == before_count

print("\nFINAL VALIDATION SUMMARY")
print("Original records:", before_count)
print("Records after retry:", after_count)
print("Records after rejected publication:", final_count)
print("Duplicate keys:", duplicate_keys)
print("Timestamps unchanged:", timestamps_unchanged)
print("Failed publication blocked:", publication_blocked)

print(
    "\nPUBLICATION IDEMPOTENCY AND FAILURE PROTECTION: PASS"
)


# COMMAND ----------


# Step 15.30: Define end-to-end parameterized job orchestration

print("=" * 65)
print("END-TO-END PARAMETERIZED JOB ORCHESTRATION")
print("=" * 65)


def execute_telco_batch_job(
    source_table,
    preprocessing_pipeline,
    registered_model,
    model_name,
    model_version,
    threshold,
    batch_id,
    destination_table
):
    """
    Execute the Telco churn batch inference workflow.

    Stages:
      1. Load source data
      2. Generate predictions
      3. Attach batch metadata
      4. Validate prediction quality
      5. Publish approved predictions

    Returns:
        Dictionary containing execution results.
    """

    # ----------------------------------------------
    # 1. Validate job configuration
    # ----------------------------------------------

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid classification threshold.")

    if not batch_id or not batch_id.strip():
        raise ValueError("Batch ID cannot be empty.")

    if not spark.catalog.tableExists(source_table):
        raise ValueError(
            f"Source table does not exist: {source_table}"
        )

    if not destination_table.startswith(
        "workspace.default."
    ):
        raise ValueError(
            "Destination must be in workspace.default."
        )

    print("\nSTAGE 1: LOAD SOURCE DATA")

    source_df = spark.table(source_table)

    if "customerID" not in source_df.columns:
        raise ValueError("Missing customerID column.")

    # Exclude historical target columns from inference.
    # The saved preprocessing pipeline uses customer
    # features, not the observed churn outcome.
    source_df = source_df.drop("Churn", "label")

    source_count = source_df.count()

    if source_count == 0:
        raise ValueError("Source table contains no records.")

    print("Source records:", source_count)

    # ----------------------------------------------
    # 2. Generate predictions
    # ----------------------------------------------

    print("\nSTAGE 2: GENERATE PREDICTIONS")

    predictions_df = run_batch_inference(
        source_df=source_df,
        preprocessing_pipeline=preprocessing_pipeline,
        registered_model=registered_model,
        threshold=threshold
    )

    # ----------------------------------------------
    # 3. Attach batch metadata
    # ----------------------------------------------

    print("\nSTAGE 3: ATTACH BATCH METADATA")

    enriched_df = add_batch_metadata(
        predictions_df=predictions_df,
        batch_id=batch_id,
        model_name=model_name,
        model_version=model_version,
        threshold=threshold,
        source_table=source_table
    )

    # ----------------------------------------------
    # 4. Validate prediction quality
    # ----------------------------------------------

    print("\nSTAGE 4: VALIDATE PREDICTION QUALITY")

    quality_results = validate_batch_predictions(
        predictions_df=enriched_df,
        expected_count=source_count,
        batch_id=batch_id,
        model_name=model_name,
        model_version=model_version,
        threshold=threshold,
        source_table=source_table
    )

    print(
        "Quality gates passed:",
        quality_results["passed_gates"],
        "/",
        quality_results["total_gates"]
    )

    if quality_results["overall_status"] != "PASS":
        raise RuntimeError(
            "Batch quality validation failed. "
            "Publication has been blocked."
        )

    # ----------------------------------------------
    # 5. Publish approved predictions
    # ----------------------------------------------

    print("\nSTAGE 5: PUBLISH APPROVED PREDICTIONS")

    publication_results = publish_quality_approved_batch(
        predictions_df=enriched_df,
        quality_results=quality_results,
        destination_table=destination_table
    )

    # ----------------------------------------------
    # 6. Return execution summary
    # ----------------------------------------------

    return {
        "batch_id": batch_id,
        "source_table": source_table,
        "source_count": source_count,
        "model_name": model_name,
        "model_version": model_version,
        "threshold": threshold,
        "quality_status": quality_results["overall_status"],
        "passed_gates": quality_results["passed_gates"],
        "published_count": publication_results["published_count"],
        "destination_table": destination_table,
        "execution_status": "PASS"
    }


print("\nEnd-to-end orchestration function: DEFINED")

# --------------------------------------------------
# Verify required components are available
# --------------------------------------------------

required_functions = [
    "run_batch_inference",
    "add_batch_metadata",
    "validate_batch_predictions",
    "publish_quality_approved_batch",
    "execute_telco_batch_job"
]

missing_functions = [
    name for name in required_functions
    if name not in globals() or not callable(globals()[name])
]

assert not missing_functions, (
    f"Missing workflow functions: {missing_functions}"
)

assert JOB_THRESHOLD >= 0 and JOB_THRESHOLD <= 1
assert JOB_BATCH_ID
assert spark.catalog.tableExists(JOB_SOURCE_TABLE)

print("\nORCHESTRATION READINESS VALIDATION")
print("Required functions:", len(required_functions))
print("Missing functions:", len(missing_functions))
print("Source table:", JOB_SOURCE_TABLE)
print("Model:", JOB_MODEL_NAME)
print("Model version:", JOB_MODEL_VERSION)
print("Classification threshold:", JOB_THRESHOLD)
print("Destination:", JOB_DEV_OUTPUT_TABLE)

print("\nEND-TO-END ORCHESTRATION DEFINITION: PASS")


# COMMAND ----------


# Step 15.31: Execute end-to-end parameterized batch job

from pyspark.sql import functions as F

print("=" * 65)
print("END-TO-END TELCO BATCH JOB EXECUTION")
print("=" * 65)

# --------------------------------------------------
# 1. Execute the complete orchestration function
# --------------------------------------------------

job_execution_results = execute_telco_batch_job(
    source_table=JOB_SOURCE_TABLE,
    preprocessing_pipeline=job_preprocessing_pipeline,
    registered_model=job_registered_model,
    model_name=JOB_MODEL_NAME,
    model_version=JOB_MODEL_VERSION,
    threshold=JOB_THRESHOLD,
    batch_id=JOB_BATCH_ID,
    destination_table=JOB_DEV_OUTPUT_TABLE
)

# --------------------------------------------------
# 2. Display execution summary
# --------------------------------------------------

print("\nEND-TO-END JOB EXECUTION SUMMARY")

for key, value in job_execution_results.items():
    print(f"{key}: {value}")

# --------------------------------------------------
# 3. Independently verify published records
# --------------------------------------------------

verified_batch_df = (
    spark.table(JOB_DEV_OUTPUT_TABLE)
    .filter(F.col("batch_id") == JOB_BATCH_ID)
)

verified_count = verified_batch_df.count()

verified_unique_customers = (
    verified_batch_df
    .select("customerID")
    .distinct()
    .count()
)

verified_churners = (
    verified_batch_df
    .filter(F.col("predicted_churn") == "Yes")
    .count()
)

verified_non_churners = (
    verified_batch_df
    .filter(F.col("predicted_churn") == "No")
    .count()
)

# --------------------------------------------------
# 4. Enforce final reconciliation
# --------------------------------------------------

assert job_execution_results["execution_status"] == "PASS"

assert job_execution_results["quality_status"] == "PASS"

assert job_execution_results["passed_gates"] == 6

assert verified_count == job_execution_results["source_count"]

assert verified_unique_customers == verified_count

assert verified_churners + verified_non_churners == verified_count

# --------------------------------------------------
# 5. Display final verification
# --------------------------------------------------

print("\nFINAL DELTA PUBLICATION VERIFICATION")

print("Published records:", verified_count)
print("Unique customers:", verified_unique_customers)
print("Predicted churners:", verified_churners)
print("Predicted non-churners:", verified_non_churners)
print("Duplicate customers:", verified_count - verified_unique_customers)

print("\nEND-TO-END PARAMETERIZED BATCH JOB: PASS")


# COMMAND ----------


# Step 15.32: Audit batch consistency before scheduling

from pyspark.sql import functions as F

print("=" * 65)
print("PRODUCTION BATCH CONSISTENCY AUDIT")
print("=" * 65)

audit_df = (
    spark.table(JOB_DEV_OUTPUT_TABLE)
    .filter(F.col("batch_id") == JOB_BATCH_ID)
)

# 1. Record and customer checks
record_count = audit_df.count()

unique_customers = (
    audit_df.select("customerID").distinct().count()
)

invalid_customers = (
    audit_df.filter(
        F.col("customerID").isNull()
        | (F.trim(F.col("customerID")) == "")
    ).count()
)

duplicate_keys = record_count - unique_customers

# 2. Metadata consistency checks
metadata_summary = audit_df.agg(
    F.countDistinct("prediction_timestamp").alias(
        "prediction_timestamps"
    ),
    F.countDistinct("model_name").alias("model_names"),
    F.countDistinct("model_version").alias("model_versions"),
    F.countDistinct("classification_threshold").alias(
        "thresholds"
    ),
    F.countDistinct("source_table").alias("source_tables"),
    F.countDistinct("publication_timestamp").alias(
        "publication_timestamps"
    )
).first()

# 3. Validate expected configuration values
configuration_mismatches = audit_df.filter(
    ~F.col("model_name").eqNullSafe(F.lit(JOB_MODEL_NAME))
    | ~F.col("model_version").eqNullSafe(
        F.lit(JOB_MODEL_VERSION)
    )
    | ~F.col("classification_threshold").eqNullSafe(
        F.lit(float(JOB_THRESHOLD))
    )
    | ~F.col("source_table").eqNullSafe(
        F.lit(JOB_SOURCE_TABLE)
    )
).count()

# 4. Check for missing timestamps
missing_timestamps = audit_df.filter(
    F.col("prediction_timestamp").isNull()
    | F.col("publication_timestamp").isNull()
).count()

# 5. Display audit findings
print("\nRECORD INTEGRITY")
print("Published records:", record_count)
print("Unique customers:", unique_customers)
print("Duplicate keys:", duplicate_keys)
print("Invalid customer IDs:", invalid_customers)

print("\nMETADATA CONSISTENCY")
print(
    "Distinct prediction timestamps:",
    metadata_summary["prediction_timestamps"]
)
print(
    "Distinct publication timestamps:",
    metadata_summary["publication_timestamps"]
)
print("Distinct model names:", metadata_summary["model_names"])
print("Distinct model versions:", metadata_summary["model_versions"])
print("Distinct thresholds:", metadata_summary["thresholds"])
print("Distinct source tables:", metadata_summary["source_tables"])
print("Configuration mismatches:", configuration_mismatches)
print("Missing timestamps:", missing_timestamps)

# 6. Enforce fundamental integrity checks
assert record_count == 1409
assert duplicate_keys == 0
assert invalid_customers == 0
assert configuration_mismatches == 0
assert missing_timestamps == 0

print("\nFUNDAMENTAL BATCH INTEGRITY: PASS")

# Timestamp counts are diagnostic in this step.
# We will not assert they equal 1 because the
# development batch was published in two executions.

if (
    metadata_summary["prediction_timestamps"] == 1
    and metadata_summary["publication_timestamps"] == 1
):
    print("BATCH TIMESTAMP CONSISTENCY: PASS")
else:
    print("BATCH TIMESTAMP CONSISTENCY: NEEDS HARDENING")

print("\nBATCH CONSISTENCY AUDIT: COMPLETE")


# COMMAND ----------

F.current_timestamp()

# COMMAND ----------


# Step 15.33: Stable batch timestamp implementation

from datetime import datetime, timezone
from pyspark.sql import functions as F

print("=" * 65)
print("STABLE BATCH TIMESTAMP IMPLEMENTATION")
print("=" * 65)


# --------------------------------------------------
# 1. Define improved metadata function
# --------------------------------------------------

def add_stable_batch_metadata(
    predictions_df,
    batch_id,
    model_name,
    model_version,
    threshold,
    source_table,
    batch_timestamp
):
    """
    Attach stable metadata to prediction records.

    All predictions in the batch receive the same
    explicitly supplied timestamp.
    """

    if not batch_id or not batch_id.strip():
        raise ValueError("Batch ID cannot be empty.")

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid classification threshold.")

    if batch_timestamp is None:
        raise ValueError("Batch timestamp cannot be None.")

    enriched_df = (
        predictions_df
        .withColumn(
            "batch_id",
            F.lit(batch_id)
        )
        .withColumn(
            "prediction_timestamp",
            F.lit(batch_timestamp).cast("timestamp")
        )
        .withColumn(
            "model_name",
            F.lit(model_name)
        )
        .withColumn(
            "model_version",
            F.lit(model_version)
        )
        .withColumn(
            "classification_threshold",
            F.lit(float(threshold))
        )
        .withColumn(
            "source_table",
            F.lit(source_table)
        )
    )

    return enriched_df


print("\nStable metadata function: DEFINED")


# --------------------------------------------------
# 2. Generate one timestamp for the sample batch
# --------------------------------------------------

STABLE_TEST_TIMESTAMP = datetime.now(timezone.utc)

print("Stable batch timestamp:", STABLE_TEST_TIMESTAMP)


# --------------------------------------------------
# 3. Create sample predictions
# --------------------------------------------------

stable_test_source_df = (
    spark.table(JOB_SOURCE_TABLE)
    .limit(10)
)

stable_test_predictions_df = run_batch_inference(
    source_df=stable_test_source_df,
    preprocessing_pipeline=job_preprocessing_pipeline,
    registered_model=job_registered_model,
    threshold=JOB_THRESHOLD
)


# --------------------------------------------------
# 4. Attach stable metadata
# --------------------------------------------------

stable_test_enriched_df = add_stable_batch_metadata(
    predictions_df=stable_test_predictions_df,
    batch_id=JOB_BATCH_ID,
    model_name=JOB_MODEL_NAME,
    model_version=JOB_MODEL_VERSION,
    threshold=JOB_THRESHOLD,
    source_table=JOB_SOURCE_TABLE,
    batch_timestamp=STABLE_TEST_TIMESTAMP
)


# --------------------------------------------------
# 5. Validate timestamp consistency
# --------------------------------------------------

timestamp_validation = stable_test_enriched_df.agg(
    F.count("*").alias("record_count"),
    F.countDistinct("prediction_timestamp").alias(
        "distinct_timestamps"
    ),
    F.min("prediction_timestamp").alias(
        "minimum_timestamp"
    ),
    F.max("prediction_timestamp").alias(
        "maximum_timestamp"
    ),
    F.sum(
        F.when(
            F.col("prediction_timestamp").isNull(), 1
        ).otherwise(0)
    ).alias("missing_timestamps")
).first()


# --------------------------------------------------
# 6. Enforce validation
# --------------------------------------------------

assert timestamp_validation["record_count"] == 10

assert timestamp_validation["distinct_timestamps"] == 1

assert timestamp_validation["missing_timestamps"] == 0

assert (
    timestamp_validation["minimum_timestamp"]
    == timestamp_validation["maximum_timestamp"]
)


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print("\nSTABLE TIMESTAMP VALIDATION")

print(
    "Prediction records:",
    timestamp_validation["record_count"]
)

print(
    "Distinct timestamps:",
    timestamp_validation["distinct_timestamps"]
)

print(
    "Missing timestamps:",
    timestamp_validation["missing_timestamps"]
)

print(
    "Minimum timestamp:",
    timestamp_validation["minimum_timestamp"]
)

print(
    "Maximum timestamp:",
    timestamp_validation["maximum_timestamp"]
)

print("\nSTABLE BATCH TIMESTAMP VALIDATION: PASS")


# COMMAND ----------


# Step 15.34: Deterministic source batch fingerprint

import hashlib
import json
from pyspark.sql import functions as F

print("=" * 65)
print("SOURCE BATCH IDENTITY AND FINGERPRINT VALIDATION")
print("=" * 65)


def calculate_source_fingerprint(source_df):
    """
    Calculate a deterministic SHA-256 fingerprint.

    Includes:
      - Input schema
      - Customer IDs
      - Input feature values

    Intended for batch identity validation.
    """

    # Exclude historical labels from inference inputs
    input_df = source_df.drop("Churn", "label")

    if "customerID" not in input_df.columns:
        raise ValueError("Missing customerID column.")

    # Require valid, unique customer IDs
    invalid_ids = input_df.filter(
        F.col("customerID").isNull()
        | (F.trim(F.col("customerID")) == "")
    ).count()

    if invalid_ids:
        raise ValueError("Invalid customer IDs detected.")

    total_count = input_df.count()

    if total_count == 0:
        raise ValueError("Source batch cannot be empty.")

    unique_count = (
        input_df.select("customerID").distinct().count()
    )

    if unique_count != total_count:
        raise ValueError("Duplicate customer IDs detected.")

    # Canonical schema representation
    schema_json = json.dumps(
        sorted(
            [
                (field.name, field.dataType.simpleString())
                for field in input_df.schema.fields
            ]
        ),
        separators=(",", ":")
    )

    # Canonical row representation:
    # sorted columns + deterministic customer ordering
    ordered_columns = sorted(input_df.columns)

    ordered_rows = (
        input_df
        .select(*ordered_columns)
        .orderBy("customerID")
        .toLocalIterator()
    )

    digest = hashlib.sha256()
    digest.update(schema_json.encode("utf-8"))
    digest.update(b"\n")

    for row in ordered_rows:
        row_json = json.dumps(
            row.asDict(recursive=True),
            sort_keys=True,
            separators=(",", ":"),
            default=str
        )
        digest.update(row_json.encode("utf-8"))
        digest.update(b"\n")

    return {
        "record_count": total_count,
        "unique_customers": unique_count,
        "source_fingerprint": digest.hexdigest()
    }


print("\nSource fingerprint function: DEFINED")


# --------------------------------------------------
# 1. Load the original source
# --------------------------------------------------

fingerprint_source_df = spark.table(JOB_SOURCE_TABLE)

# --------------------------------------------------
# 2. Calculate fingerprint twice
# --------------------------------------------------

fingerprint_run_1 = calculate_source_fingerprint(
    fingerprint_source_df
)

fingerprint_run_2 = calculate_source_fingerprint(
    fingerprint_source_df
)

# --------------------------------------------------
# 3. Validate deterministic behavior
# --------------------------------------------------

assert fingerprint_run_1["record_count"] == 1409

assert fingerprint_run_1["unique_customers"] == 1409

assert (
    fingerprint_run_1["source_fingerprint"]
    == fingerprint_run_2["source_fingerprint"]
)

print("\nSOURCE FINGERPRINT VALIDATION")

print(
    "Source records:",
    fingerprint_run_1["record_count"]
)

print(
    "Unique customers:",
    fingerprint_run_1["unique_customers"]
)

print(
    "Fingerprint run 1:",
    fingerprint_run_1["source_fingerprint"]
)

print(
    "Fingerprint run 2:",
    fingerprint_run_2["source_fingerprint"]
)

print(
    "Fingerprints identical:",
    fingerprint_run_1["source_fingerprint"]
    == fingerprint_run_2["source_fingerprint"]
)

print("\nSOURCE FINGERPRINT DETERMINISM: PASS")


# COMMAND ----------

# DBTITLE 1,Step 15.35: Batch identity registry and retry protection

# Step 15.35: Batch identity registry and retry protection

from datetime import datetime, timezone
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType,
    LongType, DoubleType, TimestampType
)

print("=" * 65)
print("BATCH IDENTITY REGISTRY AND RETRY PROTECTION")
print("=" * 65)

JOB_BATCH_REGISTRY_TABLE = (
    "workspace.default.telco_15_job_batch_registry"
)

# --------------------------------------------------
# 1. Define registry schema
# --------------------------------------------------

registry_schema = StructType([
    StructField("batch_id", StringType(), False),
    StructField("source_table", StringType(), False),
    StructField("source_fingerprint", StringType(), False),
    StructField("record_count", LongType(), False),
    StructField("model_name", StringType(), False),
    StructField("model_version", StringType(), False),
    StructField("classification_threshold", DoubleType(), False),
    StructField("batch_timestamp", TimestampType(), False),
    StructField("registered_at", TimestampType(), False)
])

# --------------------------------------------------
# 2. Create registry table if it does not exist
# --------------------------------------------------

if not spark.catalog.tableExists(JOB_BATCH_REGISTRY_TABLE):

    empty_registry_df = spark.createDataFrame(
        [],
        registry_schema
    )

    (
        empty_registry_df.write
        .format("delta")
        .mode("error")
        .saveAsTable(JOB_BATCH_REGISTRY_TABLE)
    )

    print("Batch registry table: CREATED")

else:
    print("Batch registry table: EXISTS")


# --------------------------------------------------
# 3. Define reusable registration function
# --------------------------------------------------

def register_batch_identity(
    batch_id,
    source_table,
    source_fingerprint,
    record_count,
    model_name,
    model_version,
    threshold,
    registry_table
):
    """
    Register a logical inference batch.

    NEW:
        Register previously unseen batch ID.

    RETRY:
        Return original timestamp if all identity
        fields match.

    CONFLICT:
        Reject changed data or configuration.
    """

    if not batch_id or not batch_id.strip():
        raise ValueError("Batch ID cannot be empty.")

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold.")

    existing_rows = (
        spark.table(registry_table)
        .filter(F.col("batch_id") == batch_id)
        .limit(2)
        .collect()
    )

    if len(existing_rows) > 1:
        raise RuntimeError(
            "Duplicate batch registry entries detected."
        )

    identity = {
        "batch_id": batch_id,
        "source_table": source_table,
        "source_fingerprint": source_fingerprint,
        "record_count": int(record_count),
        "model_name": model_name,
        "model_version": str(model_version),
        "classification_threshold": float(threshold)
    }

    # ----------------------------------------------
    # Existing batch: validate retry identity
    # ----------------------------------------------

    if existing_rows:

        existing = existing_rows[0].asDict()

        for field_name, expected_value in identity.items():

            if existing[field_name] != expected_value:
                raise ValueError(
                    "BATCH IDENTITY CONFLICT: "
                    f"{field_name} does not match "
                    "the registered batch."
                )

        stored_timestamp = existing["batch_timestamp"]

        # Spark returns naive datetimes; restore the
        # UTC offset so the retry timestamp compares
        # equal to the one returned at registration
        if (
            stored_timestamp is not None
            and stored_timestamp.tzinfo is None
        ):
            stored_timestamp = stored_timestamp.replace(
                tzinfo=timezone.utc
            )

        return {
            "batch_id": batch_id,
            "registration_status": "RETRY",
            "batch_timestamp": stored_timestamp
        }

    # ----------------------------------------------
    # New batch: create registry entry
    # ----------------------------------------------

    batch_timestamp = datetime.now(timezone.utc)

    registry_record = {
        **identity,
        "batch_timestamp": batch_timestamp,
        "registered_at": batch_timestamp
    }

    new_registry_df = spark.createDataFrame(
        [registry_record],
        schema=registry_schema
    )

    (
        new_registry_df.write
        .format("delta")
        .mode("append")
        .saveAsTable(registry_table)
    )

    return {
        "batch_id": batch_id,
        "registration_status": "NEW",
        "batch_timestamp": batch_timestamp
    }


print("\nBatch registration function: DEFINED")


# --------------------------------------------------
# 4. Register current development batch
# --------------------------------------------------

registration_1 = register_batch_identity(
    batch_id=JOB_BATCH_ID,
    source_table=JOB_SOURCE_TABLE,
    source_fingerprint=fingerprint_run_1[
        "source_fingerprint"
    ],
    record_count=fingerprint_run_1["record_count"],
    model_name=JOB_MODEL_NAME,
    model_version=JOB_MODEL_VERSION,
    threshold=JOB_THRESHOLD,
    registry_table=JOB_BATCH_REGISTRY_TABLE
)

print("\nFIRST REGISTRATION")
print("Status:", registration_1["registration_status"])
print("Batch timestamp:", registration_1["batch_timestamp"])


# --------------------------------------------------
# 5. Retry with identical configuration
# --------------------------------------------------

registration_2 = register_batch_identity(
    batch_id=JOB_BATCH_ID,
    source_table=JOB_SOURCE_TABLE,
    source_fingerprint=fingerprint_run_2[
        "source_fingerprint"
    ],
    record_count=fingerprint_run_2["record_count"],
    model_name=JOB_MODEL_NAME,
    model_version=JOB_MODEL_VERSION,
    threshold=JOB_THRESHOLD,
    registry_table=JOB_BATCH_REGISTRY_TABLE
)

print("\nIDENTICAL RETRY")
print("Status:", registration_2["registration_status"])
print("Batch timestamp:", registration_2["batch_timestamp"])

assert registration_2["registration_status"] == "RETRY"

assert (
    registration_1["batch_timestamp"]
    == registration_2["batch_timestamp"]
)


# --------------------------------------------------
# 6. Simulate a conflicting source fingerprint
# --------------------------------------------------

conflict_blocked = False

try:

    register_batch_identity(
        batch_id=JOB_BATCH_ID,
        source_table=JOB_SOURCE_TABLE,
        source_fingerprint="0" * 64,
        record_count=fingerprint_run_1["record_count"],
        model_name=JOB_MODEL_NAME,
        model_version=JOB_MODEL_VERSION,
        threshold=JOB_THRESHOLD,
        registry_table=JOB_BATCH_REGISTRY_TABLE
    )

except ValueError as error:

    conflict_blocked = True

    print("\nEXPECTED CONFLICT REJECTION")
    print(str(error))

assert conflict_blocked


# --------------------------------------------------
# 7. Validate registry uniqueness
# --------------------------------------------------

registered_rows = (
    spark.table(JOB_BATCH_REGISTRY_TABLE)
    .filter(F.col("batch_id") == JOB_BATCH_ID)
    .count()
)

assert registered_rows == 1

print("\nBATCH IDENTITY VALIDATION")
print("Registered rows:", registered_rows)
print("Identical retry accepted: True")
print("Conflicting fingerprint blocked:", conflict_blocked)
print("Stable batch timestamp: True")

print("\nBATCH IDENTITY AND RETRY PROTECTION: PASS")


# COMMAND ----------


# Step 15.36: Identity-protected end-to-end orchestration

print("=" * 65)
print("IDENTITY-PROTECTED BATCH JOB ORCHESTRATION")
print("=" * 65)


def execute_identity_protected_batch_job(
    source_table,
    preprocessing_pipeline,
    registered_model,
    model_name,
    model_version,
    threshold,
    batch_id,
    destination_table,
    registry_table
):
    """
    Execute a batch inference workflow with:
      - Source fingerprint verification
      - Registered batch identity
      - Stable prediction timestamps
      - Six quality gates
      - Quality-gated Delta publication

    Intended for sequential development job executions.
    """

    # ----------------------------------------------
    # Stage 1: Validate configuration
    # ----------------------------------------------

    if not batch_id or not batch_id.strip():
        raise ValueError("Batch ID cannot be empty.")

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid classification threshold.")

    if not spark.catalog.tableExists(source_table):
        raise ValueError(
            f"Source table does not exist: {source_table}"
        )

    if not destination_table.startswith("workspace.default."):
        raise ValueError("Invalid destination table.")

    print("\nSTAGE 1: LOAD AND IDENTIFY SOURCE")

    source_df = (
        spark.table(source_table)
        .drop("Churn", "label")
    )

    # ----------------------------------------------
    # Stage 2: Calculate source fingerprint
    # ----------------------------------------------

    print("\nSTAGE 2: CALCULATE SOURCE FINGERPRINT")

    fingerprint_results = calculate_source_fingerprint(
        source_df
    )

    source_count = fingerprint_results["record_count"]

    print("Source records:", source_count)
    print(
        "Source fingerprint:",
        fingerprint_results["source_fingerprint"]
    )

    # ----------------------------------------------
    # Stage 3: Register or validate batch identity
    # ----------------------------------------------

    print("\nSTAGE 3: REGISTER BATCH IDENTITY")

    registration_results = register_batch_identity(
        batch_id=batch_id,
        source_table=source_table,
        source_fingerprint=fingerprint_results[
            "source_fingerprint"
        ],
        record_count=source_count,
        model_name=model_name,
        model_version=model_version,
        threshold=threshold,
        registry_table=registry_table
    )

    batch_timestamp = registration_results[
        "batch_timestamp"
    ]

    print(
        "Registration status:",
        registration_results["registration_status"]
    )

    print("Stable batch timestamp:", batch_timestamp)

    # ----------------------------------------------
    # Stage 4: Generate predictions
    # ----------------------------------------------

    print("\nSTAGE 4: GENERATE PREDICTIONS")

    predictions_df = run_batch_inference(
        source_df=source_df,
        preprocessing_pipeline=preprocessing_pipeline,
        registered_model=registered_model,
        threshold=threshold
    )

    # ----------------------------------------------
    # Stage 5: Attach stable metadata
    # ----------------------------------------------

    print("\nSTAGE 5: ATTACH STABLE METADATA")

    enriched_df = add_stable_batch_metadata(
        predictions_df=predictions_df,
        batch_id=batch_id,
        model_name=model_name,
        model_version=model_version,
        threshold=threshold,
        source_table=source_table,
        batch_timestamp=batch_timestamp
    )

    # ----------------------------------------------
    # Stage 6: Execute quality gates
    # ----------------------------------------------

    print("\nSTAGE 6: VALIDATE PREDICTION QUALITY")

    quality_results = validate_batch_predictions(
        predictions_df=enriched_df,
        expected_count=source_count,
        batch_id=batch_id,
        model_name=model_name,
        model_version=model_version,
        threshold=threshold,
        source_table=source_table
    )

    print(
        "Quality gates passed:",
        quality_results["passed_gates"],
        "/",
        quality_results["total_gates"]
    )

    if quality_results["overall_status"] != "PASS":
        raise RuntimeError(
            "Quality validation failed. Publication blocked."
        )

    # ----------------------------------------------
    # Stage 7: Publish approved predictions
    # ----------------------------------------------

    print("\nSTAGE 7: PUBLISH APPROVED PREDICTIONS")

    publication_results = publish_quality_approved_batch(
        predictions_df=enriched_df,
        quality_results=quality_results,
        destination_table=destination_table
    )

    # ----------------------------------------------
    # Stage 8: Return execution summary
    # ----------------------------------------------

    return {
        "batch_id": batch_id,
        "registration_status": registration_results[
            "registration_status"
        ],
        "source_count": source_count,
        "source_fingerprint": fingerprint_results[
            "source_fingerprint"
        ],
        "batch_timestamp": batch_timestamp,
        "model_name": model_name,
        "model_version": model_version,
        "threshold": threshold,
        "quality_status": quality_results[
            "overall_status"
        ],
        "passed_gates": quality_results[
            "passed_gates"
        ],
        "published_count": publication_results[
            "published_count"
        ],
        "destination_table": destination_table,
        "execution_status": "PASS"
    }


print("\nIdentity-protected orchestration function: DEFINED")


# --------------------------------------------------
# Validate required dependencies
# --------------------------------------------------

required_functions = [
    "calculate_source_fingerprint",
    "register_batch_identity",
    "run_batch_inference",
    "add_stable_batch_metadata",
    "validate_batch_predictions",
    "publish_quality_approved_batch",
    "execute_identity_protected_batch_job"
]

missing_functions = [
    name for name in required_functions
    if name not in globals()
    or not callable(globals()[name])
]

assert not missing_functions, (
    f"Missing functions: {missing_functions}"
)

assert spark.catalog.tableExists(JOB_SOURCE_TABLE)

assert spark.catalog.tableExists(
    JOB_BATCH_REGISTRY_TABLE
)

assert 0 <= JOB_THRESHOLD <= 1

print("\nORCHESTRATION READINESS VALIDATION")
print("Required functions:", len(required_functions))
print("Missing functions:", len(missing_functions))
print("Source table:", JOB_SOURCE_TABLE)
print("Registry table:", JOB_BATCH_REGISTRY_TABLE)
print("Destination:", JOB_DEV_OUTPUT_TABLE)

print("\nIDENTITY-PROTECTED ORCHESTRATION DEFINITION: PASS")


# COMMAND ----------


# Step 15.37: Execute identity-protected batch workflow

import uuid
from pyspark.sql import functions as F

print("=" * 65)
print("IDENTITY-PROTECTED BATCH JOB EXECUTION")
print("=" * 65)

# --------------------------------------------------
# 1. Create a NEW logical batch ID
# --------------------------------------------------

# Preserve the same ID when rerunning this cell
# in the current notebook session.

if "JOB_PROTECTED_BATCH_ID" not in globals():
    JOB_PROTECTED_BATCH_ID = str(uuid.uuid4())

print("\nProtected batch ID:", JOB_PROTECTED_BATCH_ID)


# --------------------------------------------------
# 2. Execute the identity-protected workflow
# --------------------------------------------------

protected_job_results = execute_identity_protected_batch_job(
    source_table=JOB_SOURCE_TABLE,
    preprocessing_pipeline=job_preprocessing_pipeline,
    registered_model=job_registered_model,
    model_name=JOB_MODEL_NAME,
    model_version=JOB_MODEL_VERSION,
    threshold=JOB_THRESHOLD,
    batch_id=JOB_PROTECTED_BATCH_ID,
    destination_table=JOB_DEV_OUTPUT_TABLE,
    registry_table=JOB_BATCH_REGISTRY_TABLE
)


# --------------------------------------------------
# 3. Display execution summary
# --------------------------------------------------

print("\nIDENTITY-PROTECTED EXECUTION SUMMARY")

for key, value in protected_job_results.items():
    print(f"{key}: {value}")


# --------------------------------------------------
# 4. Verify the published batch
# --------------------------------------------------

published_batch_df = (
    spark.table(JOB_DEV_OUTPUT_TABLE)
    .filter(
        F.col("batch_id") == JOB_PROTECTED_BATCH_ID
    )
)

publication_audit = published_batch_df.agg(
    F.count("*").alias("record_count"),
    F.countDistinct("customerID").alias("unique_customers"),
    F.countDistinct("prediction_timestamp").alias(
        "distinct_prediction_timestamps"
    ),
    F.countDistinct("model_version").alias(
        "distinct_model_versions"
    ),
    F.countDistinct("classification_threshold").alias(
        "distinct_thresholds"
    ),
    F.sum(
        F.when(
            F.col("customerID").isNull()
            | (F.trim(F.col("customerID")) == ""),
            1
        ).otherwise(0)
    ).alias("invalid_customer_ids")
).first()


# --------------------------------------------------
# 5. Verify registered batch identity
# --------------------------------------------------

registry_record = (
    spark.table(JOB_BATCH_REGISTRY_TABLE)
    .filter(
        F.col("batch_id") == JOB_PROTECTED_BATCH_ID
    )
    .first()
)

assert registry_record is not None

assert (
    registry_record["source_fingerprint"]
    == protected_job_results["source_fingerprint"]
)

assert (
    registry_record["batch_timestamp"]
    == protected_job_results["batch_timestamp"].replace(
        tzinfo=None
    )
)


# --------------------------------------------------
# 6. Enforce final quality checks
# --------------------------------------------------

assert protected_job_results["execution_status"] == "PASS"

assert protected_job_results["quality_status"] == "PASS"

assert protected_job_results["passed_gates"] == 6

assert publication_audit["record_count"] == 1409

assert publication_audit["unique_customers"] == 1409

assert publication_audit["invalid_customer_ids"] == 0

assert (
    publication_audit["distinct_prediction_timestamps"]
    == 1
)

assert publication_audit["distinct_model_versions"] == 1

assert publication_audit["distinct_thresholds"] == 1


# --------------------------------------------------
# 7. Display final verification
# --------------------------------------------------

print("\nPROTECTED BATCH PUBLICATION VERIFICATION")

print(
    "Published records:",
    publication_audit["record_count"]
)

print(
    "Unique customers:",
    publication_audit["unique_customers"]
)

print(
    "Distinct prediction timestamps:",
    publication_audit["distinct_prediction_timestamps"]
)

print(
    "Invalid customer IDs:",
    publication_audit["invalid_customer_ids"]
)

print(
    "Registered fingerprint matches:",
    True
)

print(
    "Stable registered timestamp:",
    True
)

print("\nIDENTITY-PROTECTED BATCH EXECUTION: PASS")


# COMMAND ----------


# Step 15.38: Idempotent retry and conflict rejection

from pyspark.sql import functions as F

print("=" * 65)
print("IDENTITY-PROTECTED RETRY AND CONFLICT TEST")
print("=" * 65)

# --------------------------------------------------
# 1. Capture published state before retry
# --------------------------------------------------

def get_published_batch_state(batch_id):
    df = (
        spark.table(JOB_DEV_OUTPUT_TABLE)
        .filter(F.col("batch_id") == batch_id)
    )

    summary = df.agg(
        F.count("*").alias("record_count"),
        F.countDistinct("customerID").alias(
            "unique_customers"
        ),
        F.countDistinct("prediction_timestamp").alias(
            "prediction_timestamps"
        ),
        F.countDistinct("publication_timestamp").alias(
            "publication_timestamps"
        ),
        F.min("prediction_timestamp").alias(
            "min_prediction_timestamp"
        ),
        F.max("prediction_timestamp").alias(
            "max_prediction_timestamp"
        ),
        F.min("publication_timestamp").alias(
            "min_publication_timestamp"
        ),
        F.max("publication_timestamp").alias(
            "max_publication_timestamp"
        )
    ).first()

    return summary.asDict()


before_retry = get_published_batch_state(
    JOB_PROTECTED_BATCH_ID
)

print("\nSTATE BEFORE RETRY")
print("Records:", before_retry["record_count"])
print("Unique customers:", before_retry["unique_customers"])


# --------------------------------------------------
# 2. Retry the SAME logical batch
# --------------------------------------------------

print("\nTEST A: IDENTICAL BATCH RETRY")

retry_results = execute_identity_protected_batch_job(
    source_table=JOB_SOURCE_TABLE,
    preprocessing_pipeline=job_preprocessing_pipeline,
    registered_model=job_registered_model,
    model_name=JOB_MODEL_NAME,
    model_version=JOB_MODEL_VERSION,
    threshold=JOB_THRESHOLD,
    batch_id=JOB_PROTECTED_BATCH_ID,
    destination_table=JOB_DEV_OUTPUT_TABLE,
    registry_table=JOB_BATCH_REGISTRY_TABLE
)

assert retry_results["registration_status"] == "RETRY"
assert retry_results["quality_status"] == "PASS"
assert retry_results["passed_gates"] == 6

after_retry = get_published_batch_state(
    JOB_PROTECTED_BATCH_ID
)

assert before_retry == after_retry

print("Registration status:", retry_results["registration_status"])
print("Records after retry:", after_retry["record_count"])
print("Published state unchanged:", True)


# --------------------------------------------------
# 3. Simulate a conflicting source fingerprint
# --------------------------------------------------

print("\nTEST B: CONFLICTING BATCH IDENTITY")

conflict_blocked = False

try:
    register_batch_identity(
        batch_id=JOB_PROTECTED_BATCH_ID,
        source_table=JOB_SOURCE_TABLE,
        source_fingerprint="0" * 64,
        record_count=1409,
        model_name=JOB_MODEL_NAME,
        model_version=JOB_MODEL_VERSION,
        threshold=JOB_THRESHOLD,
        registry_table=JOB_BATCH_REGISTRY_TABLE
    )

except ValueError as error:
    if "BATCH IDENTITY CONFLICT" not in str(error):
        raise

    conflict_blocked = True
    print("Expected rejection:", str(error))

assert conflict_blocked


# --------------------------------------------------
# 4. Confirm publication was not modified
# --------------------------------------------------

after_conflict = get_published_batch_state(
    JOB_PROTECTED_BATCH_ID
)

assert after_conflict == after_retry

assert after_conflict["record_count"] == 1409
assert after_conflict["unique_customers"] == 1409
assert after_conflict["prediction_timestamps"] == 1

registry_rows = (
    spark.table(JOB_BATCH_REGISTRY_TABLE)
    .filter(
        F.col("batch_id") == JOB_PROTECTED_BATCH_ID
    )
    .count()
)

assert registry_rows == 1


# --------------------------------------------------
# 5. Final verification
# --------------------------------------------------

print("\nRETRY AND CONFLICT VERIFICATION")
print("Registration status:", retry_results["registration_status"])
print("Published records:", after_conflict["record_count"])
print("Unique customers:", after_conflict["unique_customers"])
print("Distinct prediction timestamps:",
      after_conflict["prediction_timestamps"])
print("Published state unchanged:", before_retry == after_conflict)
print("Conflicting fingerprint blocked:", conflict_blocked)
print("Registry entries:", registry_rows)

print("\nIDENTITY-PROTECTED RETRY AND CONFLICT TEST: PASS")


# COMMAND ----------


# Step 15.39: Customer-level prediction integrity

from pyspark.sql import functions as F

print("=" * 65)
print("CUSTOMER-LEVEL PREDICTION INTEGRITY VALIDATION")
print("=" * 65)

# --------------------------------------------------
# 1. Load published predictions
# --------------------------------------------------

published_df = (
    spark.table(JOB_DEV_OUTPUT_TABLE)
    .filter(F.col("batch_id") == JOB_PROTECTED_BATCH_ID)
    .select(
        "customerID",
        "churn_probability",
        "predicted_class",
        "predicted_churn",
        "prediction_timestamp"
    )
)

# --------------------------------------------------
# 2. Regenerate predictions from registered model
# --------------------------------------------------

source_df = (
    spark.table(JOB_SOURCE_TABLE)
    .drop("Churn", "label")
)

regenerated_df = run_batch_inference(
    source_df=source_df,
    preprocessing_pipeline=job_preprocessing_pipeline,
    registered_model=job_registered_model,
    threshold=JOB_THRESHOLD
).select(
    "customerID",
    "churn_probability",
    "predicted_class",
    "predicted_churn"
)

# --------------------------------------------------
# 3. Check for missing or unexpected customers
# --------------------------------------------------

missing_customers = (
    regenerated_df.select("customerID")
    .join(
        published_df.select("customerID"),
        on="customerID",
        how="left_anti"
    )
    .count()
)

unexpected_customers = (
    published_df.select("customerID")
    .join(
        regenerated_df.select("customerID"),
        on="customerID",
        how="left_anti"
    )
    .count()
)

# --------------------------------------------------
# 4. Compare predictions for matching customers
# --------------------------------------------------

comparison_df = (
    published_df.alias("p")
    .join(
        regenerated_df.alias("r"),
        on="customerID",
        how="inner"
    )
)

comparison_results = comparison_df.agg(
    F.count("*").alias("matched_customers"),

    F.sum(
        F.when(
            F.col("p.churn_probability").isNull()
            | F.col("r.churn_probability").isNull()
            | (
                F.abs(
                    F.col("p.churn_probability")
                    - F.col("r.churn_probability")
                ) > 1e-10
            ),
            1
        ).otherwise(0)
    ).alias("probability_mismatches"),

    F.sum(
        F.when(
            ~F.col("p.predicted_class").eqNullSafe(
                F.col("r.predicted_class")
            ),
            1
        ).otherwise(0)
    ).alias("class_mismatches"),

    F.sum(
        F.when(
            ~F.col("p.predicted_churn").eqNullSafe(
                F.col("r.predicted_churn")
            ),
            1
        ).otherwise(0)
    ).alias("label_mismatches")
).first()

# --------------------------------------------------
# 5. Verify registered timestamp
# --------------------------------------------------

registered_timestamp = (
    spark.table(JOB_BATCH_REGISTRY_TABLE)
    .filter(F.col("batch_id") == JOB_PROTECTED_BATCH_ID)
    .select("batch_timestamp")
    .first()["batch_timestamp"]
)

timestamp_mismatches = published_df.filter(
    ~F.col("prediction_timestamp").eqNullSafe(
        F.lit(registered_timestamp).cast("timestamp")
    )
).count()

# --------------------------------------------------
# 6. Validate all reconciliation checks
# --------------------------------------------------

assert published_df.count() == 1409
assert regenerated_df.count() == 1409

assert missing_customers == 0
assert unexpected_customers == 0

assert comparison_results["matched_customers"] == 1409

assert comparison_results["probability_mismatches"] == 0
assert comparison_results["class_mismatches"] == 0
assert comparison_results["label_mismatches"] == 0

assert timestamp_mismatches == 0

# --------------------------------------------------
# 7. Display reconciliation results
# --------------------------------------------------

print("\nCUSTOMER-LEVEL RECONCILIATION")

print(
    "Matched customers:",
    comparison_results["matched_customers"]
)

print("Missing customers:", missing_customers)
print("Unexpected customers:", unexpected_customers)

print(
    "Probability mismatches:",
    comparison_results["probability_mismatches"]
)

print(
    "Predicted class mismatches:",
    comparison_results["class_mismatches"]
)

print(
    "Predicted churn label mismatches:",
    comparison_results["label_mismatches"]
)

print(
    "Prediction timestamp mismatches:",
    timestamp_mismatches
)

print("\nCUSTOMER-LEVEL PREDICTION INTEGRITY: PASS")


# COMMAND ----------


# Step 15.40: Pin a consistent Delta source snapshot

from pyspark.sql import functions as F
from delta.tables import DeltaTable

print("=" * 65)
print("DELTA SOURCE SNAPSHOT CONSISTENCY")
print("=" * 65)

# --------------------------------------------------
# 1. Inspect Delta table history
# --------------------------------------------------

source_delta_table = DeltaTable.forName(
    spark,
    JOB_SOURCE_TABLE
)

history_df = source_delta_table.history(1)

latest_history = history_df.select(
    "version",
    "timestamp",
    "operation"
).first()

if latest_history is None:
    raise RuntimeError("No Delta history available.")

JOB_SOURCE_VERSION = int(latest_history["version"])

print("\nSOURCE DELTA VERSION")
print("Source table:", JOB_SOURCE_TABLE)
print("Pinned version:", JOB_SOURCE_VERSION)
print("Commit timestamp:", latest_history["timestamp"])
print("Last operation:", latest_history["operation"])


# --------------------------------------------------
# 2. Read the pinned version
# --------------------------------------------------

def load_pinned_source_snapshot(
    source_table,
    source_version
):
    """
    Read a specific Delta table version.

    Repeated reads use the same logical snapshot
    rather than the latest table version.
    """

    return (
        spark.read
        .format("delta")
        .option("versionAsOf", int(source_version))
        .table(source_table)
        .drop("Churn", "label")
    )


pinned_source_df = load_pinned_source_snapshot(
    JOB_SOURCE_TABLE,
    JOB_SOURCE_VERSION
)


# --------------------------------------------------
# 3. Calculate fingerprint twice
# --------------------------------------------------

snapshot_fingerprint_1 = calculate_source_fingerprint(
    pinned_source_df
)

# Independently reload the same version
pinned_source_df_again = load_pinned_source_snapshot(
    JOB_SOURCE_TABLE,
    JOB_SOURCE_VERSION
)

snapshot_fingerprint_2 = calculate_source_fingerprint(
    pinned_source_df_again
)


# --------------------------------------------------
# 4. Validate snapshot identity
# --------------------------------------------------

assert snapshot_fingerprint_1["record_count"] == 1409

assert snapshot_fingerprint_1["unique_customers"] == 1409

assert (
    snapshot_fingerprint_1["source_fingerprint"]
    == snapshot_fingerprint_2["source_fingerprint"]
)

assert (
    snapshot_fingerprint_1["source_fingerprint"]
    == protected_job_results["source_fingerprint"]
), (
    "Pinned source differs from the previously "
    "registered development batch."
)


# --------------------------------------------------
# 5. Display verification
# --------------------------------------------------

print("\nPINNED SOURCE VALIDATION")

print(
    "Source records:",
    snapshot_fingerprint_1["record_count"]
)

print(
    "Unique customers:",
    snapshot_fingerprint_1["unique_customers"]
)

print(
    "Pinned fingerprint:",
    snapshot_fingerprint_1["source_fingerprint"]
)

print(
    "Repeated fingerprint matches:",
    snapshot_fingerprint_1["source_fingerprint"]
    == snapshot_fingerprint_2["source_fingerprint"]
)

print(
    "Registered fingerprint matches:",
    snapshot_fingerprint_1["source_fingerprint"]
    == protected_job_results["source_fingerprint"]
)

print("\nDELTA SOURCE SNAPSHOT CONSISTENCY: PASS")


# COMMAND ----------


# Step 15.41: Snapshot-protected batch orchestration

print("=" * 65)
print("SNAPSHOT-PROTECTED BATCH JOB ORCHESTRATION")
print("=" * 65)


def execute_snapshot_protected_batch_job(
    source_table,
    source_version,
    preprocessing_pipeline,
    registered_model,
    model_name,
    model_version,
    threshold,
    batch_id,
    destination_table,
    registry_table
):
    """
    Execute batch inference against one pinned
    Delta source version.

    The version-qualified source identity is
    registered with the batch fingerprint.
    """

    # ----------------------------------------------
    # Stage 1: Validate configuration
    # ----------------------------------------------

    if not batch_id or not batch_id.strip():
        raise ValueError("Batch ID cannot be empty.")

    if not isinstance(source_version, int) or source_version < 0:
        raise ValueError("Invalid Delta source version.")

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid classification threshold.")

    if not spark.catalog.tableExists(source_table):
        raise ValueError("Source table does not exist.")

    if not spark.catalog.tableExists(registry_table):
        raise ValueError("Batch registry does not exist.")

    if not destination_table.startswith("workspace.default."):
        raise ValueError("Invalid destination table.")

    # ----------------------------------------------
    # Stage 2: Read pinned Delta source snapshot
    # ----------------------------------------------

    print("\nSTAGE 1: LOAD PINNED SOURCE SNAPSHOT")

    source_df = load_pinned_source_snapshot(
        source_table=source_table,
        source_version=source_version
    )

    source_identity = (
        f"{source_table}@v{source_version}"
    )

    print("Source identity:", source_identity)

    # ----------------------------------------------
    # Stage 3: Calculate snapshot fingerprint
    # ----------------------------------------------

    print("\nSTAGE 2: CALCULATE SNAPSHOT FINGERPRINT")

    fingerprint_results = calculate_source_fingerprint(
        source_df
    )

    source_count = fingerprint_results["record_count"]

    print("Source records:", source_count)
    print(
        "Source fingerprint:",
        fingerprint_results["source_fingerprint"]
    )

    # ----------------------------------------------
    # Stage 4: Register or validate batch identity
    # ----------------------------------------------

    print("\nSTAGE 3: REGISTER SNAPSHOT IDENTITY")

    registration_results = register_batch_identity(
        batch_id=batch_id,
        source_table=source_identity,
        source_fingerprint=fingerprint_results[
            "source_fingerprint"
        ],
        record_count=source_count,
        model_name=model_name,
        model_version=model_version,
        threshold=threshold,
        registry_table=registry_table
    )

    batch_timestamp = registration_results[
        "batch_timestamp"
    ]

    print(
        "Registration status:",
        registration_results["registration_status"]
    )

    print("Stable batch timestamp:", batch_timestamp)

    # ----------------------------------------------
    # Stage 5: Generate predictions
    # ----------------------------------------------

    print("\nSTAGE 4: GENERATE PREDICTIONS")

    predictions_df = run_batch_inference(
        source_df=source_df,
        preprocessing_pipeline=preprocessing_pipeline,
        registered_model=registered_model,
        threshold=threshold
    )

    # ----------------------------------------------
    # Stage 6: Attach stable metadata
    # ----------------------------------------------

    print("\nSTAGE 5: ATTACH STABLE METADATA")

    enriched_df = add_stable_batch_metadata(
        predictions_df=predictions_df,
        batch_id=batch_id,
        model_name=model_name,
        model_version=model_version,
        threshold=threshold,
        source_table=source_identity,
        batch_timestamp=batch_timestamp
    )

    # ----------------------------------------------
    # Stage 7: Validate quality gates
    # ----------------------------------------------

    print("\nSTAGE 6: VALIDATE PREDICTION QUALITY")

    quality_results = validate_batch_predictions(
        predictions_df=enriched_df,
        expected_count=source_count,
        batch_id=batch_id,
        model_name=model_name,
        model_version=model_version,
        threshold=threshold,
        source_table=source_identity
    )

    print(
        "Quality gates passed:",
        quality_results["passed_gates"],
        "/",
        quality_results["total_gates"]
    )

    if quality_results["overall_status"] != "PASS":
        raise RuntimeError(
            "Quality validation failed. Publication blocked."
        )

    # ----------------------------------------------
    # Stage 8: Publish approved predictions
    # ----------------------------------------------

    print("\nSTAGE 7: PUBLISH APPROVED PREDICTIONS")

    publication_results = publish_quality_approved_batch(
        predictions_df=enriched_df,
        quality_results=quality_results,
        destination_table=destination_table
    )

    # ----------------------------------------------
    # Stage 9: Return execution summary
    # ----------------------------------------------

    return {
        "batch_id": batch_id,
        "registration_status": registration_results[
            "registration_status"
        ],
        "source_table": source_table,
        "source_version": source_version,
        "source_identity": source_identity,
        "source_count": source_count,
        "source_fingerprint": fingerprint_results[
            "source_fingerprint"
        ],
        "batch_timestamp": batch_timestamp,
        "model_name": model_name,
        "model_version": model_version,
        "threshold": threshold,
        "quality_status": quality_results[
            "overall_status"
        ],
        "passed_gates": quality_results[
            "passed_gates"
        ],
        "published_count": publication_results[
            "published_count"
        ],
        "destination_table": destination_table,
        "execution_status": "PASS"
    }


print("\nSnapshot-protected orchestration function: DEFINED")


# --------------------------------------------------
# Validate required functions
# --------------------------------------------------

required_functions = [
    "load_pinned_source_snapshot",
    "calculate_source_fingerprint",
    "register_batch_identity",
    "run_batch_inference",
    "add_stable_batch_metadata",
    "validate_batch_predictions",
    "publish_quality_approved_batch",
    "execute_snapshot_protected_batch_job"
]

missing_functions = [
    name for name in required_functions
    if name not in globals()
    or not callable(globals()[name])
]

assert not missing_functions, (
    f"Missing functions: {missing_functions}"
)

assert JOB_SOURCE_VERSION == 2

assert spark.catalog.tableExists(JOB_SOURCE_TABLE)

assert spark.catalog.tableExists(
    JOB_BATCH_REGISTRY_TABLE
)

print("\nSNAPSHOT ORCHESTRATION READINESS")

print("Required functions:", len(required_functions))
print("Missing functions:", len(missing_functions))
print("Pinned source version:", JOB_SOURCE_VERSION)
print("Source table:", JOB_SOURCE_TABLE)
print("Registry table:", JOB_BATCH_REGISTRY_TABLE)
print("Development destination:", JOB_DEV_OUTPUT_TABLE)

print("\nSNAPSHOT-PROTECTED ORCHESTRATION DEFINITION: PASS")


# COMMAND ----------


# Step 15.42: Execute snapshot-protected batch job

import uuid
from pyspark.sql import functions as F

print("=" * 65)
print("SNAPSHOT-PROTECTED BATCH JOB EXECUTION")
print("=" * 65)

# --------------------------------------------------
# 1. Create a new logical batch ID
# --------------------------------------------------

# Reuse the ID if this cell is rerun in the same session.

if "JOB_SNAPSHOT_BATCH_ID" not in globals():
    JOB_SNAPSHOT_BATCH_ID = str(uuid.uuid4())

print("\nSnapshot batch ID:", JOB_SNAPSHOT_BATCH_ID)
print("Pinned source version:", JOB_SOURCE_VERSION)


# --------------------------------------------------
# 2. Execute snapshot-protected workflow
# --------------------------------------------------

snapshot_job_results = execute_snapshot_protected_batch_job(
    source_table=JOB_SOURCE_TABLE,
    source_version=JOB_SOURCE_VERSION,
    preprocessing_pipeline=job_preprocessing_pipeline,
    registered_model=job_registered_model,
    model_name=JOB_MODEL_NAME,
    model_version=JOB_MODEL_VERSION,
    threshold=JOB_THRESHOLD,
    batch_id=JOB_SNAPSHOT_BATCH_ID,
    destination_table=JOB_DEV_OUTPUT_TABLE,
    registry_table=JOB_BATCH_REGISTRY_TABLE
)


# --------------------------------------------------
# 3. Display execution summary
# --------------------------------------------------

print("\nSNAPSHOT-PROTECTED EXECUTION SUMMARY")

for key, value in snapshot_job_results.items():
    print(f"{key}: {value}")


# --------------------------------------------------
# 4. Verify published predictions
# --------------------------------------------------

snapshot_published_df = (
    spark.table(JOB_DEV_OUTPUT_TABLE)
    .filter(
        F.col("batch_id") == JOB_SNAPSHOT_BATCH_ID
    )
)

publication_audit = snapshot_published_df.agg(
    F.count("*").alias("record_count"),
    F.countDistinct("customerID").alias(
        "unique_customers"
    ),
    F.countDistinct("prediction_timestamp").alias(
        "prediction_timestamps"
    ),
    F.countDistinct("source_table").alias(
        "source_identities"
    ),
    F.sum(
        F.when(
            F.col("customerID").isNull()
            | (F.trim(F.col("customerID")) == ""),
            1
        ).otherwise(0)
    ).alias("invalid_customer_ids")
).first()


# --------------------------------------------------
# 5. Verify batch registry
# --------------------------------------------------

registry_record = (
    spark.table(JOB_BATCH_REGISTRY_TABLE)
    .filter(
        F.col("batch_id") == JOB_SNAPSHOT_BATCH_ID
    )
    .first()
)

assert registry_record is not None

expected_source_identity = (
    f"{JOB_SOURCE_TABLE}@v{JOB_SOURCE_VERSION}"
)

assert (
    registry_record["source_table"]
    == expected_source_identity
)

assert (
    registry_record["source_fingerprint"]
    == snapshot_job_results["source_fingerprint"]
)

assert (
    registry_record["batch_timestamp"]
    == snapshot_job_results["batch_timestamp"].replace(
        tzinfo=None
    )
)


# --------------------------------------------------
# 6. Validate published source identity
# --------------------------------------------------

source_identity_mismatches = (
    snapshot_published_df
    .filter(
        ~F.col("source_table").eqNullSafe(
            F.lit(expected_source_identity)
        )
    )
    .count()
)


# --------------------------------------------------
# 7. Enforce final assertions
# --------------------------------------------------

assert snapshot_job_results["execution_status"] == "PASS"

assert snapshot_job_results["quality_status"] == "PASS"

assert snapshot_job_results["passed_gates"] == 6

assert snapshot_job_results["source_version"] == 2

assert publication_audit["record_count"] == 1409

assert publication_audit["unique_customers"] == 1409

assert publication_audit["prediction_timestamps"] == 1

assert publication_audit["source_identities"] == 1

assert publication_audit["invalid_customer_ids"] == 0

assert source_identity_mismatches == 0


# --------------------------------------------------
# 8. Display final verification
# --------------------------------------------------

print("\nSNAPSHOT PUBLICATION VERIFICATION")

print(
    "Published records:",
    publication_audit["record_count"]
)

print(
    "Unique customers:",
    publication_audit["unique_customers"]
)

print(
    "Distinct prediction timestamps:",
    publication_audit["prediction_timestamps"]
)

print(
    "Source identity:",
    expected_source_identity
)

print(
    "Source identity mismatches:",
    source_identity_mismatches
)

print(
    "Registered fingerprint matches:",
    True
)

print(
    "Registered timestamp matches:",
    True
)

print("\nSNAPSHOT-PROTECTED BATCH EXECUTION: PASS")


# COMMAND ----------


# Step 15.43: Snapshot retry and version conflict validation

from pyspark.sql import functions as F

print("=" * 65)
print("SNAPSHOT RETRY AND VERSION CONFLICT VALIDATION")
print("=" * 65)

# --------------------------------------------------
# 1. Capture published state before retry
# --------------------------------------------------

def get_snapshot_batch_state(batch_id):
    df = (
        spark.table(JOB_DEV_OUTPUT_TABLE)
        .filter(F.col("batch_id") == batch_id)
    )

    summary = df.agg(
        F.count("*").alias("record_count"),
        F.countDistinct("customerID").alias("unique_customers"),
        F.countDistinct("prediction_timestamp").alias(
            "prediction_timestamps"
        ),
        F.min("prediction_timestamp").alias(
            "min_prediction_timestamp"
        ),
        F.max("prediction_timestamp").alias(
            "max_prediction_timestamp"
        ),
        F.min("publication_timestamp").alias(
            "min_publication_timestamp"
        ),
        F.max("publication_timestamp").alias(
            "max_publication_timestamp"
        )
    ).first()

    return summary.asDict()


before_retry = get_snapshot_batch_state(
    JOB_SNAPSHOT_BATCH_ID
)

assert before_retry["record_count"] == 1409

print("\nSTATE BEFORE RETRY")
print("Published records:", before_retry["record_count"])
print("Unique customers:", before_retry["unique_customers"])


# --------------------------------------------------
# 2. Retry the same pinned Delta snapshot
# --------------------------------------------------

print("\nTEST A: IDENTICAL SNAPSHOT RETRY")

snapshot_retry_results = execute_snapshot_protected_batch_job(
    source_table=JOB_SOURCE_TABLE,
    source_version=JOB_SOURCE_VERSION,
    preprocessing_pipeline=job_preprocessing_pipeline,
    registered_model=job_registered_model,
    model_name=JOB_MODEL_NAME,
    model_version=JOB_MODEL_VERSION,
    threshold=JOB_THRESHOLD,
    batch_id=JOB_SNAPSHOT_BATCH_ID,
    destination_table=JOB_DEV_OUTPUT_TABLE,
    registry_table=JOB_BATCH_REGISTRY_TABLE
)

assert snapshot_retry_results["registration_status"] == "RETRY"
assert snapshot_retry_results["quality_status"] == "PASS"
assert snapshot_retry_results["passed_gates"] == 6

after_retry = get_snapshot_batch_state(
    JOB_SNAPSHOT_BATCH_ID
)

assert before_retry == after_retry

print("Registration status:",
      snapshot_retry_results["registration_status"])
print("Published state unchanged:", before_retry == after_retry)


# --------------------------------------------------
# 3. Simulate reuse with another Delta version
# --------------------------------------------------

print("\nTEST B: CONFLICTING SOURCE VERSION")

conflicting_version = JOB_SOURCE_VERSION + 1

conflicting_source_identity = (
    f"{JOB_SOURCE_TABLE}@v{conflicting_version}"
)

version_conflict_blocked = False

try:
    register_batch_identity(
        batch_id=JOB_SNAPSHOT_BATCH_ID,
        source_table=conflicting_source_identity,
        source_fingerprint=snapshot_job_results[
            "source_fingerprint"
        ],
        record_count=1409,
        model_name=JOB_MODEL_NAME,
        model_version=JOB_MODEL_VERSION,
        threshold=JOB_THRESHOLD,
        registry_table=JOB_BATCH_REGISTRY_TABLE
    )

except ValueError as error:
    if "BATCH IDENTITY CONFLICT" not in str(error):
        raise

    version_conflict_blocked = True
    print("Expected rejection:", str(error))

assert version_conflict_blocked


# --------------------------------------------------
# 4. Confirm published data remains unchanged
# --------------------------------------------------

after_conflict = get_snapshot_batch_state(
    JOB_SNAPSHOT_BATCH_ID
)

assert after_conflict == before_retry

registry_rows = (
    spark.table(JOB_BATCH_REGISTRY_TABLE)
    .filter(F.col("batch_id") == JOB_SNAPSHOT_BATCH_ID)
    .count()
)

assert registry_rows == 1


# --------------------------------------------------
# 5. Final verification
# --------------------------------------------------

print("\nSNAPSHOT RETRY VERIFICATION")

print(
    "Registration status:",
    snapshot_retry_results["registration_status"]
)

print("Published records:", after_conflict["record_count"])
print("Unique customers:", after_conflict["unique_customers"])
print(
    "Distinct prediction timestamps:",
    after_conflict["prediction_timestamps"]
)

print("Published state unchanged:", before_retry == after_conflict)
print("Version conflict blocked:", version_conflict_blocked)
print("Registry entries:", registry_rows)

print("\nSNAPSHOT RETRY AND VERSION CONFLICT: PASS")


# COMMAND ----------

# DBTITLE 1,Step 15.44: Define strict publication integrity checks

# Step 15.44: Define strict publication integrity checks

from pyspark.sql import functions as F

print("=" * 65)
print("STRICT DELTA PUBLICATION INTEGRITY")
print("=" * 65)


def validate_strict_publication_integrity(
    predictions_df,
    destination_table
):
    """
    Verify that existing published predictions
    do not conflict with incoming predictions.

    This function is read-only.
    """

    required_columns = [
        "batch_id",
        "customerID",
        "churn_probability",
        "predicted_class",
        "predicted_churn",
        "prediction_timestamp",
        "model_name",
        "model_version",
        "classification_threshold",
        "source_table"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in predictions_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing prediction columns: {missing_columns}"
        )

    # ----------------------------------------------
    # 1. Validate incoming batch identity
    # ----------------------------------------------

    batch_ids = (
        predictions_df
        .select("batch_id")
        .distinct()
        .collect()
    )

    if len(batch_ids) != 1 or batch_ids[0]["batch_id"] is None:
        raise ValueError(
            "Incoming predictions must contain exactly one batch ID."
        )

    batch_id = batch_ids[0]["batch_id"]

    incoming_count = predictions_df.count()

    if incoming_count == 0:
        raise ValueError("Incoming prediction batch is empty.")

    incoming_unique = (
        predictions_df
        .select("customerID")
        .distinct()
        .count()
    )

    invalid_ids = predictions_df.filter(
        F.col("customerID").isNull()
        | (F.trim(F.col("customerID")) == "")
    ).count()

    if incoming_count != incoming_unique or invalid_ids > 0:
        raise ValueError(
            "Incoming predictions contain duplicate or invalid customer IDs."
        )

    # ----------------------------------------------
    # 2. Check whether destination exists
    # ----------------------------------------------

    if not spark.catalog.tableExists(destination_table):
        return {
            "status": "PASS",
            "batch_id": batch_id,
            "incoming_count": incoming_count,
            "existing_count": 0,
            "conflicting_records": 0,
            "publication_mode": "NEW_BATCH"
        }

    existing_df = (
        spark.table(destination_table)
        .filter(F.col("batch_id") == batch_id)
    )

    existing_count = existing_df.count()

    if existing_count == 0:
        return {
            "status": "PASS",
            "batch_id": batch_id,
            "incoming_count": incoming_count,
            "existing_count": 0,
            "conflicting_records": 0,
            "publication_mode": "NEW_BATCH"
        }

    # ----------------------------------------------
    # 3. Check existing customer uniqueness
    # ----------------------------------------------

    existing_unique = (
        existing_df
        .select("customerID")
        .distinct()
        .count()
    )

    if existing_count != existing_unique:
        raise ValueError(
            "PUBLICATION INTEGRITY CONFLICT: "
            "Existing batch contains duplicate customers."
        )

    # ----------------------------------------------
    # 4. Compare customer populations
    # ----------------------------------------------

    missing_existing = (
        predictions_df.select("customerID")
        .join(
            existing_df.select("customerID"),
            "customerID",
            "left_anti"
        )
        .count()
    )

    unexpected_existing = (
        existing_df.select("customerID")
        .join(
            predictions_df.select("customerID"),
            "customerID",
            "left_anti"
        )
        .count()
    )

    if missing_existing > 0 or unexpected_existing > 0:
        raise ValueError(
            "PUBLICATION INTEGRITY CONFLICT: "
            "Customer populations differ."
        )

    # ----------------------------------------------
    # 5. Compare published and incoming values
    # ----------------------------------------------

    p = existing_df.alias("p")
    n = predictions_df.alias("n")

    comparison_df = p.join(
        n,
        on="customerID",
        how="inner"
    )

    exact_columns = [
        "predicted_class",
        "predicted_churn",
        "prediction_timestamp",
        "model_name",
        "model_version",
        "classification_threshold",
        "source_table"
    ]

    conflict_condition = (
        F.col("p.churn_probability").isNull()
        | F.col("n.churn_probability").isNull()
        | (
            F.abs(
                F.col("p.churn_probability")
                - F.col("n.churn_probability")
            ) > 1e-10
        )
    )

    for column_name in exact_columns:
        conflict_condition = (
            conflict_condition
            | ~F.col(f"p.{column_name}").eqNullSafe(
                F.col(f"n.{column_name}")
            )
        )

    conflicting_records = (
        comparison_df
        .filter(conflict_condition)
        .count()
    )

    if conflicting_records > 0:
        raise ValueError(
            "PUBLICATION INTEGRITY CONFLICT: "
            f"{conflicting_records} customer predictions differ."
        )

    # ----------------------------------------------
    # 6. Return validation result
    # ----------------------------------------------

    return {
        "status": "PASS",
        "batch_id": batch_id,
        "incoming_count": incoming_count,
        "existing_count": existing_count,
        "conflicting_records": 0,
        "publication_mode": "IDENTICAL_RETRY"
    }


print("\nStrict publication integrity function: DEFINED")

assert callable(validate_strict_publication_integrity)

print("\nSTRICT PUBLICATION READINESS")
print("Validation function available:", True)
print("Development destination:", JOB_DEV_OUTPUT_TABLE)
print("Existing snapshot batch:", JOB_SNAPSHOT_BATCH_ID)

print("\nSTRICT PUBLICATION INTEGRITY DEFINITION: PASS")


# COMMAND ----------


# Step 15.45: Test strict publication integrity

from pyspark.sql import functions as F

print("=" * 65)
print("STRICT PUBLICATION INTEGRITY TESTS")
print("=" * 65)

# --------------------------------------------------
# 1. Load existing published batch
# --------------------------------------------------

original_df = (
    spark.table(JOB_DEV_OUTPUT_TABLE)
    .filter(F.col("batch_id") == JOB_SNAPSHOT_BATCH_ID)
)

assert original_df.count() == 1409

# Choose one customer to simulate a conflicting record.
test_customer_id = (
    original_df
    .select("customerID")
    .orderBy("customerID")
    .first()["customerID"]
)

print("\nTest customer:", test_customer_id)


# --------------------------------------------------
# TEST A: Identical predictions
# --------------------------------------------------

print("\nTEST A: IDENTICAL RETRY")

identical_result = validate_strict_publication_integrity(
    predictions_df=original_df,
    destination_table=JOB_DEV_OUTPUT_TABLE
)

assert identical_result["status"] == "PASS"

assert (
    identical_result["publication_mode"]
    == "IDENTICAL_RETRY"
)

assert identical_result["conflicting_records"] == 0

print("Validation status:", identical_result["status"])
print("Publication mode:", identical_result["publication_mode"])
print("Conflicting records:", identical_result["conflicting_records"])


# --------------------------------------------------
# TEST B: Modify one customer's probability
# --------------------------------------------------

print("\nTEST B: CONFLICTING CHURN PROBABILITY")

probability_conflict_df = original_df.withColumn(
    "churn_probability",
    F.when(
        F.col("customerID") == test_customer_id,
        F.when(
            F.col("churn_probability") >= 0.5,
            F.lit(0.01)
        ).otherwise(F.lit(0.99))
    ).otherwise(F.col("churn_probability"))
)

probability_conflict_blocked = False

try:
    validate_strict_publication_integrity(
        predictions_df=probability_conflict_df,
        destination_table=JOB_DEV_OUTPUT_TABLE
    )

except ValueError as error:
    if "PUBLICATION INTEGRITY CONFLICT" not in str(error):
        raise

    probability_conflict_blocked = True
    print("Expected rejection:", str(error))

assert probability_conflict_blocked


# --------------------------------------------------
# TEST C: Modify one customer's churn label
# --------------------------------------------------

print("\nTEST C: CONFLICTING CHURN LABEL")

label_conflict_df = original_df.withColumn(
    "predicted_churn",
    F.when(
        F.col("customerID") == test_customer_id,
        F.when(
            F.col("predicted_churn") == "Yes",
            F.lit("No")
        ).otherwise(F.lit("Yes"))
    ).otherwise(F.col("predicted_churn"))
)

label_conflict_blocked = False

try:
    validate_strict_publication_integrity(
        predictions_df=label_conflict_df,
        destination_table=JOB_DEV_OUTPUT_TABLE
    )

except ValueError as error:
    if "PUBLICATION INTEGRITY CONFLICT" not in str(error):
        raise

    label_conflict_blocked = True
    print("Expected rejection:", str(error))

assert label_conflict_blocked


# --------------------------------------------------
# 4. Verify published table remains unchanged
# --------------------------------------------------

after_tests_df = (
    spark.table(JOB_DEV_OUTPUT_TABLE)
    .filter(F.col("batch_id") == JOB_SNAPSHOT_BATCH_ID)
)

assert after_tests_df.count() == 1409

assert (
    after_tests_df.select("customerID").distinct().count()
    == 1409
)

# --------------------------------------------------
# 5. Display results
# --------------------------------------------------

print("\nSTRICT PUBLICATION TEST SUMMARY")

print(
    "Identical retry accepted:",
    identical_result["status"] == "PASS"
)

print(
    "Probability conflict blocked:",
    probability_conflict_blocked
)

print(
    "Churn label conflict blocked:",
    label_conflict_blocked
)

print("Published records:", after_tests_df.count())

print("\nSTRICT PUBLICATION INTEGRITY TESTS: PASS")


# COMMAND ----------


# Step 15.46: Integrate strict publication validation

print("=" * 65)
print("STRICT PUBLICATION WORKFLOW INTEGRATION")
print("=" * 65)

# --------------------------------------------------
# 1. Preserve the original Delta publisher
# --------------------------------------------------

# Keep the original publisher available for the wrapper.
# This guard prevents accidentally wrapping the wrapper
# if the notebook cell is executed more than once.

if "_original_publish_quality_approved_batch" not in globals():
    _original_publish_quality_approved_batch = (
        publish_quality_approved_batch
    )

assert callable(_original_publish_quality_approved_batch)


# --------------------------------------------------
# 2. Define the strict publication wrapper
# --------------------------------------------------

def publish_quality_approved_batch_strict(
    predictions_df,
    quality_results,
    destination_table
):
    """
    Validate prediction integrity before publishing
    a quality-approved batch to Delta.

    Conflicting existing predictions are rejected.
    """

    # Require quality approval before publication.
    if quality_results["overall_status"] != "PASS":
        raise ValueError(
            "PUBLICATION BLOCKED: Quality gates did not pass."
        )

    print("\nSTRICT PUBLICATION VALIDATION")

    integrity_results = validate_strict_publication_integrity(
        predictions_df=predictions_df,
        destination_table=destination_table
    )

    if integrity_results["status"] != "PASS":
        raise RuntimeError(
            "PUBLICATION BLOCKED: Integrity validation failed."
        )

    print(
        "Integrity status:",
        integrity_results["status"]
    )

    print(
        "Publication mode:",
        integrity_results["publication_mode"]
    )

    print(
        "Conflicting records:",
        integrity_results["conflicting_records"]
    )

    # Invoke the preserved original Delta publisher.
    publication_results = (
        _original_publish_quality_approved_batch(
            predictions_df=predictions_df,
            quality_results=quality_results,
            destination_table=destination_table
        )
    )

    return publication_results


# --------------------------------------------------
# 3. Install the strict publisher
# --------------------------------------------------

publish_quality_approved_batch = (
    publish_quality_approved_batch_strict
)


# --------------------------------------------------
# 4. Verify integration
# --------------------------------------------------

assert callable(publish_quality_approved_batch)

assert callable(
    _original_publish_quality_approved_batch
)

assert (
    publish_quality_approved_batch
    is publish_quality_approved_batch_strict
)

assert (
    publish_quality_approved_batch
    is not _original_publish_quality_approved_batch
)


print("\nSTRICT PUBLICATION INTEGRATION READINESS")

print(
    "Original publisher preserved:",
    callable(_original_publish_quality_approved_batch)
)

print(
    "Strict validator available:",
    callable(validate_strict_publication_integrity)
)

print(
    "Strict publisher installed:",
    publish_quality_approved_batch
    is publish_quality_approved_batch_strict
)

print(
    "Snapshot orchestrator available:",
    callable(execute_snapshot_protected_batch_job)
)

print("\nSTRICT PUBLICATION INTEGRATION: PASS")


# COMMAND ----------

# DBTITLE 1,Step 15.47: Execute integrated strict publication retry

# Step 15.47: Execute integrated strict publication retry

from pyspark.sql import functions as F

print("=" * 65)
print("INTEGRATED STRICT PUBLICATION RETRY TEST")
print("=" * 65)

# --------------------------------------------------
# 1. Verify strict publisher is installed
# --------------------------------------------------

assert (
    publish_quality_approved_batch
    is publish_quality_approved_batch_strict
)

assert callable(execute_snapshot_protected_batch_job)

print("\nStrict publisher installed: True")


# --------------------------------------------------
# 2. Capture existing published predictions
# --------------------------------------------------

audit_columns = [
    "customerID",
    "churn_probability",
    "predicted_class",
    "predicted_churn",
    "prediction_timestamp",
    "publication_timestamp",
    "model_name",
    "model_version",
    "classification_threshold",
    "source_table"
]

before_df = (
    spark.table(JOB_DEV_OUTPUT_TABLE)
    .filter(F.col("batch_id") == JOB_SNAPSHOT_BATCH_ID)
    .select(*audit_columns)
)

before_count = before_df.count()

assert before_count == 1409

# Create a deterministic row-level fingerprint.
# This is a comparison checksum for this test.

def calculate_published_state_hash(df):
    import hashlib

    digest = hashlib.sha256()

    for row in (
        df.orderBy("customerID")
        .toLocalIterator()
    ):
        digest.update(
            repr(tuple(row)).encode("utf-8")
        )
        digest.update(b"\n")

    return digest.hexdigest()


before_hash = calculate_published_state_hash(before_df)

print("\nSTATE BEFORE RETRY")
print("Published records:", before_count)
print("Published state hash:", before_hash)


# --------------------------------------------------
# 3. Execute snapshot-protected retry
# --------------------------------------------------

print("\nEXECUTING INTEGRATED WORKFLOW")

integrated_retry_results = execute_snapshot_protected_batch_job(
    source_table=JOB_SOURCE_TABLE,
    source_version=JOB_SOURCE_VERSION,
    preprocessing_pipeline=job_preprocessing_pipeline,
    registered_model=job_registered_model,
    model_name=JOB_MODEL_NAME,
    model_version=JOB_MODEL_VERSION,
    threshold=JOB_THRESHOLD,
    batch_id=JOB_SNAPSHOT_BATCH_ID,
    destination_table=JOB_DEV_OUTPUT_TABLE,
    registry_table=JOB_BATCH_REGISTRY_TABLE
)


# --------------------------------------------------
# 4. Verify execution status
# --------------------------------------------------

assert (
    integrated_retry_results["registration_status"]
    == "RETRY"
)

assert (
    integrated_retry_results["quality_status"]
    == "PASS"
)

assert integrated_retry_results["passed_gates"] == 6

assert (
    integrated_retry_results["execution_status"]
    == "PASS"
)


# --------------------------------------------------
# 5. Capture published state after retry
# --------------------------------------------------

after_df = (
    spark.table(JOB_DEV_OUTPUT_TABLE)
    .filter(F.col("batch_id") == JOB_SNAPSHOT_BATCH_ID)
    .select(*audit_columns)
)

after_count = after_df.count()

after_hash = calculate_published_state_hash(after_df)

assert after_count == before_count

assert after_hash == before_hash


# --------------------------------------------------
# 6. Verify registry identity
# --------------------------------------------------

registry_rows = (
    spark.table(JOB_BATCH_REGISTRY_TABLE)
    .filter(F.col("batch_id") == JOB_SNAPSHOT_BATCH_ID)
    .count()
)

assert registry_rows == 1


# --------------------------------------------------
# 7. Display final verification
# --------------------------------------------------

print("\nINTEGRATED STRICT RETRY VERIFICATION")

print(
    "Registration status:",
    integrated_retry_results["registration_status"]
)

print(
    "Quality gates passed:",
    integrated_retry_results["passed_gates"]
)

print("Published records before:", before_count)
print("Published records after:", after_count)

print("Published state unchanged:", before_hash == after_hash)

print("Registry entries:", registry_rows)

print(
    "Strict publisher active:",
    publish_quality_approved_batch
    is publish_quality_approved_batch_strict
)

print("\nINTEGRATED STRICT PUBLICATION RETRY: PASS")


# COMMAND ----------

# DBTITLE 1,Step 15.48: Integrated publication conflict blocking

# Step 15.48: Integrated publication conflict blocking

from pyspark.sql import functions as F

print("=" * 65)
print("INTEGRATED PUBLICATION CONFLICT TESTS")
print("=" * 65)

assert (
    publish_quality_approved_batch
    is publish_quality_approved_batch_strict
)

# --------------------------------------------------
# 1. Load existing development batch
# --------------------------------------------------

baseline_df = (
    spark.table(JOB_DEV_OUTPUT_TABLE)
    .filter(F.col("batch_id") == JOB_SNAPSHOT_BATCH_ID)
)

assert baseline_df.count() == 1409

test_customer = (
    baseline_df
    .select("customerID")
    .orderBy("customerID")
    .first()["customerID"]
)

print("\nTest customer:", test_customer)

# Reuse the successful quality result structure.
# The publisher checks overall_status before writing.

approved_quality = {
    "overall_status": "PASS",
    "passed_gates": 6,
    "total_gates": 6,
    "failed_gates": 0,
    "batch_id": JOB_SNAPSHOT_BATCH_ID,
    "record_count": 1409
}

# Preserve a checksum of the published records.
baseline_hash = calculate_published_state_hash(
    baseline_df.select(*audit_columns)
)

# --------------------------------------------------
# TEST A: Identical retry through strict publisher
# --------------------------------------------------

print("\nTEST A: IDENTICAL RETRY")

identical_publication = publish_quality_approved_batch_strict(
    predictions_df=baseline_df,
    quality_results=approved_quality,
    destination_table=JOB_DEV_OUTPUT_TABLE
)

print("Identical retry: ACCEPTED")

# --------------------------------------------------
# TEST B: Conflicting probability
# --------------------------------------------------

print("\nTEST B: CONFLICTING PROBABILITY")

probability_conflict_df = baseline_df.withColumn(
    "churn_probability",
    F.when(
        F.col("customerID") == test_customer,
        F.when(
            F.col("churn_probability") >= 0.5,
            F.lit(0.01)
        ).otherwise(F.lit(0.99))
    ).otherwise(F.col("churn_probability"))
)

probability_blocked = False

try:
    publish_quality_approved_batch_strict(
        predictions_df=probability_conflict_df,
        quality_results=approved_quality,
        destination_table=JOB_DEV_OUTPUT_TABLE
    )

except ValueError as error:
    if "PUBLICATION INTEGRITY CONFLICT" not in str(error):
        raise

    probability_blocked = True
    print("Expected rejection:", str(error))

assert probability_blocked

# --------------------------------------------------
# TEST C: Conflicting predicted class
# --------------------------------------------------

print("\nTEST C: CONFLICTING PREDICTED CLASS")

class_conflict_df = baseline_df.withColumn(
    "predicted_class",
    F.when(
        F.col("customerID") == test_customer,
        F.when(
            F.col("predicted_class") == 1,
            F.lit(0)
        ).otherwise(F.lit(1))
    ).otherwise(F.col("predicted_class"))
)

class_blocked = False

try:
    publish_quality_approved_batch_strict(
        predictions_df=class_conflict_df,
        quality_results=approved_quality,
        destination_table=JOB_DEV_OUTPUT_TABLE
    )

except ValueError as error:
    if "PUBLICATION INTEGRITY CONFLICT" not in str(error):
        raise

    class_blocked = True
    print("Expected rejection:", str(error))

assert class_blocked

# --------------------------------------------------
# 4. Verify development table remains unchanged
# --------------------------------------------------

final_df = (
    spark.table(JOB_DEV_OUTPUT_TABLE)
    .filter(F.col("batch_id") == JOB_SNAPSHOT_BATCH_ID)
)

final_count = final_df.count()

final_hash = calculate_published_state_hash(
    final_df.select(*audit_columns)
)

assert final_count == 1409
assert final_hash == baseline_hash

# --------------------------------------------------
# 5. Final verification
# --------------------------------------------------

print("\nINTEGRATED PUBLICATION TEST SUMMARY")

print("Identical retry accepted:", True)
print("Probability conflict blocked:", probability_blocked)
print("Predicted class conflict blocked:", class_blocked)
print("Published records:", final_count)
print("Published state unchanged:", final_hash == baseline_hash)

print("\nINTEGRATED PUBLICATION CONFLICT TESTS: PASS")


# COMMAND ----------


# Step 15.49: Define Databricks job concurrency configuration

print("=" * 65)
print("DATABRICKS JOB CONCURRENCY CONFIGURATION")
print("=" * 65)

# --------------------------------------------------
# 1. Define job concurrency settings
# --------------------------------------------------

JOB_CONCURRENCY_CONFIG = {
    "max_concurrent_runs": 1,
    "queue_enabled": True,
    "timeout_seconds": 3600,
    "max_retries": 2,
    "min_retry_interval_millis": 60000,
    "retry_on_timeout": False
}

# --------------------------------------------------
# 2. Validate configuration
# --------------------------------------------------

assert JOB_CONCURRENCY_CONFIG["max_concurrent_runs"] == 1

assert JOB_CONCURRENCY_CONFIG["queue_enabled"] is True

assert JOB_CONCURRENCY_CONFIG["timeout_seconds"] > 0

assert JOB_CONCURRENCY_CONFIG["max_retries"] >= 0

assert (
    JOB_CONCURRENCY_CONFIG["min_retry_interval_millis"]
    >= 60000
)

# --------------------------------------------------
# 3. Define intended job behavior
# --------------------------------------------------

JOB_EXECUTION_POLICY = {
    "job_name": "Telco_Churn_Snapshot_Protected_Inference",
    "source_table": JOB_SOURCE_TABLE,
    "source_version": JOB_SOURCE_VERSION,
    "model_name": JOB_MODEL_NAME,
    "model_version": JOB_MODEL_VERSION,
    "destination_table": JOB_DEV_OUTPUT_TABLE,
    "registry_table": JOB_BATCH_REGISTRY_TABLE,
    "concurrency": JOB_CONCURRENCY_CONFIG
}

# --------------------------------------------------
# 4. Display configuration
# --------------------------------------------------

print("\nJOB CONCURRENCY SETTINGS")

for key, value in JOB_CONCURRENCY_CONFIG.items():
    print(f"{key}: {value}")

print("\nJOB EXECUTION POLICY")

print("Job name:", JOB_EXECUTION_POLICY["job_name"])
print("Source table:", JOB_EXECUTION_POLICY["source_table"])
print("Source version:", JOB_EXECUTION_POLICY["source_version"])
print("Model version:", JOB_EXECUTION_POLICY["model_version"])
print("Destination:", JOB_EXECUTION_POLICY["destination_table"])
print("Registry:", JOB_EXECUTION_POLICY["registry_table"])

print("\nJOB CONCURRENCY CONFIGURATION: PASS")


# COMMAND ----------


# Step 15.50: Prepare Databricks Jobs API definition

import json

print("=" * 65)
print("DATABRICKS JOB DEFINITION PREPARATION")
print("=" * 65)

# --------------------------------------------------
# 1. Define the notebook path
# --------------------------------------------------

# IMPORTANT:
# Replace this placeholder with the actual workspace
# path of Notebook 15 before deploying the job.

JOB_NOTEBOOK_PATH = (
    "/Workspace/Shared/Telco_Customer_Churn_ML/"
    "15_Telco_Production_Batch_Inference"
)

# --------------------------------------------------
# 2. Build job-level configuration
# --------------------------------------------------

JOB_DEFINITION = {
    "name": JOB_EXECUTION_POLICY["job_name"],

    "max_concurrent_runs": (
        JOB_CONCURRENCY_CONFIG["max_concurrent_runs"]
    ),

    "queue": {
        "enabled": JOB_CONCURRENCY_CONFIG["queue_enabled"]
    },

    "timeout_seconds": (
        JOB_CONCURRENCY_CONFIG["timeout_seconds"]
    ),

    "tasks": [
        {
            "task_key": "telco_snapshot_batch_inference",

            "notebook_task": {
                "notebook_path": JOB_NOTEBOOK_PATH,
                "source": "WORKSPACE"
            },

            "timeout_seconds": (
                JOB_CONCURRENCY_CONFIG["timeout_seconds"]
            ),

            "max_retries": (
                JOB_CONCURRENCY_CONFIG["max_retries"]
            ),

            "min_retry_interval_millis": (
                JOB_CONCURRENCY_CONFIG[
                    "min_retry_interval_millis"
                ]
            ),

            "retry_on_timeout": (
                JOB_CONCURRENCY_CONFIG["retry_on_timeout"]
            )
        }
    ]
}

# --------------------------------------------------
# 3. Validate job structure
# --------------------------------------------------

assert JOB_DEFINITION["max_concurrent_runs"] == 1

assert JOB_DEFINITION["queue"]["enabled"] is True

assert len(JOB_DEFINITION["tasks"]) == 1

task = JOB_DEFINITION["tasks"][0]

assert task["task_key"] == "telco_snapshot_batch_inference"

assert task["max_retries"] == 2

assert task["min_retry_interval_millis"] == 60000

assert task["retry_on_timeout"] is False

assert task["notebook_task"]["source"] == "WORKSPACE"

# --------------------------------------------------
# 4. Display job definition
# --------------------------------------------------

print("\nJOB DEFINITION")

print(json.dumps(JOB_DEFINITION, indent=2))

print("\nJOB DEFINITION VALIDATION")

print("Job name:", JOB_DEFINITION["name"])
print("Concurrent runs:", JOB_DEFINITION["max_concurrent_runs"])
print("Queue enabled:", JOB_DEFINITION["queue"]["enabled"])
print("Task count:", len(JOB_DEFINITION["tasks"]))
print("Task retries:", task["max_retries"])
print("Task timeout:", task["timeout_seconds"])
print("Notebook path:", JOB_NOTEBOOK_PATH)

print("\nDATABRICKS JOB DEFINITION: PASS")


# COMMAND ----------


# Step 15.51: Verify notebook path and compute availability

print("=" * 65)
print("DATABRICKS NOTEBOOK AND COMPUTE DISCOVERY")
print("=" * 65)

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# --------------------------------------------------
# 1. Verify configured notebook path
# --------------------------------------------------

print("\nNOTEBOOK PATH VERIFICATION")

configured_path = JOB_DEFINITION[
    "tasks"
][0]["notebook_task"]["notebook_path"]

print("Configured path:", configured_path)

notebook_verified = False

try:
    notebook_info = w.workspace.get_status(
        path=configured_path
    )

    print("Workspace object type:", notebook_info.object_type)
    print("Workspace object ID:", notebook_info.object_id)

    notebook_verified = (
        str(notebook_info.object_type).upper().endswith("NOTEBOOK")
    )

except Exception as error:
    print("Configured notebook path could not be verified.")
    print("Details:", str(error)[:500])

print("Notebook path verified:", notebook_verified)


# --------------------------------------------------
# 2. Identify current notebook, if available
# --------------------------------------------------

print("\nCURRENT NOTEBOOK CONTEXT")

try:
    current_context = (
        dbutils.notebook.entry_point
        .getDbutils()
        .notebook()
        .getContext()
    )

    current_path = current_context.notebookPath().get()

    print("Current notebook path:", current_path)

except Exception as error:
    current_path = None
    print("Current notebook path unavailable.")
    print("Details:", str(error)[:300])


# --------------------------------------------------
# 3. Inspect available clusters
# --------------------------------------------------

print("\nAVAILABLE CLUSTERS")

try:
    available_clusters = list(w.clusters.list())

    print("Clusters found:", len(available_clusters))

    for cluster in available_clusters[:10]:
        print(
            "Cluster:",
            cluster.cluster_name,
            "| ID:",
            cluster.cluster_id,
            "| State:",
            cluster.state
        )

except Exception as error:
    print("Cluster discovery unavailable.")
    print("Details:", str(error)[:500])


# --------------------------------------------------
# 4. Inspect serverless job capability
# --------------------------------------------------

print("\nJOB COMPUTE DISCOVERY")

print(
    "Serverless Jobs eligibility must be confirmed "
    "in the workspace configuration."
)

print(
    "No compute resources were created or modified."
)

print("\nDATABRICKS DISCOVERY CHECK: COMPLETE")


# COMMAND ----------


# Step 15.52: Correct and verify the Databricks notebook path

from databricks.sdk import WorkspaceClient

print("=" * 65)
print("DATABRICKS JOB NOTEBOOK PATH CORRECTION")
print("=" * 65)

w = WorkspaceClient()

# --------------------------------------------------
# 1. Set the actual notebook path
# --------------------------------------------------

JOB_NOTEBOOK_PATH = (
    "/Telco_Customer_Churn_ML/"
    "15_Telco_Production_Model_Deployment"
)

# --------------------------------------------------
# 2. Update the existing job definition
# --------------------------------------------------

JOB_DEFINITION["tasks"][0]["notebook_task"][
    "notebook_path"
] = JOB_NOTEBOOK_PATH

# --------------------------------------------------
# 3. Verify notebook exists
# --------------------------------------------------

notebook_info = w.workspace.get_status(
    path=JOB_NOTEBOOK_PATH
)

object_type = str(notebook_info.object_type)

assert object_type.upper().endswith("NOTEBOOK"), (
    f"Expected a notebook, found: {object_type}"
)

# --------------------------------------------------
# 4. Verify updated job definition
# --------------------------------------------------

configured_path = (
    JOB_DEFINITION["tasks"][0]
    ["notebook_task"]["notebook_path"]
)

assert configured_path == JOB_NOTEBOOK_PATH

assert JOB_DEFINITION["max_concurrent_runs"] == 1

assert JOB_DEFINITION["queue"]["enabled"] is True

# --------------------------------------------------
# 5. Display results
# --------------------------------------------------

print("\nNOTEBOOK PATH VERIFICATION")

print("Notebook path:", JOB_NOTEBOOK_PATH)
print("Workspace object type:", object_type)
print("Notebook object ID:", notebook_info.object_id)

print("\nJOB DEFINITION VERIFICATION")

print("Configured path matches:", configured_path == JOB_NOTEBOOK_PATH)
print("Maximum concurrent runs:", JOB_DEFINITION["max_concurrent_runs"])
print("Queue enabled:", JOB_DEFINITION["queue"]["enabled"])

print("\nDATABRICKS NOTEBOOK PATH CORRECTION: PASS")


# COMMAND ----------


# Step 15.53: Discover serverless Jobs compute availability

from databricks.sdk import WorkspaceClient

print("=" * 65)
print("SERVERLESS JOBS COMPUTE DISCOVERY")
print("=" * 65)

w = WorkspaceClient()

# --------------------------------------------------
# 1. Check workspace serverless feature settings
# --------------------------------------------------

print("\nWORKSPACE SERVERLESS SETTINGS")

serverless_setting = None

try:
    setting_response = (
        w.settings.serverless_compute.get()
    )

    serverless_setting = setting_response

    print("Serverless compute setting:")
    print(setting_response)

except Exception as error:
    print("Serverless setting not directly available.")
    print("Details:", str(error)[:400])


# --------------------------------------------------
# 2. Check available serverless environments
# --------------------------------------------------

print("\nSERVERLESS ENVIRONMENT DISCOVERY")

serverless_environments = []

try:
    serverless_environments = list(
        w.environments.list()
    )

    print(
        "Serverless environments found:",
        len(serverless_environments)
    )

    for environment in serverless_environments[:10]:
        print(environment)

except Exception as error:
    print("Environment discovery unavailable.")
    print("Details:", str(error)[:400])


# --------------------------------------------------
# 3. Verify job notebook configuration
# --------------------------------------------------

print("\nJOB CONFIGURATION CHECK")

task = JOB_DEFINITION["tasks"][0]

assert task["notebook_task"]["source"] == "WORKSPACE"

assert (
    task["notebook_task"]["notebook_path"]
    == JOB_NOTEBOOK_PATH
)

assert JOB_DEFINITION["max_concurrent_runs"] == 1

print("Notebook path:", JOB_NOTEBOOK_PATH)
print("Notebook source:", task["notebook_task"]["source"])
print("Maximum concurrent runs:", JOB_DEFINITION["max_concurrent_runs"])

# --------------------------------------------------
# 4. Display discovery summary
# --------------------------------------------------

print("\nSERVERLESS DISCOVERY SUMMARY")

print(
    "Workspace serverless setting retrieved:",
    serverless_setting is not None
)

print(
    "Environment listing count:",
    len(serverless_environments)
)

print("No jobs or compute resources were created.")

print("\nSERVERLESS JOBS DISCOVERY: COMPLETE")


# COMMAND ----------


# Step 15.54: Inspect Databricks SDK and serverless capabilities

import databricks.sdk
from importlib.metadata import version, PackageNotFoundError
from databricks.sdk import WorkspaceClient

print("=" * 65)
print("DATABRICKS SDK AND SERVERLESS CAPABILITY CHECK")
print("=" * 65)

w = WorkspaceClient()

# --------------------------------------------------
# 1. Inspect installed SDK version
# --------------------------------------------------

print("\nDATABRICKS SDK VERSION")

try:
    sdk_version = version("databricks-sdk")
except PackageNotFoundError:
    sdk_version = "Unknown"

print("Installed SDK version:", sdk_version)


# --------------------------------------------------
# 2. Inspect Jobs API
# --------------------------------------------------

print("\nJOBS API CAPABILITIES")

jobs_methods = [
    name for name in dir(w.jobs)
    if not name.startswith("_")
]

for method_name in [
    "create",
    "get",
    "list",
    "run_now",
    "submit",
    "validate"
]:
    print(
        f"{method_name}:",
        method_name in jobs_methods
    )


# --------------------------------------------------
# 3. Inspect available SDK job models
# --------------------------------------------------

print("\nJOB MODEL SUPPORT")

from databricks.sdk.service import jobs

model_names = [
    "JobSettings",
    "Task",
    "NotebookTask",
    "QueueSettings",
    "Environment",
    "JobEnvironment"
]

for model_name in model_names:
    print(
        f"{model_name}:",
        hasattr(jobs, model_name)
    )


# --------------------------------------------------
# 4. Inspect workspace compute configuration
# --------------------------------------------------

print("\nWORKSPACE COMPUTE CAPABILITIES")

try:
    workspace_config = w.workspace_conf.get_status()

    if hasattr(workspace_config, "to_dict"):
        config_dict = workspace_config.to_dict()
    else:
        config_dict = dict(workspace_config)

    serverless_related = {
        key: value
        for key, value in config_dict.items()
        if "serverless" in key.lower()
    }

    print(
        "Serverless-related configuration:",
        serverless_related
    )

    if not serverless_related:
        print(
            "No explicit serverless configuration "
            "was returned."
        )

except Exception as error:
    print("Workspace configuration unavailable.")
    print("Details:", str(error)[:400])


# --------------------------------------------------
# 5. Verify existing job definition
# --------------------------------------------------

print("\nJOB DEFINITION READINESS")

assert JOB_DEFINITION["max_concurrent_runs"] == 1

assert JOB_DEFINITION["queue"]["enabled"] is True

assert (
    JOB_DEFINITION["tasks"][0]["notebook_task"][
        "notebook_path"
    ] == JOB_NOTEBOOK_PATH
)

print("Notebook path:", JOB_NOTEBOOK_PATH)
print("Concurrent runs:", JOB_DEFINITION["max_concurrent_runs"])
print("Queue enabled:", JOB_DEFINITION["queue"]["enabled"])

print("\nSDK CAPABILITY DISCOVERY: COMPLETE")


# COMMAND ----------


# Step 15.55: Workspace configuration and Jobs API discovery

from databricks.sdk import WorkspaceClient

print("=" * 65)
print("WORKSPACE SERVERLESS AND JOBS ACCESS CHECK")
print("=" * 65)

w = WorkspaceClient()

# --------------------------------------------------
# 1. Inspect serverless-related workspace settings
# --------------------------------------------------

print("\nWORKSPACE SERVERLESS CONFIGURATION")

candidate_keys = [
    "enableServerlessCompute",
    "enableServerlessJobs",
    "enableServerlessWorkflows"
]

for key in candidate_keys:
    try:
        response = w.workspace_conf.get_status(
            keys=key
        )

        print(f"{key}: {response}")

    except Exception as error:
        print(
            f"{key}: Not confirmed - "
            f"{str(error)[:200]}"
        )


# --------------------------------------------------
# 2. Check existing Jobs API access
# --------------------------------------------------

print("\nDATABRICKS JOBS ACCESS")

try:
    existing_jobs = []

    for job in w.jobs.list():
        existing_jobs.append(job)

        if len(existing_jobs) >= 10:
            break

    print(
        "Jobs returned (up to 10):",
        len(existing_jobs)
    )

    for job in existing_jobs:
        print(
            "Job ID:",
            job.job_id,
            "| Name:",
            job.settings.name if job.settings else "Unknown"
        )

    print("Jobs API listing: SUCCESS")

except Exception as error:
    print("Jobs API listing: UNAVAILABLE")
    print("Details:", str(error)[:500])


# --------------------------------------------------
# 3. Inspect job environment model
# --------------------------------------------------

print("\nJOB ENVIRONMENT SUPPORT")

from databricks.sdk.service import jobs

print(
    "JobEnvironment model available:",
    hasattr(jobs, "JobEnvironment")
)

print(
    "QueueSettings model available:",
    hasattr(jobs, "QueueSettings")
)

print(
    "NotebookTask model available:",
    hasattr(jobs, "NotebookTask")
)


# --------------------------------------------------
# 4. Final summary
# --------------------------------------------------

print("\nDISCOVERY SUMMARY")

print("SDK version: 0.122.0")
print("Notebook path:", JOB_NOTEBOOK_PATH)
print("Traditional clusters previously found: 0")
print("No job created or executed.")

print("\nWORKSPACE SERVERLESS DISCOVERY: COMPLETE")


# COMMAND ----------


# Step 15.56: Inspect existing Databricks Jobs compute

from databricks.sdk import WorkspaceClient

print("=" * 65)
print("EXISTING DATABRICKS JOB COMPUTE INSPECTION")
print("=" * 65)

w = WorkspaceClient()

JOB_IDS_TO_INSPECT = [
    738907313537928,
    922096382896462,
    557580375973778
]

for job_id in JOB_IDS_TO_INSPECT:

    print("\n" + "-" * 65)
    print("JOB ID:", job_id)
    print("-" * 65)

    try:
        job = w.jobs.get(job_id=job_id)

        settings = job.settings

        print("Job name:", settings.name)

        tasks = settings.tasks or []

        print("Number of tasks:", len(tasks))
        print(
            "Job environments:",
            len(settings.environments or [])
        )

        for task in tasks:

            print("\nTask key:", task.task_key)

            if task.notebook_task:
                print(
                    "Notebook path:",
                    task.notebook_task.notebook_path
                )

            if task.existing_cluster_id:
                print("Compute type: EXISTING CLUSTER")
                print(
                    "Cluster ID:",
                    task.existing_cluster_id
                )

            elif task.new_cluster:
                print("Compute type: NEW JOB CLUSTER")
                print(
                    "Spark version:",
                    task.new_cluster.spark_version
                )
                print(
                    "Node type:",
                    task.new_cluster.node_type_id
                )

            elif task.job_cluster_key:
                print("Compute type: SHARED JOB CLUSTER")
                print(
                    "Job cluster key:",
                    task.job_cluster_key
                )

            elif task.environment_key:
                print("Compute type: SERVERLESS CANDIDATE")
                print(
                    "Environment key:",
                    task.environment_key
                )

            else:
                print(
                    "Compute type: NO EXPLICIT "
                    "CLUSTER CONFIGURATION"
                )
                print(
                    "Serverless may be used; "
                    "additional verification required."
                )

        if settings.job_clusters:
            print(
                "\nJob cluster definitions:",
                len(settings.job_clusters)
            )

    except Exception as error:
        print("Inspection failed.")
        print("Details:", str(error)[:500])


print("\n" + "=" * 65)
print("EXISTING JOB COMPUTE INSPECTION: COMPLETE")
print("=" * 65)


# COMMAND ----------


# Step 15.57: Verify existing serverless job execution

from databricks.sdk import WorkspaceClient

print("=" * 65)
print("SERVERLESS JOB EXECUTION VERIFICATION")
print("=" * 65)

w = WorkspaceClient()

REFERENCE_JOB_ID = 738907313537928

# --------------------------------------------------
# 1. Retrieve reference job
# --------------------------------------------------

job = w.jobs.get(job_id=REFERENCE_JOB_ID)
settings = job.settings

print("\nREFERENCE JOB")

print("Job name:", settings.name)
print("Job ID:", REFERENCE_JOB_ID)

# --------------------------------------------------
# 2. Inspect environment configuration
# --------------------------------------------------

print("\nJOB ENVIRONMENT CONFIGURATION")

for environment in settings.environments or []:

    print("Environment key:", environment.environment_key)

    if environment.spec:
        print(
            "Environment specification:",
            environment.spec.as_dict()
        )
    else:
        print("Environment specification: None")

# --------------------------------------------------
# 3. Inspect task configuration
# --------------------------------------------------

print("\nTASK COMPUTE CONFIGURATION")

for task in settings.tasks or []:

    print("Task key:", task.task_key)
    print("Environment key:", task.environment_key)
    print("Existing cluster:", task.existing_cluster_id)
    print("New cluster:", task.new_cluster is not None)
    print("Job cluster key:", task.job_cluster_key)

# --------------------------------------------------
# 4. Inspect recent job executions
# --------------------------------------------------

print("\nRECENT JOB EXECUTIONS")

successful_runs = 0
runs_found = 0

try:
    for run in w.jobs.list_runs(
        job_id=REFERENCE_JOB_ID,
        limit=10
    ):

        runs_found += 1

        life_cycle = (
            str(run.state.life_cycle_state)
            if run.state else "UNKNOWN"
        )

        result_state = (
            str(run.state.result_state)
            if run.state else "UNKNOWN"
        )

        print(
            "Run ID:", run.run_id,
            "| Lifecycle:", life_cycle,
            "| Result:", result_state
        )

        if result_state.upper().endswith("SUCCESS"):
            successful_runs += 1

except Exception as error:
    print("Run history inspection unavailable.")
    print("Details:", str(error)[:500])

# --------------------------------------------------
# 5. Summary
# --------------------------------------------------

print("\nSERVERLESS EXECUTION SUMMARY")

print("Runs inspected:", runs_found)
print("Successful runs:", successful_runs)

if successful_runs > 0:
    print("Reference job has successful execution history.")
else:
    print(
        "Successful execution not established "
        "from the inspected runs."
    )

print("No jobs were created or executed.")

print("\nSERVERLESS JOB VERIFICATION: COMPLETE")


# COMMAND ----------


# Step 15.58: Configure serverless Jobs compute

import json

print("=" * 65)
print("SERVERLESS JOB COMPUTE CONFIGURATION")
print("=" * 65)

# --------------------------------------------------
# 1. Define serverless environment
# --------------------------------------------------

JOB_ENVIRONMENT_KEY = (
    "telco_snapshot_inference_environment"
)

JOB_ENVIRONMENT_VERSION = "6"

# --------------------------------------------------
# 2. Configure job environment
# --------------------------------------------------

JOB_DEFINITION["environments"] = [
    {
        "environment_key": JOB_ENVIRONMENT_KEY,
        "spec": {
            "environment_version": JOB_ENVIRONMENT_VERSION
        }
    }
]

# --------------------------------------------------
# 3. Configure task environment
# --------------------------------------------------

task = JOB_DEFINITION["tasks"][0]

task["environment_key"] = JOB_ENVIRONMENT_KEY

# Remove any traditional cluster configuration
for key in [
    "existing_cluster_id",
    "new_cluster",
    "job_cluster_key"
]:
    task.pop(key, None)

# --------------------------------------------------
# 4. Validate serverless configuration
# --------------------------------------------------

assert len(JOB_DEFINITION["environments"]) == 1

assert (
    JOB_DEFINITION["environments"][0]["environment_key"]
    == JOB_ENVIRONMENT_KEY
)

assert (
    JOB_DEFINITION["environments"][0]["spec"][
        "environment_version"
    ] == "6"
)

assert task["environment_key"] == JOB_ENVIRONMENT_KEY

assert "existing_cluster_id" not in task
assert "new_cluster" not in task
assert "job_cluster_key" not in task

assert (
    task["notebook_task"]["notebook_path"]
    == JOB_NOTEBOOK_PATH
)

# --------------------------------------------------
# 5. Display configuration
# --------------------------------------------------

print("\nSERVERLESS ENVIRONMENT")

print("Environment key:", JOB_ENVIRONMENT_KEY)
print("Environment version:", JOB_ENVIRONMENT_VERSION)

print("\nTASK COMPUTE")

print("Task key:", task["task_key"])
print("Environment key:", task["environment_key"])
print("Existing cluster:", task.get("existing_cluster_id"))
print("New cluster:", task.get("new_cluster"))
print("Job cluster key:", task.get("job_cluster_key"))

print("\nJOB ENVIRONMENT DEFINITION")

print(
    json.dumps(
        JOB_DEFINITION["environments"],
        indent=2
    )
)

print("\nSERVERLESS JOB CONFIGURATION: PASS")


# COMMAND ----------


# Step 15.59: Inspect production workflow dependencies

print("=" * 65)
print("PRODUCTION NOTEBOOK DEPENDENCY INVENTORY")
print("=" * 65)

# --------------------------------------------------
# 1. Required production functions
# --------------------------------------------------

required_functions = [
    "load_pinned_source_snapshot",
    "calculate_source_fingerprint",
    "register_batch_identity",
    "run_batch_inference",
    "add_stable_batch_metadata",
    "validate_batch_predictions",
    "validate_strict_publication_integrity",
    "publish_quality_approved_batch",
    "execute_snapshot_protected_batch_job"
]

print("\nPRODUCTION FUNCTIONS")

missing_functions = []

for name in required_functions:

    obj = globals().get(name)

    available = callable(obj)

    print(
        f"{name}:",
        "AVAILABLE" if available else "MISSING"
    )

    if not available:
        missing_functions.append(name)


# --------------------------------------------------
# 2. Required runtime objects
# --------------------------------------------------

required_objects = [
    "spark",
    "job_preprocessing_pipeline",
    "job_registered_model",
    "JOB_SOURCE_TABLE",
    "JOB_SOURCE_VERSION",
    "JOB_MODEL_NAME",
    "JOB_MODEL_VERSION",
    "JOB_DEV_OUTPUT_TABLE",
    "JOB_BATCH_REGISTRY_TABLE",
    "JOB_EXECUTION_POLICY",
    "JOB_DEFINITION"
]

print("\nPRODUCTION RUNTIME OBJECTS")

missing_objects = []

for name in required_objects:

    available = (
        name in globals()
        and globals()[name] is not None
    )

    print(
        f"{name}:",
        "AVAILABLE" if available else "MISSING"
    )

    if not available:
        missing_objects.append(name)


# --------------------------------------------------
# 3. Inspect publication function integration
# --------------------------------------------------

print("\nSTRICT PUBLICATION INTEGRATION")

publisher = globals().get(
    "publish_quality_approved_batch"
)

strict_publisher = globals().get(
    "publish_quality_approved_batch_strict"
)

strict_publisher_active = (
    callable(publisher)
    and callable(strict_publisher)
    and publisher is strict_publisher
)

print(
    "Strict publisher active:",
    strict_publisher_active
)


# --------------------------------------------------
# 4. Dependency inventory summary
# --------------------------------------------------

print("\nDEPENDENCY INVENTORY SUMMARY")

print(
    "Required functions:",
    len(required_functions)
)

print(
    "Available functions:",
    len(required_functions) - len(missing_functions)
)

print(
    "Missing functions:",
    len(missing_functions)
)

print(
    "Required runtime objects:",
    len(required_objects)
)

print(
    "Available runtime objects:",
    len(required_objects) - len(missing_objects)
)

print(
    "Missing runtime objects:",
    len(missing_objects)
)

print(
    "Strict publication integration:",
    "ACTIVE" if strict_publisher_active else "NOT ACTIVE"
)

if missing_functions:
    print("\nMissing function names:", missing_functions)

if missing_objects:
    print("\nMissing object names:", missing_objects)

print("\nDEPENDENCY INVENTORY: COMPLETE")
print(
    "Fresh-session execution has NOT yet been verified."
)


# COMMAND ----------


# Step 15.60: Inspect saved notebook execution structure

import base64
import re

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

print("=" * 65)
print("NOTEBOOK EXECUTION STRUCTURE INSPECTION")
print("=" * 65)

w = WorkspaceClient()

# --------------------------------------------------
# 1. Export saved notebook source
# --------------------------------------------------

export_result = w.workspace.export(
    path=JOB_NOTEBOOK_PATH,
    format=ExportFormat.SOURCE
)

assert export_result.content is not None, (
    "Notebook export returned no content."
)

notebook_source = base64.b64decode(
    export_result.content
).decode("utf-8")

print("\nNOTEBOOK SOURCE")

print("Notebook path:", JOB_NOTEBOOK_PATH)
print("Source characters:", len(notebook_source))

# --------------------------------------------------
# 2. Split notebook into cells
# --------------------------------------------------

cells = re.split(
    r"(?m)^# COMMAND ----------\s*$",
    notebook_source
)

print("Exported cell sections:", len(cells))

# --------------------------------------------------
# 3. Locate required function definitions
# --------------------------------------------------

print("\nREQUIRED FUNCTION DEFINITIONS")

required_functions = [
    "load_pinned_source_snapshot",
    "calculate_source_fingerprint",
    "register_batch_identity",
    "run_batch_inference",
    "add_stable_batch_metadata",
    "validate_batch_predictions",
    "validate_strict_publication_integrity",
    "publish_quality_approved_batch",
    "execute_snapshot_protected_batch_job"
]

for function_name in required_functions:

    pattern = (
        r"(?m)^\s*(?:async\s+)?def\s+"
        + re.escape(function_name)
        + r"\s*\("
    )

    matches = [
        index + 1
        for index, cell in enumerate(cells)
        if re.search(pattern, cell)
    ]

    print(
        f"{function_name}:",
        f"Cell sections {matches}"
        if matches else "NOT FOUND"
    )

# --------------------------------------------------
# 4. Inspect initialization references
# --------------------------------------------------

print("\nMODEL AND PIPELINE INITIALIZATION REFERENCES")

initialization_terms = [
    "PipelineModel.load",
    "mlflow.spark.load_model",
    "models:/",
    "job_preprocessing_pipeline",
    "job_registered_model"
]

for term in initialization_terms:

    matching_cells = [
        index + 1
        for index, cell in enumerate(cells)
        if term in cell
    ]

    print(
        f"{term}:",
        matching_cells[:20]
    )

# --------------------------------------------------
# 5. Identify potential execution side effects
# --------------------------------------------------

print("\nPOTENTIAL EXECUTION SIDE EFFECTS")

side_effect_terms = [
    "execute_snapshot_protected_batch_job(",
    "saveAsTable(",
    "merge(",
    "w.jobs.create(",
    "dbutils.notebook.exit("
]

for term in side_effect_terms:

    matching_cells = [
        index + 1
        for index, cell in enumerate(cells)
        if term in cell
    ]

    print(
        f"{term}:",
        matching_cells[:20]
    )

# --------------------------------------------------
# 6. Summary
# --------------------------------------------------

print("\nNOTEBOOK STRUCTURE SUMMARY")

print("Notebook source exported: True")
print("Exported cell sections:", len(cells))
print("Inspection mode: READ ONLY")

print("\nNOTEBOOK STRUCTURE INSPECTION: COMPLETE")
print(
    "Fresh-session execution has NOT yet been verified."
)


# COMMAND ----------


# Step 15.61: Identify top-level execution statements

import ast
import re

print("=" * 65)
print("NOTEBOOK TOP-LEVEL EXECUTION ANALYSIS")
print("=" * 65)

# notebook_source was exported in Step 15.60

cells = re.split(
    r"(?m)^# COMMAND ----------\s*$",
    notebook_source
)

target_operations = [
    "execute_snapshot_protected_batch_job",
    "execute_identity_protected_batch_job",
    "saveAsTable",
    "merge",
    "register_batch_identity",
    "publish_quality_approved_batch",
    "publish_quality_approved_batch_strict"
]

print("\nTOP-LEVEL OPERATION INVENTORY")

operation_count = 0
parse_failures = []

for cell_number, cell_source in enumerate(cells, start=1):

    # Databricks SOURCE exports may contain magic commands.
    # Ignore comment-only lines and convert simple magic
    # directives into comments for this static inspection.

    cleaned_lines = []

    for line in cell_source.splitlines():

        stripped = line.lstrip()

        if stripped.startswith(("%", "!")):
            cleaned_lines.append("# " + line)
        else:
            cleaned_lines.append(line)

    cleaned_source = "\n".join(cleaned_lines)

    try:
        tree = ast.parse(cleaned_source)

    except SyntaxError as error:
        parse_failures.append(
            (cell_number, str(error)[:120])
        )
        continue

    # Inspect executable top-level statements.
    # Exclude function and class bodies, but include
    # calls nested inside top-level control flow.

    class TopLevelCallVisitor(ast.NodeVisitor):

        def visit_FunctionDef(self, node):
            pass

        def visit_AsyncFunctionDef(self, node):
            pass

        def visit_ClassDef(self, node):
            pass

        def visit_Lambda(self, node):
            pass

        def visit_Call(self, node):

            global operation_count

            if isinstance(node.func, ast.Name):
                operation_name = node.func.id

            elif isinstance(node.func, ast.Attribute):
                operation_name = node.func.attr

            else:
                operation_name = None

            if operation_name in target_operations:

                operation_count += 1

                print(
                    f"Section {cell_number} | "
                    f"Line {node.lineno} | "
                    f"{operation_name}"
                )

            self.generic_visit(node)

    visitor = TopLevelCallVisitor()

    for statement in tree.body:
        visitor.visit(statement)


print("\nANALYSIS SUMMARY")

print("Exported sections:", len(cells))
print("Top-level target calls found:", operation_count)
print("Sections not parsed:", len(parse_failures))

if parse_failures:
    print("\nUNPARSED SECTIONS")

    for cell_number, error in parse_failures[:15]:
        print(f"Section {cell_number}: {error}")

print("\nTOP-LEVEL EXECUTION ANALYSIS: COMPLETE")
print("No notebook code was executed.")


# COMMAND ----------


# Step 15.62: Analyze production function dependencies

import ast
import re
import builtins

print("=" * 65)
print("PRODUCTION FUNCTION DEPENDENCY ANALYSIS")
print("=" * 65)

cells = re.split(
    r"(?m)^# COMMAND ----------\s*$",
    notebook_source
)

required_functions = [
    "load_pinned_source_snapshot",
    "calculate_source_fingerprint",
    "register_batch_identity",
    "run_batch_inference",
    "add_stable_batch_metadata",
    "validate_batch_predictions",
    "validate_strict_publication_integrity",
    "publish_quality_approved_batch",
    "execute_snapshot_protected_batch_job"
]

# --------------------------------------------------
# 1. Collect notebook function definitions
# --------------------------------------------------

function_definitions = {}

for section_number, source in enumerate(cells, start=1):

    tree = ast.parse(source)

    for node in tree.body:

        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):

            function_definitions.setdefault(
                node.name, []
            ).append((section_number, node))

print("\nFUNCTION DEFINITION LOCATIONS")

for name in required_functions:

    locations = [
        section
        for section, node
        in function_definitions.get(name, [])
    ]

    print(f"{name}: {locations}")

# --------------------------------------------------
# 2. Inspect global name references
# --------------------------------------------------

print("\nPRODUCTION FUNCTION DEPENDENCIES")

builtin_names = set(dir(builtins))
all_function_names = set(function_definitions)

for function_name in required_functions:

    definitions = function_definitions.get(
        function_name, []
    )

    if not definitions:
        print(f"\n{function_name}: NOT FOUND")
        continue

    section_number, node = definitions[-1]

    # Collect names referenced in the function.
    loaded_names = {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name)
        and isinstance(item.ctx, ast.Load)
    }

    # Collect names assigned within the function.
    assigned_names = {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name)
        and isinstance(item.ctx, (
            ast.Store,
            ast.Del
        ))
    }

    parameter_names = {
        argument.arg
        for argument in (
            node.args.posonlyargs
            + node.args.args
            + node.args.kwonlyargs
        )
    }

    if node.args.vararg:
        parameter_names.add(
            node.args.vararg.arg
        )

    if node.args.kwarg:
        parameter_names.add(
            node.args.kwarg.arg
        )

    external_names = (
        loaded_names
        - assigned_names
        - parameter_names
        - builtin_names
    )

    helper_functions = sorted(
        external_names & all_function_names
    )

    other_dependencies = sorted(
        external_names - all_function_names
    )

    print(f"\nFUNCTION: {function_name}")
    print("Defined in section:", section_number)

    print(
        "Helper function references:",
        helper_functions
    )

    print(
        "Other external name candidates:",
        other_dependencies
    )

# --------------------------------------------------
# 3. Summary
# --------------------------------------------------

print("\nDEPENDENCY ANALYSIS SUMMARY")

print(
    "Required production functions:",
    len(required_functions)
)

print(
    "Functions found:",
    sum(
        bool(function_definitions.get(name))
        for name in required_functions
    )
)

print(
    "Total distinct notebook functions:",
    len(function_definitions)
)

print(
    "Analysis mode: READ ONLY"
)

print("\nPRODUCTION DEPENDENCY ANALYSIS: COMPLETE")


# COMMAND ----------


# Step 15.63: Inspect production function source boundaries

import ast
import re

print("=" * 65)
print("PRODUCTION FUNCTION SOURCE INSPECTION")
print("=" * 65)

cells = re.split(
    r"(?m)^# COMMAND ----------\s*$",
    notebook_source
)

required_functions = [
    "load_pinned_source_snapshot",
    "calculate_source_fingerprint",
    "register_batch_identity",
    "run_batch_inference",
    "add_stable_batch_metadata",
    "validate_batch_predictions",
    "validate_strict_publication_integrity",
    "publish_quality_approved_batch",
    "execute_snapshot_protected_batch_job"
]

function_inventory = {}

# --------------------------------------------------
# 1. Locate exact function source boundaries
# --------------------------------------------------

for section_number, source in enumerate(cells, start=1):

    tree = ast.parse(source)

    source_lines = source.splitlines(
        keepends=True
    )

    for node in tree.body:

        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue

        if node.name not in required_functions:
            continue

        start_line = min(
            [node.lineno]
            + [
                decorator.lineno
                for decorator in node.decorator_list
            ]
        )

        end_line = node.end_lineno

        function_source = "".join(
            source_lines[start_line - 1:end_line]
        )

        function_inventory.setdefault(
            node.name, []
        ).append({
            "section": section_number,
            "start_line": start_line,
            "end_line": end_line,
            "source": function_source,
            "node": node
        })

# --------------------------------------------------
# 2. Inspect function definitions
# --------------------------------------------------

print("\nFUNCTION SOURCE INVENTORY")

for function_name in required_functions:

    definitions = function_inventory.get(
        function_name, []
    )

    print("\nFunction:", function_name)
    print("Definitions found:", len(definitions))

    for definition in definitions:

        node = definition["node"]

        print(
            "Section:",
            definition["section"]
        )

        print(
            "Source lines:",
            definition["start_line"],
            "to",
            definition["end_line"]
        )

        print(
            "Decorators:",
            len(node.decorator_list)
        )

        print(
            "Positional defaults:",
            len(node.args.defaults)
        )

        print(
            "Keyword defaults:",
            sum(
                value is not None
                for value in node.args.kw_defaults
            )
        )

        nested_functions = [
            item.name
            for item in ast.walk(node)
            if isinstance(
                item,
                (ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and item is not node
        ]

        print(
            "Nested functions:",
            nested_functions
        )

# --------------------------------------------------
# 3. Check for duplicate definitions
# --------------------------------------------------

print("\nDUPLICATE FUNCTION CHECK")

duplicate_functions = [
    name
    for name, definitions
    in function_inventory.items()
    if len(definitions) > 1
]

print(
    "Duplicate required functions:",
    duplicate_functions
)

# --------------------------------------------------
# 4. Summary
# --------------------------------------------------

print("\nSOURCE INSPECTION SUMMARY")

print(
    "Required functions:",
    len(required_functions)
)

print(
    "Functions located:",
    len(function_inventory)
)

print(
    "Duplicate definitions:",
    len(duplicate_functions)
)

print(
    "Source extraction performed:",
    "IN MEMORY ONLY"
)

print(
    "Notebook modifications:",
    "NONE"
)

print("\nPRODUCTION FUNCTION SOURCE INSPECTION: COMPLETE")


# COMMAND ----------

def predict(threshold=0.5):
    ...

# COMMAND ----------

def predict(threshold=JOB_THRESHOLD):
    ...

# COMMAND ----------


# Step 15.64: Inspect definition-time dependencies

import ast

print("=" * 65)
print("FUNCTION DEFAULT AND ANNOTATION VALIDATION")
print("=" * 65)

definition_time_issues = []

print("\nFUNCTION DEFINITION-TIME EXPRESSIONS")

for function_name in required_functions:

    definitions = function_inventory.get(
        function_name, []
    )

    for definition in definitions:

        node = definition["node"]
        section = definition["section"]

        expressions = []

        # Positional and keyword defaults
        expressions.extend(
            ("positional default", value)
            for value in node.args.defaults
        )

        expressions.extend(
            ("keyword default", value)
            for value in node.args.kw_defaults
            if value is not None
        )

        # Parameter annotations
        arguments = (
            node.args.posonlyargs
            + node.args.args
            + node.args.kwonlyargs
        )

        if node.args.vararg:
            arguments.append(node.args.vararg)

        if node.args.kwarg:
            arguments.append(node.args.kwarg)

        expressions.extend(
            ("parameter annotation", arg.annotation)
            for arg in arguments
            if arg.annotation is not None
        )

        if node.returns is not None:
            expressions.append(
                ("return annotation", node.returns)
            )

        if not expressions:
            print(
                f"{function_name}: "
                "No definition-time expressions"
            )
            continue

        print(f"\nFunction: {function_name}")
        print("Section:", section)

        for expression_type, expression in expressions:

            expression_text = ast.unparse(expression)

            print(
                f"{expression_type}: "
                f"{expression_text}"
            )

            # Literal constants are safe to evaluate
            # without external runtime variables.
            try:
                ast.literal_eval(expression)
                literal_safe = True
            except (ValueError, TypeError, SyntaxError,
                    MemoryError, RecursionError):
                literal_safe = False

            print("Literal-safe:", literal_safe)

            if not literal_safe:
                definition_time_issues.append({
                    "function": function_name,
                    "section": section,
                    "type": expression_type,
                    "expression": expression_text
                })


print("\nVALIDATION SUMMARY")

print(
    "Required functions inspected:",
    len(required_functions)
)

print(
    "Non-literal definition-time expressions:",
    len(definition_time_issues)
)

for issue in definition_time_issues:
    print(
        "REVIEW:",
        issue["function"],
        "|",
        issue["type"],
        "|",
        issue["expression"]
    )

print("\nFUNCTION DEFINITION VALIDATION: COMPLETE")
print("No notebook code was executed.")


# COMMAND ----------


# Step 15.65: Assemble production function source in memory

import ast

print("=" * 65)
print("PRODUCTION FUNCTION MODULE ASSEMBLY")
print("=" * 65)

# --------------------------------------------------
# 1. Define function execution order
# --------------------------------------------------

PRODUCTION_FUNCTION_ORDER = [
    "load_pinned_source_snapshot",
    "calculate_source_fingerprint",
    "register_batch_identity",
    "run_batch_inference",
    "add_stable_batch_metadata",
    "validate_batch_predictions",
    "validate_strict_publication_integrity",
    "publish_quality_approved_batch",
    "execute_snapshot_protected_batch_job"
]

# --------------------------------------------------
# 2. Extract verified function definitions
# --------------------------------------------------

production_function_sources = []

for function_name in PRODUCTION_FUNCTION_ORDER:

    definitions = function_inventory.get(
        function_name, []
    )

    assert len(definitions) == 1, (
        f"Expected one definition for {function_name}"
    )

    source = definitions[0]["source"]

    # Validate extracted source
    parsed = ast.parse(source)

    assert len(parsed.body) == 1
    assert isinstance(
        parsed.body[0],
        (ast.FunctionDef, ast.AsyncFunctionDef)
    )

    assert parsed.body[0].name == function_name

    production_function_sources.append(source)

# --------------------------------------------------
# 3. Assemble module source
# --------------------------------------------------

PRODUCTION_FUNCTION_MODULE = (
    "\n\n\n".join(production_function_sources)
)

# --------------------------------------------------
# 4. Validate assembled module
# --------------------------------------------------

module_tree = ast.parse(
    PRODUCTION_FUNCTION_MODULE
)

module_functions = [
    node.name
    for node in module_tree.body
    if isinstance(
        node,
        (ast.FunctionDef, ast.AsyncFunctionDef)
    )
]

assert module_functions == PRODUCTION_FUNCTION_ORDER

assert all(
    isinstance(
        node,
        (ast.FunctionDef, ast.AsyncFunctionDef)
    )
    for node in module_tree.body
)

# --------------------------------------------------
# 5. Inspect module summary
# --------------------------------------------------

print("\nASSEMBLED FUNCTIONS")

for index, name in enumerate(
    module_functions,
    start=1
):
    print(f"{index}. {name}")

print("\nMODULE VALIDATION")

print(
    "Functions assembled:",
    len(module_functions)
)

print(
    "Module source characters:",
    len(PRODUCTION_FUNCTION_MODULE)
)

print(
    "Module syntax:",
    "PASS"
)

print(
    "Top-level executable statements:",
    0
)

print(
    "Module saved to workspace:",
    False
)

print(
    "Production inference executed:",
    False
)

print("\nPRODUCTION FUNCTION MODULE ASSEMBLY: PASS")


# COMMAND ----------


# Step 15.66: Assemble strict publication integration

import ast
import re

print("=" * 65)
print("STRICT PUBLICATION MODULE INTEGRATION")
print("=" * 65)

# --------------------------------------------------
# 1. Locate the strict publisher definition
# --------------------------------------------------

STRICT_WRAPPER_NAME = (
    "publish_quality_approved_batch_strict"
)

wrapper_matches = []

for section_number, source in enumerate(cells, start=1):

    tree = ast.parse(source)
    source_lines = source.splitlines(keepends=True)

    for node in tree.body:

        if (
            isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and node.name == STRICT_WRAPPER_NAME
        ):

            start_line = min(
                [node.lineno]
                + [
                    decorator.lineno
                    for decorator in node.decorator_list
                ]
            )

            wrapper_source = "".join(
                source_lines[
                    start_line - 1:node.end_lineno
                ]
            )

            wrapper_matches.append({
                "section": section_number,
                "source": wrapper_source
            })

assert len(wrapper_matches) == 1, (
    "Expected exactly one strict publisher definition. "
    f"Found: {len(wrapper_matches)}"
)

STRICT_WRAPPER_SOURCE = wrapper_matches[0]["source"]

print("\nSTRICT WRAPPER SOURCE")

print(
    "Function:",
    STRICT_WRAPPER_NAME
)

print(
    "Defined in section:",
    wrapper_matches[0]["section"]
)

print(
    "Source characters:",
    len(STRICT_WRAPPER_SOURCE)
)

# --------------------------------------------------
# 2. Define explicit strict publisher integration
# --------------------------------------------------

STRICT_PUBLISHER_INTEGRATION = """
# Preserve the original Delta publisher.
_original_publish_quality_approved_batch = (
    publish_quality_approved_batch
)

# Activate strict publication validation.
publish_quality_approved_batch = (
    publish_quality_approved_batch_strict
)
"""

# --------------------------------------------------
# 3. Assemble production module
# --------------------------------------------------

PRODUCTION_MODULE_WITH_STRICT_PUBLISHER = (
    PRODUCTION_FUNCTION_MODULE
    + "\n\n\n"
    + STRICT_WRAPPER_SOURCE
    + "\n\n"
    + STRICT_PUBLISHER_INTEGRATION
)

# --------------------------------------------------
# 4. Validate assembled source
# --------------------------------------------------

module_tree = ast.parse(
    PRODUCTION_MODULE_WITH_STRICT_PUBLISHER
)

module_function_names = [
    node.name
    for node in module_tree.body
    if isinstance(
        node,
        (ast.FunctionDef, ast.AsyncFunctionDef)
    )
]

assert len(module_function_names) == 10

assert (
    module_function_names[-1]
    == STRICT_WRAPPER_NAME
)

assert (
    "_original_publish_quality_approved_batch"
    in PRODUCTION_MODULE_WITH_STRICT_PUBLISHER
)

# Verify the wrapper references the validator
# and original publisher.

wrapper_tree = ast.parse(STRICT_WRAPPER_SOURCE)

wrapper_names = {
    node.id
    for node in ast.walk(wrapper_tree)
    if isinstance(node, ast.Name)
}

assert (
    "validate_strict_publication_integrity"
    in wrapper_names
)

assert (
    "_original_publish_quality_approved_batch"
    in wrapper_names
)

# --------------------------------------------------
# 5. Summary
# --------------------------------------------------

print("\nSTRICT PUBLICATION VALIDATION")

print("Production functions:", len(module_function_names))

print(
    "Strict wrapper included:",
    STRICT_WRAPPER_NAME in module_function_names
)

print(
    "Integrity validator referenced:",
    True
)

print(
    "Original publisher preserved:",
    True
)

print(
    "Strict publisher activation included:",
    True
)

print(
    "Module syntax:",
    "PASS"
)

print(
    "Module source characters:",
    len(PRODUCTION_MODULE_WITH_STRICT_PUBLISHER)
)

print(
    "Module saved to workspace:",
    False
)

print(
    "Delta publication executed:",
    False
)

print("\nSTRICT PUBLICATION MODULE INTEGRATION: PASS")


# COMMAND ----------

# DBTITLE 1,Step 15.67: Assemble production imports and initialization

# Step 15.67: Assemble production imports and initialization

import ast

print("=" * 65)
print("PRODUCTION IMPORTS AND INITIALIZATION ASSEMBLY")
print("=" * 65)

# --------------------------------------------------
# 1. Define production imports
# --------------------------------------------------

PRODUCTION_IMPORTS = '''
import hashlib
import json

from datetime import datetime, timezone

import mlflow
import mlflow.spark

from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    LongType,
    IntegerType,
    DoubleType,
    TimestampType
)

from delta.tables import DeltaTable
'''

# --------------------------------------------------
# 2. Define production configuration
# --------------------------------------------------

PRODUCTION_CONFIGURATION = '''
# Production batch inference configuration

JOB_SOURCE_TABLE = "workspace.default.telco_test_raw"

# Pinned Delta source version
JOB_SOURCE_VERSION = 2

# Registered MLflow model
JOB_MODEL_NAME = (
    "workspace.default.telco_churn_logistic_regression"
)

JOB_MODEL_VERSION = 1

JOB_MODEL_URI = (
    f"models:/{JOB_MODEL_NAME}/{JOB_MODEL_VERSION}"
)

# Saved preprocessing pipeline
JOB_PIPELINE_PATH = (
    "/Volumes/workspace/default/telco_data/"
    "telco_preprocessing_pipeline"
)

# Development-only publication destination
JOB_DEV_OUTPUT_TABLE = (
    "workspace.default.telco_15_job_dev_predictions"
)

# Batch identity registry
JOB_BATCH_REGISTRY_TABLE = (
    "workspace.default.telco_15_job_batch_registry"
)

# Prediction decision threshold
JOB_THRESHOLD = 0.5
'''

# --------------------------------------------------
# 3. Define model and pipeline initialization
# --------------------------------------------------

PRODUCTION_MODEL_INITIALIZATION = '''
# Initialize production preprocessing pipeline

job_preprocessing_pipeline = PipelineModel.load(
    JOB_PIPELINE_PATH
)

# Initialize registered MLflow Spark model

job_registered_model = mlflow.spark.load_model(
    JOB_MODEL_URI
)
'''

# --------------------------------------------------
# 4. Assemble initialization source
# --------------------------------------------------

PRODUCTION_INITIALIZATION_SOURCE = (
    PRODUCTION_IMPORTS
    + "\n"
    + PRODUCTION_CONFIGURATION
    + "\n"
    + PRODUCTION_MODEL_INITIALIZATION
)

# --------------------------------------------------
# 5. Validate Python syntax
# --------------------------------------------------

initialization_tree = ast.parse(
    PRODUCTION_INITIALIZATION_SOURCE
)

assert len(initialization_tree.body) > 0

assert (
    "PipelineModel.load"
    in PRODUCTION_INITIALIZATION_SOURCE
)

assert (
    "mlflow.spark.load_model"
    in PRODUCTION_INITIALIZATION_SOURCE
)

assert (
    JOB_DEV_OUTPUT_TABLE
    in PRODUCTION_INITIALIZATION_SOURCE
)

# --------------------------------------------------
# 6. Display validation summary
# --------------------------------------------------

print("\nPRODUCTION CONFIGURATION")

print("Source table:", JOB_SOURCE_TABLE)
print("Source version:", JOB_SOURCE_VERSION)
print("Model name:", JOB_MODEL_NAME)
print("Model version:", JOB_MODEL_VERSION)

print(
    "Preprocessing pipeline:",
    "/Volumes/workspace/default/telco_data/"
    "telco_preprocessing_pipeline"
)

print("Development destination:", JOB_DEV_OUTPUT_TABLE)
print("Batch registry:", JOB_BATCH_REGISTRY_TABLE)

print("\nINITIALIZATION VALIDATION")

print(
    "Initialization source characters:",
    len(PRODUCTION_INITIALIZATION_SOURCE)
)

print("Python syntax:", "PASS")
print("Pipeline loading included:", True)
print("Registered model loading included:", True)
print("Initialization executed:", False)
print("Workspace modified:", False)

print("\nPRODUCTION INITIALIZATION ASSEMBLY: PASS")


# COMMAND ----------


# Step 15.68: Inspect and reconstruct batch registry schema

import ast

print("=" * 65)
print("BATCH REGISTRY SCHEMA VALIDATION")
print("=" * 65)

# --------------------------------------------------
# 1. Verify existing registry schema
# --------------------------------------------------

print("\nINTERACTIVE REGISTRY SCHEMA")

assert "registry_schema" in globals(), (
    "registry_schema is missing from the current session."
)

registry_schema_json = registry_schema.json()

print(
    "Schema field count:",
    len(registry_schema.fields)
)

for field in registry_schema.fields:
    print(
        field.name,
        "|",
        field.dataType.simpleString(),
        "| nullable:",
        field.nullable
    )

# --------------------------------------------------
# 2. Compare against Delta registry table
# --------------------------------------------------

print("\nDELTA REGISTRY TABLE SCHEMA")

registry_table_schema = (
    spark.table(JOB_BATCH_REGISTRY_TABLE).schema
)

print(
    "Delta table field count:",
    len(registry_table_schema.fields)
)

for field in registry_table_schema.fields:
    print(
        field.name,
        "|",
        field.dataType.simpleString(),
        "| nullable:",
        field.nullable
    )

# --------------------------------------------------
# 3. Compare field names and data types
# --------------------------------------------------

expected_fields = [
    (
        field.name,
        field.dataType.simpleString()
    )
    for field in registry_schema.fields
]

actual_fields = [
    (
        field.name,
        field.dataType.simpleString()
    )
    for field in registry_table_schema.fields
]

schema_matches = expected_fields == actual_fields

print("\nSCHEMA COMPARISON")

print(
    "Field names and types match:",
    schema_matches
)

assert schema_matches, (
    "Registry schema does not match the Delta table."
)

# --------------------------------------------------
# 4. Prepare reconstruction source
# --------------------------------------------------

PRODUCTION_REGISTRY_SCHEMA_SOURCE = (
    "from pyspark.sql.types import StructType\n\n"
    "registry_schema = StructType.fromJson("
    + repr(registry_schema.jsonValue())
    + ")\n"
)

# Validate generated Python syntax
ast.parse(PRODUCTION_REGISTRY_SCHEMA_SOURCE)

# Verify reconstructed schema
from pyspark.sql.types import StructType

reconstructed_schema = StructType.fromJson(
    registry_schema.jsonValue()
)

assert reconstructed_schema == registry_schema

# --------------------------------------------------
# 5. Summary
# --------------------------------------------------

print("\nREGISTRY SCHEMA VALIDATION")

print("Schema field count:", len(expected_fields))
print("Schema matches Delta table:", schema_matches)
print("Reconstructed schema matches:", True)
print("Production schema source prepared:", True)
print("Registry table modified:", False)

print("\nBATCH REGISTRY SCHEMA VALIDATION: PASS")


# COMMAND ----------


# Step 15.69: Integrate production registry schema

import ast

print("=" * 65)
print("COMPLETE PRODUCTION INITIALIZATION ASSEMBLY")
print("=" * 65)

# --------------------------------------------------
# 1. Combine initialization components
# --------------------------------------------------

PRODUCTION_INITIALIZATION_COMPLETE = (
    PRODUCTION_IMPORTS
    + "\n\n"
    + PRODUCTION_CONFIGURATION
    + "\n\n"
    + PRODUCTION_REGISTRY_SCHEMA_SOURCE
    + "\n\n"
    + PRODUCTION_MODEL_INITIALIZATION
)

# --------------------------------------------------
# 2. Validate Python syntax
# --------------------------------------------------

initialization_tree = ast.parse(
    PRODUCTION_INITIALIZATION_COMPLETE
)

# --------------------------------------------------
# 3. Verify component ordering
# --------------------------------------------------

import_position = (
    PRODUCTION_INITIALIZATION_COMPLETE.find(
        "import hashlib"
    )
)

configuration_position = (
    PRODUCTION_INITIALIZATION_COMPLETE.find(
        "JOB_SOURCE_TABLE ="
    )
)

schema_position = (
    PRODUCTION_INITIALIZATION_COMPLETE.find(
        "registry_schema ="
    )
)

pipeline_position = (
    PRODUCTION_INITIALIZATION_COMPLETE.find(
        "job_preprocessing_pipeline ="
    )
)

model_position = (
    PRODUCTION_INITIALIZATION_COMPLETE.find(
        "job_registered_model ="
    )
)

positions = [
    import_position,
    configuration_position,
    schema_position,
    pipeline_position,
    model_position
]

assert all(position >= 0 for position in positions)

assert positions == sorted(positions), (
    "Production initialization components "
    "are not in the expected order."
)

# --------------------------------------------------
# 4. Confirm registry schema source
# --------------------------------------------------

assert (
    "StructType.fromJson"
    in PRODUCTION_INITIALIZATION_COMPLETE
)

assert (
    "registry_schema ="
    in PRODUCTION_INITIALIZATION_COMPLETE
)

# --------------------------------------------------
# 5. Confirm model initialization
# --------------------------------------------------

assert (
    "PipelineModel.load"
    in PRODUCTION_INITIALIZATION_COMPLETE
)

assert (
    "mlflow.spark.load_model"
    in PRODUCTION_INITIALIZATION_COMPLETE
)

# --------------------------------------------------
# 6. Summary
# --------------------------------------------------

print("\nINITIALIZATION COMPONENTS")

print("Python and Spark imports: INCLUDED")
print("Production configuration: INCLUDED")
print("Registry schema: INCLUDED")
print("Preprocessing pipeline loading: INCLUDED")
print("Registered model loading: INCLUDED")

print("\nVALIDATION SUMMARY")

print(
    "Initialization source characters:",
    len(PRODUCTION_INITIALIZATION_COMPLETE)
)

print("Python syntax: PASS")
print("Component ordering: PASS")
print("Registry schema reconstruction: PASS")
print("Model initialization source: PASS")
print("Initialization executed: False")
print("Workspace modified: False")

print(
    "\nCOMPLETE PRODUCTION INITIALIZATION "
    "ASSEMBLY: PASS"
)


# COMMAND ----------


# Step 15.70: Assemble complete production notebook source
# Static validation only — no execution or workspace writes.

import ast

print("=" * 65)
print("COMPLETE PRODUCTION NOTEBOOK SOURCE ASSEMBLY")
print("=" * 65)

# --------------------------------------------------
# 1. Assemble initialization and production functions
# --------------------------------------------------

PRODUCTION_NOTEBOOK_SOURCE = (
    "# Databricks notebook source\n"
    "# Telco Churn - Snapshot-Protected Batch Inference\n"
    "# Production notebook - initialization and functions\n\n"
    + PRODUCTION_INITIALIZATION_COMPLETE
    + "\n\n\n"
    + PRODUCTION_MODULE_WITH_STRICT_PUBLISHER
    + "\n"
)

# --------------------------------------------------
# 2. Parse the assembled source without executing it
# --------------------------------------------------

production_tree = ast.parse(
    PRODUCTION_NOTEBOOK_SOURCE
)

# --------------------------------------------------
# 3. Inspect function definitions
# --------------------------------------------------

production_functions = [
    node.name
    for node in production_tree.body
    if isinstance(
        node,
        (ast.FunctionDef, ast.AsyncFunctionDef)
    )
]

expected_functions = (
    PRODUCTION_FUNCTION_ORDER
    + ["publish_quality_approved_batch_strict"]
)

assert production_functions == expected_functions, (
    "Production function inventory does not match."
)

# --------------------------------------------------
# 4. Inspect top-level statements
# --------------------------------------------------

statement_counts = {}

for node in production_tree.body:
    statement_type = type(node).__name__

    statement_counts[statement_type] = (
        statement_counts.get(statement_type, 0) + 1
    )

print("\nTOP-LEVEL STATEMENT TYPES")

for statement_type, count in sorted(
    statement_counts.items()
):
    print(f"{statement_type}: {count}")

# --------------------------------------------------
# 5. Verify strict publisher activation
# --------------------------------------------------

activation_found = False

for node in production_tree.body:

    if not isinstance(node, ast.Assign):
        continue

    for target in node.targets:

        if (
            isinstance(target, ast.Name)
            and target.id == "publish_quality_approved_batch"
            and isinstance(node.value, ast.Name)
            and node.value.id
            == "publish_quality_approved_batch_strict"
        ):
            activation_found = True

assert activation_found, (
    "Strict publisher activation is missing."
)

# --------------------------------------------------
# 6. Check for direct execution of the batch job
# --------------------------------------------------

batch_entry_calls = []

for node in production_tree.body:

    if not isinstance(node, ast.Expr):
        continue

    if not isinstance(node.value, ast.Call):
        continue

    if isinstance(node.value.func, ast.Name):

        if node.value.func.id in {
            "execute_snapshot_protected_batch_job",
            "publish_quality_approved_batch",
            "publish_quality_approved_batch_strict"
        }:
            batch_entry_calls.append(
                node.value.func.id
            )

assert not batch_entry_calls, (
    "Unexpected direct batch execution detected."
)

# --------------------------------------------------
# 7. Summary
# --------------------------------------------------

print("\nPRODUCTION NOTEBOOK VALIDATION")

print(
    "Notebook source characters:",
    len(PRODUCTION_NOTEBOOK_SOURCE)
)

print(
    "Production functions:",
    len(production_functions)
)

print(
    "Strict publisher activated:",
    activation_found
)

print(
    "Direct batch execution calls:",
    len(batch_entry_calls)
)

print("Python syntax: PASS")
print("Notebook saved to workspace: False")
print("Production inference executed: False")
print("Delta tables modified: False")

print(
    "\nCOMPLETE PRODUCTION NOTEBOOK "
    "SOURCE ASSEMBLY: PASS"
)


# COMMAND ----------


# Step 15.71: Production dependency and execution-safety audit
# Static inspection only — no production code execution.

import ast
import builtins
import symtable

print("=" * 65)
print("PRODUCTION DEPENDENCY AND EXECUTION-SAFETY AUDIT")
print("=" * 65)

source = PRODUCTION_NOTEBOOK_SOURCE
tree = ast.parse(source)

# --------------------------------------------------
# 1. Identify module-level names and unresolved globals
# --------------------------------------------------

symbol_table = symtable.symtable(
    source,
    "<production_notebook>",
    "exec"
)

module_names = {
    symbol.get_name()
    for symbol in symbol_table.get_symbols()
    if symbol.is_assigned() or symbol.is_imported()
}

builtin_names = set(dir(builtins))

# Databricks provides spark at runtime.
runtime_provided_names = {"spark"}

unresolved_globals = set()

def inspect_symbol_table(table):
    for child in table.get_children():

        for symbol in child.get_symbols():

            if not symbol.is_global():
                continue

            if not symbol.is_referenced():
                continue

            name = symbol.get_name()

            if (
                name not in module_names
                and name not in builtin_names
                and name not in runtime_provided_names
            ):
                unresolved_globals.add(name)

        inspect_symbol_table(child)

inspect_symbol_table(symbol_table)

print("\nGLOBAL DEPENDENCY AUDIT")

print("Module-defined/imported names:", len(module_names))
print("Runtime-provided names:", sorted(runtime_provided_names))
print("Potential unresolved globals:", len(unresolved_globals))

for name in sorted(unresolved_globals):
    print("REVIEW:", name)

# --------------------------------------------------
# 2. Inspect top-level function calls
# --------------------------------------------------

top_level_calls = []

for node in tree.body:

    # Only inspect calls outside function definitions.
    if isinstance(
        node,
        (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        continue

    for descendant in ast.walk(node):

        if isinstance(descendant, ast.Call):

            top_level_calls.append({
                "line": descendant.lineno,
                "expression": ast.unparse(
                    descendant.func
                )
            })

print("\nTOP-LEVEL CALL INVENTORY")

for call in top_level_calls:
    print(
        f"Line {call['line']}: "
        f"{call['expression']}"
    )

# --------------------------------------------------
# 3. Identify potentially mutating operations
# --------------------------------------------------

mutation_indicators = {
    "saveAsTable",
    "insertInto",
    "merge",
    "execute",
    "delete",
    "update",
    "create",
    "createOrReplaceTempView",
    "execute_snapshot_protected_batch_job"
}

potential_mutations = [
    call
    for call in top_level_calls
    if call["expression"].split(".")[-1]
    in mutation_indicators
]

print("\nEXECUTION-SAFETY AUDIT")

print(
    "Top-level calls found:",
    len(top_level_calls)
)

print(
    "Potential mutation calls:",
    len(potential_mutations)
)

for call in potential_mutations:
    print(
        "REVIEW:",
        call["line"],
        call["expression"]
    )

# --------------------------------------------------
# 4. Final audit summary
# --------------------------------------------------

print("\nAUDIT SUMMARY")

print("Python syntax: PASS")
print("Potential unresolved globals:", len(unresolved_globals))
print("Potential mutation calls:", len(potential_mutations))
print("Production source executed: False")
print("Workspace modified: False")

if unresolved_globals or potential_mutations:
    print(
        "\nPRODUCTION AUDIT: REVIEW REQUIRED"
    )
else:
    print(
        "\nPRODUCTION AUDIT: STATIC CHECKS PASS"
    )


# COMMAND ----------


# Step 15.72: Compile production notebook source
# No execution, model loading, or Delta table writes.

import ast
import types

print("=" * 65)
print("PRODUCTION NOTEBOOK COMPILATION VALIDATION")
print("=" * 65)

# --------------------------------------------------
# 1. Compile production source without executing it
# --------------------------------------------------

production_code_object = compile(
    PRODUCTION_NOTEBOOK_SOURCE,
    "<telco_production_notebook>",
    "exec"
)

assert isinstance(
    production_code_object,
    types.CodeType
)

print("\nCOMPILATION VALIDATION")
print("Python compilation: PASS")

# --------------------------------------------------
# 2. Inspect compiled function names
# --------------------------------------------------

compiled_function_names = [
    constant.co_name
    for constant in production_code_object.co_consts
    if isinstance(constant, types.CodeType)
]

expected_function_names = (
    PRODUCTION_FUNCTION_ORDER
    + ["publish_quality_approved_batch_strict"]
)

missing_functions = [
    name
    for name in expected_function_names
    if name not in compiled_function_names
]

assert not missing_functions, (
    f"Missing compiled functions: {missing_functions}"
)

print("\nCOMPILED FUNCTION INVENTORY")

for name in expected_function_names:
    print("PASS:", name)

# --------------------------------------------------
# 3. Verify strict publisher activation
# --------------------------------------------------

production_tree = ast.parse(
    PRODUCTION_NOTEBOOK_SOURCE
)

strict_activation_count = 0

for node in production_tree.body:

    if not isinstance(node, ast.Assign):
        continue

    if not (
        isinstance(node.value, ast.Name)
        and node.value.id
        == "publish_quality_approved_batch_strict"
    ):
        continue

    for target in node.targets:

        if (
            isinstance(target, ast.Name)
            and target.id
            == "publish_quality_approved_batch"
        ):
            strict_activation_count += 1

assert strict_activation_count == 1, (
    "Expected exactly one strict publisher activation."
)

# --------------------------------------------------
# 4. Final summary
# --------------------------------------------------

print("\nCOMPILATION SUMMARY")

print(
    "Compiled functions:",
    len(expected_function_names)
)

print(
    "Missing functions:",
    len(missing_functions)
)

print(
    "Strict publisher activations:",
    strict_activation_count
)

print(
    "Production source characters:",
    len(PRODUCTION_NOTEBOOK_SOURCE)
)

print("Compilation successful: True")
print("Production source executed: False")
print("Models loaded by this cell: False")
print("Delta tables modified: False")
print("Workspace modified: False")

print(
    "\nPRODUCTION NOTEBOOK COMPILATION: PASS"
)


# COMMAND ----------


# Step 15.79B: Inspect unexpected production notebook content
# Read-only — no notebook modifications or inference

import ast
import base64

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

w = WorkspaceClient()

production_path = (
    "/Telco_Customer_Churn_ML/"
    "15A_Telco_Snapshot_Protected_Batch_Job"
)

exported = w.workspace.export(
    path=production_path,
    format=ExportFormat.SOURCE
)

source = base64.b64decode(
    exported.content
).decode("utf-8")

tree = ast.parse(source)

# Find calls to the batch execution function
execution_calls = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "execute_snapshot_protected_batch_job"
]

print("=" * 65)
print("PRODUCTION NOTEBOOK CONTENT INSPECTION")
print("=" * 65)

print("\nNOTEBOOK CONTENT")
print("Source characters:", len(source))
print(
    "Contains Step 15.79 management cell:",
    "# Step 15.79: Update the separate production notebook"
    in source
)
print(
    "Contains controlled execution entry point:",
    'JOB_BATCH_ID = "d972dd16-029d-489f-a406-3c6c6e1e6265"'
    in source
)
print("Batch execution calls:", len(execution_calls))

print("\nSAFETY")
print("Notebook modified by this cell: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nCONTENT INSPECTION: COMPLETE")


# COMMAND ----------


# Step 15.79C: Read-only execution safety check
# Run in the original development notebook

from pyspark.sql import functions as F

print("=" * 65)
print("POST-RUN EXECUTION SAFETY CHECK")
print("=" * 65)

batch_id = "d972dd16-029d-489f-a406-3c6c6e1e6265"

tables = {
    "Batch registry":
        "workspace.default.telco_15_job_batch_registry",
    "Development predictions":
        "workspace.default.telco_15_job_dev_predictions",
    "Approved predictions":
        "workspace.default.telco_15_approved_predictions"
}

for label, table_name in tables.items():

    print(f"\n{label}")
    print("Table:", table_name)

    if not spark.catalog.tableExists(table_name):
        print("Table exists: False")
        continue

    df = spark.table(table_name)

    print("Table exists: True")
    print("Total rows:", df.count())

    if "batch_id" in df.columns:
        matching_rows = df.filter(
            F.col("batch_id") == batch_id
        ).count()

        print("Controlled test batch rows:", matching_rows)
    else:
        print("Batch ID column: Not present")

print("\nSAFETY CHECK COMPLETE")
print("Tables modified by this inspection: False")


# COMMAND ----------


# Step 15.79D: Inspect backup before recovery
# Read-only — no workspace or Delta modifications

import ast
import base64
import hashlib

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

w = WorkspaceClient()

backup_path = (
    "/Telco_Customer_Churn_ML/"
    "15A_Telco_Snapshot_Protected_Batch_Job_Backup_PreEntry"
)

result = w.workspace.export(
    path=backup_path,
    format=ExportFormat.SOURCE
)

backup_source = base64.b64decode(
    result.content
).decode("utf-8")

tree = ast.parse(backup_source)

functions = [
    node.name
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
]

markers = {
    "Backup management cell":
        "# Step 15.78: Back up production notebook",
    "Notebook update management cell":
        "# Step 15.79: Update the separate production notebook",
    "Controlled execution entry point":
        'JOB_BATCH_ID = "d972dd16-029d-489f-a406-3c6c6e1e6265"',
    "MLflow temporary directory":
        "MLFLOW_DFS_TMP"
}

print("=" * 65)
print("PRODUCTION BACKUP RECOVERY INSPECTION")
print("=" * 65)

print("\nBACKUP SOURCE")
print("Characters:", len(backup_source))
print("Functions:", len(functions))
print(
    "SHA-256:",
    hashlib.sha256(
        backup_source.encode("utf-8")
    ).hexdigest()
)

print("\nCONTENT MARKERS")

for label, marker in markers.items():
    print(f"{label}: {marker in backup_source}")

print("\nSAFETY")
print("Backup modified: False")
print("Production notebook modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nBACKUP RECOVERY INSPECTION: COMPLETE")


# COMMAND ----------


# Step 15.79E: Identify production/deployment code boundary
# Read-only — no notebook modifications or Delta writes

import base64
import re

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

print("=" * 65)
print("PRODUCTION NOTEBOOK SOURCE BOUNDARY INSPECTION")
print("=" * 65)

w = WorkspaceClient()

path = (
    "/Telco_Customer_Churn_ML/"
    "15A_Telco_Snapshot_Protected_Batch_Job"
)

result = w.workspace.export(
    path=path,
    format=ExportFormat.SOURCE
)

source = base64.b64decode(
    result.content
).decode("utf-8")

sections = re.split(
    r"(?m)^# COMMAND ----------\s*$",
    source
)

print("\nNOTEBOOK INVENTORY")
print("Source characters:", len(source))
print("Notebook sections:", len(sections))

print("\nSECTION HEADINGS")

for index, section in enumerate(sections):

    meaningful_lines = [
        line.strip()
        for line in section.splitlines()
        if line.strip()
    ]

    preview = " | ".join(meaningful_lines[:3])

    print(
        f"Section {index:02d}: "
        f"{preview[:140]}"
    )

print("\nSAFETY")
print("Workspace modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nSOURCE BOUNDARY INSPECTION: COMPLETE")


# COMMAND ----------


# Step 15.79F: Validate original production section
# Read-only — no workspace modifications or inference

import ast
import base64
import hashlib
import re

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

w = WorkspaceClient()

production_path = (
    "/Telco_Customer_Churn_ML/"
    "15A_Telco_Snapshot_Protected_Batch_Job"
)

exported = w.workspace.export(
    path=production_path,
    format=ExportFormat.SOURCE
)

source = base64.b64decode(
    exported.content
).decode("utf-8")

sections = re.split(
    r"(?m)^# COMMAND ----------\s*$",
    source
)

original_section = sections[0]

print("=" * 65)
print("ORIGINAL PRODUCTION SECTION VALIDATION")
print("=" * 65)

print("\nSOURCE INFORMATION")
print("Characters:", len(original_section))
print(
    "SHA-256:",
    hashlib.sha256(
        original_section.encode("utf-8")
    ).hexdigest()
)

try:
    tree = ast.parse(original_section)
    print("Python syntax: PASS")

    functions = [
        node.name
        for node in tree.body
        if isinstance(node, (
            ast.FunctionDef,
            ast.AsyncFunctionDef
        ))
    ]

    print("\nPRODUCTION FUNCTIONS")
    print("Function count:", len(functions))

    for name in functions:
        print("-", name)

except SyntaxError as error:
    print("Python syntax: FAIL")
    print("Error:", error)

print("\nREQUIRED COMPONENTS")
print(
    "MLflow temporary directory:",
    "MLFLOW_DFS_TMP" in original_section
)
print(
    "Batch execution function:",
    "execute_snapshot_protected_batch_job"
    in original_section
)
print(
    "Strict publisher:",
    "strict" in original_section.lower()
)

print("\nSAFETY")
print("Workspace modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nSECTION VALIDATION: COMPLETE")


# COMMAND ----------


# Step 15.79G: Prepare clean production notebook base
# No inference, workspace imports, or Delta writes

import ast
import base64
import hashlib
import re

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

print("=" * 65)
print("CLEAN PRODUCTION NOTEBOOK PREPARATION")
print("=" * 65)

w = WorkspaceClient()

production_path = (
    "/Telco_Customer_Churn_ML/"
    "15A_Telco_Snapshot_Protected_Batch_Job"
)

exported = w.workspace.export(
    path=production_path,
    format=ExportFormat.SOURCE
)

current_source = base64.b64decode(
    exported.content
).decode("utf-8")

sections = re.split(
    r"(?m)^# COMMAND ----------\s*$",
    current_source
)

# Preserve original production implementation only
CLEAN_PRODUCTION_BASE = sections[0].rstrip() + "\n"

expected_base_hash = (
    "13e41068883e35d56c7face78cdd0f0804e0497dc4f470706cc7a908db62eaf1"
)

actual_base_hash = hashlib.sha256(
    sections[0].encode("utf-8")
).hexdigest()

assert actual_base_hash == expected_base_hash, (
    "Original production section changed. Stopped."
)

tree = ast.parse(CLEAN_PRODUCTION_BASE)

function_names = [
    node.name
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
]

assert len(function_names) == 10, (
    "Expected ten production functions."
)

required_functions = {
    "execute_snapshot_protected_batch_job",
    "publish_quality_approved_batch_strict",
    "validate_strict_publication_integrity"
}

assert required_functions.issubset(
    set(function_names)
), "Required production functions missing."

for forbidden_marker in [
    "# Step 15.78: Back up production notebook",
    "# Step 15.79: Update the separate production notebook",
    "# Step 15.79A: Diagnose production notebook"
]:
    assert forbidden_marker not in CLEAN_PRODUCTION_BASE, (
        f"Unexpected management code: {forbidden_marker}"
    )

print("\nCLEAN BASE VALIDATION")
print("Source characters:", len(CLEAN_PRODUCTION_BASE))
print("Python syntax: PASS")
print("Production functions:", len(function_names))
print("Required functions: PASS")
print("Deployment-management markers: ABSENT")
print("Source fingerprint: PASS")

print("\nSAFETY")
print("Production notebook overwritten: False")
print("Backup modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nCLEAN PRODUCTION BASE: PASS")


# COMMAND ----------


# Step 15.79H: Validate controlled execution entry point
# Source preparation only — no inference or Delta writes

import ast
import hashlib

print("=" * 65)
print("CONTROLLED PRODUCTION ENTRY-POINT VALIDATION")
print("=" * 65)

CONTROLLED_ENTRY_POINT = '''
# COMMAND ----------

# Controlled snapshot-protected batch execution

JOB_SOURCE_TABLE = "workspace.default.telco_test_raw"
JOB_SOURCE_VERSION = 2

JOB_MODEL_NAME = (
    "workspace.default.telco_churn_logistic_regression"
)
JOB_MODEL_VERSION = 1

JOB_THRESHOLD = 0.50

JOB_DEV_OUTPUT_TABLE = (
    "workspace.default.telco_15_job_dev_predictions"
)

JOB_BATCH_REGISTRY_TABLE = (
    "workspace.default.telco_15_job_batch_registry"
)

JOB_BATCH_ID = "d972dd16-029d-489f-a406-3c6c6e1e6265"

assert JOB_DEV_OUTPUT_TABLE == (
    "workspace.default.telco_15_job_dev_predictions"
)

assert JOB_DEV_OUTPUT_TABLE != (
    "workspace.default.telco_15_approved_predictions"
)

job_execution_results = execute_snapshot_protected_batch_job(
    source_table=JOB_SOURCE_TABLE,
    source_version=JOB_SOURCE_VERSION,
    preprocessing_pipeline=job_preprocessing_pipeline,
    registered_model=job_registered_model,
    model_name=JOB_MODEL_NAME,
    model_version=JOB_MODEL_VERSION,
    threshold=JOB_THRESHOLD,
    batch_id=JOB_BATCH_ID,
    destination_table=JOB_DEV_OUTPUT_TABLE,
    registry_table=JOB_BATCH_REGISTRY_TABLE
)

assert job_execution_results["execution_status"] == "PASS"

for key, value in job_execution_results.items():
    print(f"{key}: {value}")
'''

entry_tree = ast.parse(CONTROLLED_ENTRY_POINT)

execution_calls = [
    node
    for node in ast.walk(entry_tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "execute_snapshot_protected_batch_job"
]

assert len(execution_calls) == 1, (
    "Expected exactly one batch execution call."
)

call = execution_calls[0]
arguments = {
    keyword.arg: keyword.value
    for keyword in call.keywords
}

assert isinstance(
    arguments["destination_table"],
    ast.Name
)

assert arguments["destination_table"].id == (
    "JOB_DEV_OUTPUT_TABLE"
)

assert "CLEAN_PRODUCTION_BASE" in globals(), (
    "Clean production base is missing."
)

combined_tree = ast.parse(
    CLEAN_PRODUCTION_BASE + "\n" + CONTROLLED_ENTRY_POINT
)

print("\nENTRY-POINT VALIDATION")
print("Python syntax: PASS")
print("Source snapshot: telco_test_raw")
print("Source version: 2")
print("Model version: 1")
print("Threshold: 0.50")
print("Destination: Development predictions")
print("Execution calls:", len(execution_calls))
print("Combined Python syntax: PASS")

print("\nSOURCE FINGERPRINT")
print(
    hashlib.sha256(
        CONTROLLED_ENTRY_POINT.encode("utf-8")
    ).hexdigest()
)

print("\nSAFETY")
print("Production notebook overwritten: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nCONTROLLED ENTRY POINT: PASS")


# COMMAND ----------


# Step 15.79I: Assemble and inspect complete production source
# Read-only — no inference, notebook updates, or Delta writes

import ast
import hashlib

print("=" * 65)
print("COMPLETE PRODUCTION NOTEBOOK ASSEMBLY")
print("=" * 65)

assert "CLEAN_PRODUCTION_BASE" in globals(), (
    "Clean production base is missing."
)

assert "CONTROLLED_ENTRY_POINT" in globals(), (
    "Controlled entry point is missing."
)

COMPLETE_CLEAN_PRODUCTION_SOURCE = (
    CLEAN_PRODUCTION_BASE.rstrip()
    + "\n\n"
    + CONTROLLED_ENTRY_POINT.lstrip()
)

tree = ast.parse(COMPLETE_CLEAN_PRODUCTION_SOURCE)

function_names = [
    node.name
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
]

execution_calls = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "execute_snapshot_protected_batch_job"
]

assert len(function_names) == 10
assert len(execution_calls) == 1

# Inspect top-level assignments in the clean base
base_tree = ast.parse(CLEAN_PRODUCTION_BASE)

assigned_names = set()

for node in base_tree.body:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )

        for target in targets:
            if isinstance(target, ast.Name):
                assigned_names.add(target.id)

required_variables = [
    "job_preprocessing_pipeline",
    "job_registered_model"
]

print("\nSOURCE VALIDATION")
print("Python syntax: PASS")
print("Production functions:", len(function_names))
print("Batch execution calls:", len(execution_calls))
print(
    "Complete source characters:",
    len(COMPLETE_CLEAN_PRODUCTION_SOURCE)
)

print("\nMODEL INITIALIZATION DEPENDENCIES")

for variable in required_variables:
    print(
        f"{variable}:",
        "Top-level assignment found"
        if variable in assigned_names
        else "NO TOP-LEVEL ASSIGNMENT FOUND"
    )

print("\nSOURCE FINGERPRINT")
print(
    hashlib.sha256(
        COMPLETE_CLEAN_PRODUCTION_SOURCE.encode("utf-8")
    ).hexdigest()
)

print("\nSAFETY")
print("Production notebook overwritten: False")
print("Backup modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nCOMPLETE SOURCE INSPECTION: FINISHED")


# COMMAND ----------


# Step 15.79J: Inspect production initialization code
# Read-only — no inference or workspace changes

import ast

print("=" * 65)
print("PRODUCTION INITIALIZATION INSPECTION")
print("=" * 65)

assert "CLEAN_PRODUCTION_BASE" in globals(), (
    "Clean production base is missing."
)

tree = ast.parse(CLEAN_PRODUCTION_BASE)

required_variables = {
    "job_preprocessing_pipeline",
    "job_registered_model"
}

found = set()

for node in tree.body:

    if isinstance(node, ast.Assign):
        targets = node.targets

    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]

    else:
        continue

    names = {
        target.id
        for target in targets
        if isinstance(target, ast.Name)
    }

    matching = names.intersection(required_variables)

    if matching:
        for name in sorted(matching):
            found.add(name)

            print(f"\nVARIABLE: {name}")
            print("-" * 50)

            print(
                ast.get_source_segment(
                    CLEAN_PRODUCTION_BASE,
                    node
                )
            )

print("\nVALIDATION")
print("Required assignments found:", len(found))
print("Missing:", sorted(required_variables - found))

print("\nSAFETY")
print("Notebook modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nINITIALIZATION INSPECTION: COMPLETE")


# COMMAND ----------


# Step 15.79K: Inspect model and pipeline configuration
# Read-only — does not load models or run inference

import ast

print("=" * 65)
print("PRODUCTION MODEL CONFIGURATION INSPECTION")
print("=" * 65)

assert "CLEAN_PRODUCTION_BASE" in globals(), (
    "Clean production source is missing."
)

tree = ast.parse(CLEAN_PRODUCTION_BASE)

required_names = {
    "JOB_PIPELINE_PATH",
    "JOB_MODEL_URI"
}

print("\nCONFIGURATION ASSIGNMENTS")

found = set()

for node in tree.body:

    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        continue

    for target in targets:
        if (
            isinstance(target, ast.Name)
            and target.id in required_names
        ):
            found.add(target.id)

            print(f"\n{target.id}:")
            print(
                ast.get_source_segment(
                    CLEAN_PRODUCTION_BASE,
                    node
                )
            )

print("\nREQUIRED IMPORTS")

imports = [
    node
    for node in tree.body
    if isinstance(node, (ast.Import, ast.ImportFrom))
]

import_source = "\n".join(
    ast.get_source_segment(CLEAN_PRODUCTION_BASE, node)
    for node in imports
)

print(
    "PipelineModel imported:",
    "PipelineModel" in import_source
)

print(
    "MLflow imported:",
    "mlflow" in import_source
)

print("\nVALIDATION")
print("Configuration variables found:", len(found))
print("Missing:", sorted(required_names - found))

print("\nSAFETY")
print("Production notebook modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nCONFIGURATION INSPECTION: COMPLETE")


# COMMAND ----------


# Step 15.79L: Model identity and publication safety
# Read-only source inspection

import ast

print("=" * 65)
print("MODEL IDENTITY AND PUBLICATION SAFETY")
print("=" * 65)

assert "CLEAN_PRODUCTION_BASE" in globals()

tree = ast.parse(CLEAN_PRODUCTION_BASE)

required_config = {
    "JOB_MODEL_NAME",
    "JOB_MODEL_VERSION",
    "JOB_DEV_OUTPUT_TABLE"
}

print("\nCONFIGURATION")

found = set()

for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id in required_config
            ):
                found.add(target.id)
                print(
                    ast.get_source_segment(
                        CLEAN_PRODUCTION_BASE, node
                    )
                )

print("\nSTRICT PUBLISHER ACTIVATION")

strict_activations = []

for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "publish_quality_approved_batch"
            ):
                strict_activations.append(
                    ast.get_source_segment(
                        CLEAN_PRODUCTION_BASE, node
                    )
                )

for activation in strict_activations:
    print(activation)

print("Activation count:", len(strict_activations))

print("\nTOP-LEVEL EXECUTION CALLS")

execution_names = {
    "execute_snapshot_protected_batch_job",
    "run_batch_inference",
    "publish_quality_approved_batch",
    "publish_quality_approved_batch_strict"
}

calls = []

for node in tree.body:
    if isinstance(node, ast.Expr) and isinstance(
        node.value, ast.Call
    ):
        func = node.value.func
        if (
            isinstance(func, ast.Name)
            and func.id in execution_names
        ):
            calls.append(func.id)

print("Direct execution calls:", calls)

print("\nVALIDATION")
print("Configuration variables found:", sorted(found))
print("Missing:", sorted(required_config - found))

print("\nSAFETY")
print("Production notebook modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nMODEL AND PUBLICATION INSPECTION: COMPLETE")


# COMMAND ----------


# Step 15.79M: Audit top-level operations
# Static inspection only — no inference or Delta writes

import ast

print("=" * 65)
print("PRODUCTION SOURCE TOP-LEVEL SAFETY AUDIT")
print("=" * 65)

assert "CLEAN_PRODUCTION_BASE" in globals()

tree = ast.parse(CLEAN_PRODUCTION_BASE)

def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return call_name(node.value) + "." + node.attr
    return "<dynamic call>"

print("\nTOP-LEVEL CALLS")

calls = []

for node in tree.body:
    if isinstance(node, (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Import,
        ast.ImportFrom
    )):
        continue

    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            calls.append(
                (
                    node.lineno,
                    call_name(child.func)
                )
            )

for line_number, name in calls:
    print(f"Line {line_number}: {name}")

print("\nAUDIT SUMMARY")
print("Top-level calls detected:", len(calls))

print("\nSAFETY")
print("Production notebook modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nTOP-LEVEL SAFETY AUDIT: COMPLETE")


# COMMAND ----------


# Step 15.79N: Create clean recovery candidate
# Creates a new notebook only — does NOT execute it

import ast
import base64
import hashlib

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.workspace import (
    ExportFormat,
    ImportFormat,
    Language
)

print("=" * 65)
print("PRODUCTION RECOVERY CANDIDATE CREATION")
print("=" * 65)

assert "COMPLETE_CLEAN_PRODUCTION_SOURCE" in globals(), (
    "Complete clean production source is missing."
)

candidate_source = COMPLETE_CLEAN_PRODUCTION_SOURCE

expected_hash = (
    "722f7d0b8840d9eacaf8a0335cd604d8ce6c0c9c649aa2eb84cf0046a96356ca"
)

actual_hash = hashlib.sha256(
    candidate_source.encode("utf-8")
).hexdigest()

assert actual_hash == expected_hash, (
    "Candidate source fingerprint mismatch. Stopped."
)

ast.parse(candidate_source)

w = WorkspaceClient()

candidate_path = (
    "/Telco_Customer_Churn_ML/"
    "15B_Telco_Snapshot_Protected_Batch_Job_Candidate"
)

# Never overwrite an existing candidate
try:
    existing = w.workspace.get_status(candidate_path)
except NotFound:
    existing = None

assert existing is None, (
    "Candidate notebook already exists. "
    "Stopped to prevent overwriting."
)

# Create candidate notebook
w.workspace.import_(
    path=candidate_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(
        candidate_source.encode("utf-8")
    ).decode("ascii"),
    overwrite=False
)

# Verify exported candidate content
candidate_export = w.workspace.export(
    path=candidate_path,
    format=ExportFormat.SOURCE
)

candidate_bytes = base64.b64decode(
    candidate_export.content
)

assert candidate_bytes == candidate_source.encode("utf-8"), (
    "Candidate content verification failed."
)

print("\nCANDIDATE VALIDATION")
print("Candidate path:", candidate_path)
print("Source fingerprint: PASS")
print("Python syntax: PASS")
print("Exported content matches: True")
print("Candidate created: True")

print("\nSAFETY")
print("Original production notebook overwritten: False")
print("Existing backup modified: False")
print("Batch inference executed: False")
print("Delta tables modified: False")

print("\nRECOVERY CANDIDATE CREATION: PASS")


# COMMAND ----------


# Step 15.79O: Independently verify candidate notebook
# Read-only — does not execute candidate or modify tables

import ast
import base64
import hashlib

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

print("=" * 65)
print("RECOVERY CANDIDATE INDEPENDENT VERIFICATION")
print("=" * 65)

w = WorkspaceClient()

candidate_path = (
    "/Telco_Customer_Churn_ML/"
    "15B_Telco_Snapshot_Protected_Batch_Job_Candidate"
)

exported = w.workspace.export(
    path=candidate_path,
    format=ExportFormat.SOURCE
)

candidate_source = base64.b64decode(
    exported.content
).decode("utf-8")

expected_hash = (
    "722f7d0b8840d9eacaf8a0335cd604d8ce6c0c9c649aa2eb84cf0046a96356ca"
)

actual_hash = hashlib.sha256(
    candidate_source.encode("utf-8")
).hexdigest()

assert actual_hash == expected_hash, (
    "Candidate fingerprint mismatch."
)

tree = ast.parse(candidate_source)

functions = [
    node.name
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
]

execution_calls = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "execute_snapshot_protected_batch_job"
]

assert len(functions) == 10
assert len(execution_calls) == 1

for marker in [
    "# Step 15.78: Back up production notebook",
    "# Step 15.79: Update the separate production notebook",
    "# Step 15.79A: Diagnose production notebook"
]:
    assert marker not in candidate_source

expected_destination = (
    "workspace.default.telco_15_job_dev_predictions"
)

assert expected_destination in candidate_source

print("\nCANDIDATE VALIDATION")
print("Source fingerprint: PASS")
print("Python syntax: PASS")
print("Production functions:", len(functions))
print("Batch execution calls:", len(execution_calls))
print("Deployment-management cells: ABSENT")
print("Development destination: PRESENT")

print("\nSAFETY")
print("Candidate executed: False")
print("Original production notebook modified: False")
print("Backup modified: False")
print("Delta tables modified: False")

print("\nINDEPENDENT CANDIDATE VERIFICATION: PASS")


# COMMAND ----------


# Step 15.79P: Corrected execution destination audit
# Read-only — no inference or workspace modifications

import ast
import base64

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

print("=" * 65)
print("CANDIDATE EXECUTION DESTINATION AUDIT")
print("=" * 65)

candidate_path = (
    "/Telco_Customer_Churn_ML/"
    "15B_Telco_Snapshot_Protected_Batch_Job_Candidate"
)

w = WorkspaceClient()

exported = w.workspace.export(
    path=candidate_path,
    format=ExportFormat.SOURCE
)

source = base64.b64decode(
    exported.content
).decode("utf-8")

tree = ast.parse(source)

# Find the top-level batch execution assignment
calls = [
    node.value
    for node in tree.body
    if isinstance(node, ast.Assign)
    and isinstance(node.value, ast.Call)
    and isinstance(node.value.func, ast.Name)
    and node.value.func.id == "execute_snapshot_protected_batch_job"
]

assert len(calls) == 1, (
    "Expected exactly one top-level execution call."
)

# calls[0] is now an ast.Call object
call = calls[0]

arguments = {
    keyword.arg: keyword.value
    for keyword in call.keywords
}

destination = arguments["destination_table"]

assert isinstance(destination, ast.Name)
assert destination.id == "JOB_DEV_OUTPUT_TABLE"

# Inspect assignments to the destination variable
destination_assignments = []

for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "JOB_DEV_OUTPUT_TABLE"
            ):
                destination_assignments.append(node.value)

assert len(destination_assignments) >= 1

resolved_values = [
    ast.literal_eval(value)
    for value in destination_assignments
]

expected = "workspace.default.telco_15_job_dev_predictions"
protected = "workspace.default.telco_15_approved_predictions"

assert all(value == expected for value in resolved_values)
assert protected not in resolved_values

print("\nEXECUTION CONFIGURATION")
print("Execution calls:", len(calls))
print("Destination argument:", destination.id)
print("Destination assignments:", len(resolved_values))
print("Resolved destination:", resolved_values[-1])

print("\nSAFETY")
print("All destination assignments match development table: True")
print("Approved table used as destination: False")
print("Candidate executed: False")
print("Production notebook modified: False")
print("Delta tables modified: False")

print("\nEXECUTION DESTINATION AUDIT: PASS")


# COMMAND ----------


# Step 15.79Q: Inspect internal publishing safeguards
# Read-only — no inference or Delta writes

import ast
import base64

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

print("=" * 65)
print("INTERNAL PUBLISHING SAFETY AUDIT")
print("=" * 65)

candidate_path = (
    "/Telco_Customer_Churn_ML/"
    "15B_Telco_Snapshot_Protected_Batch_Job_Candidate"
)

w = WorkspaceClient()

exported = w.workspace.export(
    path=candidate_path,
    format=ExportFormat.SOURCE
)

source = base64.b64decode(
    exported.content
).decode("utf-8")

tree = ast.parse(source)

target_functions = {
    "execute_snapshot_protected_batch_job",
    "publish_quality_approved_batch_strict",
    "publish_quality_approved_batch"
}

found = set()

for node in tree.body:

    if not isinstance(node, ast.FunctionDef):
        continue

    if node.name not in target_functions:
        continue

    found.add(node.name)

    print("\n" + "=" * 65)
    print("FUNCTION:", node.name)
    print("=" * 65)

    function_source = ast.get_source_segment(
        source, node
    )

    print(function_source)

print("\nAUDIT SUMMARY")
print("Functions found:", len(found))
print("Missing:", sorted(target_functions - found))

print("\nSAFETY")
print("Candidate executed: False")
print("Production notebook modified: False")
print("Backup modified: False")
print("Delta tables modified: False")

print("\nINTERNAL PUBLISHING INSPECTION: COMPLETE")


# COMMAND ----------


# Step 15.79R: Prepare destination protection patch
# In-memory only — no inference or workspace writes

import ast
import hashlib

print("=" * 65)
print("STRICT DEVELOPMENT DESTINATION PROTECTION")
print("=" * 65)

assert "COMPLETE_CLEAN_PRODUCTION_SOURCE" in globals()

original_source = COMPLETE_CLEAN_PRODUCTION_SOURCE

expected_original_hash = (
    "722f7d0b8840d9eacaf8a0335cd604d8ce6c0c9c649aa2eb84cf0046a96356ca"
)

assert hashlib.sha256(
    original_source.encode("utf-8")
).hexdigest() == expected_original_hash, (
    "Source fingerprint mismatch. Patch stopped."
)

old_validation = '''    if not destination_table.startswith("workspace.default."):
        raise ValueError("Invalid destination table.")'''

new_validation = '''    allowed_destination = (
        "workspace.default.telco_15_job_dev_predictions"
    )

    if destination_table != allowed_destination:
        raise ValueError(
            "Publication blocked: only the development "
            "predictions table is permitted."
        )'''

assert original_source.count(old_validation) == 1, (
    "Expected validation block not found exactly once."
)

PATCHED_PRODUCTION_SOURCE = original_source.replace(
    old_validation,
    new_validation,
    1
)

ast.parse(PATCHED_PRODUCTION_SOURCE)

assert old_validation not in PATCHED_PRODUCTION_SOURCE
assert new_validation in PATCHED_PRODUCTION_SOURCE

print("\nPATCH VALIDATION")
print("Original source unchanged: True")
print("Replacement blocks:", 1)
print("Python syntax: PASS")
print("Development-only destination restriction: PRESENT")

print("\nPATCHED SOURCE SHA-256")
print(
    hashlib.sha256(
        PATCHED_PRODUCTION_SOURCE.encode("utf-8")
    ).hexdigest()
)

print("\nSAFETY")
print("Candidate notebook modified: False")
print("Production notebook modified: False")
print("Backup modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nDESTINATION PROTECTION PATCH: PASS")


# COMMAND ----------


# Step 15.79S: Audit potential Delta write operations
# Static inspection only — no code execution or table writes

import ast

print("=" * 65)
print("PRODUCTION SOURCE DELTA WRITE AUDIT")
print("=" * 65)

assert "PATCHED_PRODUCTION_SOURCE" in globals(), (
    "Patched production source is missing."
)

tree = ast.parse(PATCHED_PRODUCTION_SOURCE)

write_methods = {
    "saveAsTable",
    "save",
    "insertInto",
    "merge",
    "execute",
    "sql",
    "createOrReplaceTempView",
    "writeTo",
    "delete",
    "update",
    "overwrite",
    "append"
}

def get_call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return get_call_name(node.value) + "." + node.attr
    return "<dynamic>"

print("\nPOTENTIAL WRITE OPERATIONS")

operations = []

for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
        continue

    name = get_call_name(node.func)
    method = name.split(".")[-1]

    if method in write_methods:
        operations.append((node.lineno, name))

for line_number, name in sorted(operations):
    print(f"Line {line_number}: {name}")

print("\nAUDIT SUMMARY")
print("Potential write-related calls:", len(operations))

print("\nSAFETY")
print("Notebook 15A modified: False")
print("Notebook 15B modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nDELTA WRITE AUDIT: COMPLETE")


# COMMAND ----------


# Step 15.79T: Inspect Delta write destinations
# Read-only source inspection

import ast

print("=" * 65)
print("DELTA WRITE DESTINATION INSPECTION")
print("=" * 65)

assert "PATCHED_PRODUCTION_SOURCE" in globals()

source = PATCHED_PRODUCTION_SOURCE
tree = ast.parse(source)

write_methods = {"saveAsTable", "merge", "execute"}

# Locate each write operation and its enclosing function
for function in tree.body:

    if not isinstance(function, ast.FunctionDef):
        continue

    matches = []

    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in write_methods
        ):
            matches.append(node)

    if not matches:
        continue

    print("\n" + "=" * 65)
    print("FUNCTION:", function.name)
    print("=" * 65)

    function_lines = source.splitlines()

    for call in sorted(matches, key=lambda n: n.lineno):
        start = max(function.lineno, call.lineno - 12)
        end = min(
            function.end_lineno,
            call.lineno + 8
        )

        print(f"\nWRITE OPERATION: {call.func.attr}")
        print(f"Source lines: {start}-{end}")

        for line_number in range(start, end + 1):
            print(
                f"{line_number:04d}: "
                f"{function_lines[line_number - 1]}"
            )

print("\nSAFETY")
print("Production notebook modified: False")
print("Candidate notebook modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nWRITE DESTINATION INSPECTION: COMPLETE")


# COMMAND ----------


# Step 15.79U: Add write-destination safeguards
# In-memory source modification only
# No workspace updates, inference, or Delta writes

import ast
import hashlib

print("=" * 65)
print("WRITE DESTINATION SAFEGUARD PREPARATION")
print("=" * 65)

assert "PATCHED_PRODUCTION_SOURCE" in globals(), (
    "Step 15.79R source is missing."
)

source = PATCHED_PRODUCTION_SOURCE

expected_hash = (
    "79028c5d5e5ee11cbd065e9096a5f754b37ba875da6e41de004984c26fd267fa"
)

assert hashlib.sha256(
    source.encode("utf-8")
).hexdigest() == expected_hash, (
    "Patched source fingerprint mismatch."
)

tree = ast.parse(source)

required_functions = {
    "register_batch_identity":
        ("registry_table",
         "workspace.default.telco_15_job_batch_registry"),
    "publish_quality_approved_batch":
        ("destination_table",
         "workspace.default.telco_15_job_dev_predictions")
}

# Insert checks at the beginning of each function body.
# Use AST line locations to preserve the existing source.

lines = source.splitlines(keepends=True)
insertions = []

for node in tree.body:

    if not isinstance(node, ast.FunctionDef):
        continue

    if node.name not in required_functions:
        continue

    parameter, allowed_table = required_functions[node.name]

    # Place the guard after the function's docstring.
    first_statement = node.body[0]

    if (
        isinstance(first_statement, ast.Expr)
        and isinstance(first_statement.value, ast.Constant)
        and isinstance(first_statement.value.value, str)
    ):
        insertion_line = first_statement.end_lineno
    else:
        insertion_line = node.body[0].lineno - 1

    guard = (
        f"\n    if {parameter} != {allowed_table!r}:\n"
        f"        raise ValueError(\n"
        f"            'Write blocked: unauthorized table destination.'\n"
        f"        )\n"
    )

    insertions.append((insertion_line, guard, node.name))

assert len(insertions) == 2, (
    "Expected exactly two write guards."
)

# Apply bottom-to-top so line positions remain stable.
for insertion_line, guard, name in sorted(
    insertions,
    reverse=True
):
    lines.insert(insertion_line, guard)

GUARDED_PRODUCTION_SOURCE = "".join(lines)

guarded_tree = ast.parse(GUARDED_PRODUCTION_SOURCE)

print("\nGUARD VALIDATION")

for function_name, (parameter, allowed_table) in (
    required_functions.items()
):

    function = next(
        node for node in guarded_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == function_name
    )

    guards = [
        node for node in function.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
    ]

    matching = [
        node for node in guards
        if isinstance(node.test.left, ast.Name)
        and node.test.left.id == parameter
        and any(
            isinstance(comparator, ast.Constant)
            and comparator.value == allowed_table
            for comparator in node.test.comparators
        )
    ]

    assert len(matching) == 1, (
        f"Missing destination guard: {function_name}"
    )

    print(f"{function_name}: PASS")
    print("  Allowed table:", allowed_table)

print("\nSOURCE VALIDATION")
print("Python syntax: PASS")
print("Write guards added:", len(insertions))
print(
    "Source SHA-256:",
    hashlib.sha256(
        GUARDED_PRODUCTION_SOURCE.encode("utf-8")
    ).hexdigest()
)

print("\nSAFETY")
print("Original source unchanged: True")
print("Notebook 15A modified: False")
print("Notebook 15B modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nWRITE DESTINATION SAFEGUARDS: PASS")


# COMMAND ----------


# Step 15.79V: Final guarded-source structural validation
# Read-only — no inference or workspace modifications

import ast
import hashlib

print("=" * 65)
print("GUARDED PRODUCTION SOURCE VALIDATION")
print("=" * 65)

assert "GUARDED_PRODUCTION_SOURCE" in globals(), (
    "Guarded source is missing."
)

source = GUARDED_PRODUCTION_SOURCE

expected_hash = (
    "f6f64d25801e1156512b78c74b3242581bb79506e095affc108788796ef85e0c"
)

assert hashlib.sha256(
    source.encode("utf-8")
).hexdigest() == expected_hash, (
    "Guarded source fingerprint mismatch."
)

tree = ast.parse(source)

functions = {
    node.name: node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
}

assert len(functions) == 10

required_guards = {
    "register_batch_identity": (
        "registry_table",
        "workspace.default.telco_15_job_batch_registry"
    ),
    "publish_quality_approved_batch": (
        "destination_table",
        "workspace.default.telco_15_job_dev_predictions"
    ),
    "execute_snapshot_protected_batch_job": (
        "destination_table",
        "workspace.default.telco_15_job_dev_predictions"
    )
}

print("\nDESTINATION GUARD VALIDATION")

for function_name, (parameter, expected_table) in (
    required_guards.items()
):

    function = functions[function_name]

    matching_guards = []

    for node in ast.walk(function):

        if not isinstance(node, ast.If):
            continue

        test = node.test

        if not isinstance(test, ast.Compare):
            continue

        if not isinstance(test.left, ast.Name):
            continue

        if test.left.id != parameter:
            continue

        if not any(
            isinstance(value, ast.Constant)
            and value.value == expected_table
            for value in test.comparators
        ):
            continue

        matching_guards.append(node)

    assert len(matching_guards) == 1, (
        f"Expected one destination guard in {function_name}"
    )

    guard = matching_guards[0]

    write_lines = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {
            "saveAsTable", "merge", "execute"
        }
    ]

    if write_lines:
        assert guard.lineno < min(write_lines), (
            f"Guard appears after a write in {function_name}"
        )

    print(f"{function_name}: PASS")
    print("  Guard line:", guard.lineno)
    print("  Write lines:", write_lines)

execution_calls = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "execute_snapshot_protected_batch_job"
]

assert len(execution_calls) == 1

print("\nSOURCE INTEGRITY")
print("Production functions:", len(functions))
print("Batch execution calls:", len(execution_calls))
print("Python syntax: PASS")
print("Source fingerprint: PASS")

print("\nSAFETY")
print("Notebook 15A modified: False")
print("Notebook 15B modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nGUARDED SOURCE VALIDATION: PASS")


# COMMAND ----------


# Step 15.79V.1: Inspect orchestration destination guard
# Read-only — no inference or Delta writes

import ast

print("=" * 65)
print("ORCHESTRATION DESTINATION GUARD INSPECTION")
print("=" * 65)

assert "GUARDED_PRODUCTION_SOURCE" in globals()

source = GUARDED_PRODUCTION_SOURCE
tree = ast.parse(source)

function = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "execute_snapshot_protected_batch_job"
)

print("\nDESTINATION VALIDATION CODE")

lines = source.splitlines()

for node in function.body:
    if isinstance(node, ast.If):
        segment = ast.get_source_segment(source, node)

        if "destination_table" in segment:
            print("\nLine:", node.lineno)
            print(segment)

print("\nALLOWED DESTINATION ASSIGNMENTS")

for node in function.body:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        segment = ast.get_source_segment(source, node)

        if "allowed_destination" in segment:
            print("\nLine:", node.lineno)
            print(segment)

print("\nSAFETY")
print("Production notebook modified: False")
print("Candidate notebook modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nDESTINATION GUARD INSPECTION: COMPLETE")


# COMMAND ----------


# Step 15.79V.2: Corrected guarded-source validation
# Static inspection only — no inference or Delta writes

import ast
import hashlib

print("=" * 65)
print("CORRECTED GUARDED SOURCE VALIDATION")
print("=" * 65)

source = GUARDED_PRODUCTION_SOURCE

expected_hash = (
    "f6f64d25801e1156512b78c74b3242581bb79506e095affc108788796ef85e0c"
)

assert hashlib.sha256(
    source.encode("utf-8")
).hexdigest() == expected_hash

tree = ast.parse(source)

functions = {
    node.name: node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
}

assert len(functions) == 10

expected_tables = {
    "register_batch_identity": (
        "registry_table",
        "workspace.default.telco_15_job_batch_registry"
    ),
    "publish_quality_approved_batch": (
        "destination_table",
        "workspace.default.telco_15_job_dev_predictions"
    ),
    "execute_snapshot_protected_batch_job": (
        "destination_table",
        "workspace.default.telco_15_job_dev_predictions"
    )
}

print("\nDESTINATION GUARDS")

for name, (parameter, expected_table) in expected_tables.items():

    function = functions[name]

    assignments = {}

    for node in function.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    try:
                        assignments[target.id] = ast.literal_eval(
                            node.value
                        )
                    except (ValueError, TypeError):
                        pass

    guards = []

    for node in function.body:
        if not isinstance(node, ast.If):
            continue

        test = node.test

        if not isinstance(test, ast.Compare):
            continue

        if not (
            len(test.ops) == 1
            and isinstance(test.ops[0], ast.NotEq)
            and len(test.comparators) == 1
        ):
            continue

        if not (
            isinstance(test.left, ast.Name)
            and test.left.id == parameter
        ):
            continue

        comparator = test.comparators[0]

        if isinstance(comparator, ast.Constant):
            resolved = comparator.value

        elif isinstance(comparator, ast.Name):
            resolved = assignments.get(comparator.id)

        else:
            resolved = None

        if resolved != expected_table:
            continue

        # Confirm the mismatch branch raises an exception.
        if not any(
            isinstance(statement, ast.Raise)
            for statement in node.body
        ):
            continue

        guards.append(node)

    assert len(guards) == 1, (
        f"Expected exactly one valid guard: {name}"
    )

    guard = guards[0]

    # Verify guard precedes the function's write-related calls.
    write_lines = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {
            "saveAsTable", "merge", "execute"
        }
    ]

    if write_lines:
        assert guard.lineno < min(write_lines)

    print(f"{name}: PASS")
    print("  Allowed:", expected_table)
    print("  Guard line:", guard.lineno)

execution_calls = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "execute_snapshot_protected_batch_job"
]

assert len(execution_calls) == 1

print("\nSOURCE INTEGRITY")
print("Production functions:", len(functions))
print("Controlled execution calls:", len(execution_calls))
print("Python syntax: PASS")
print("SHA-256 fingerprint: PASS")

print("\nSAFETY")
print("Notebook 15A modified: False")
print("Notebook 15B modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nCORRECTED GUARDED SOURCE VALIDATION: PASS")


# COMMAND ----------


# Step 15.79W: Inspect strict publication integrity
# Read-only — no inference or Delta writes

import ast
import hashlib

print("=" * 65)
print("STRICT PUBLICATION INTEGRITY INSPECTION")
print("=" * 65)

source = GUARDED_PRODUCTION_SOURCE

expected_hash = (
    "f6f64d25801e1156512b78c74b3242581bb79506e095affc108788796ef85e0c"
)

assert hashlib.sha256(
    source.encode("utf-8")
).hexdigest() == expected_hash

tree = ast.parse(source)

target_name = "validate_strict_publication_integrity"

matches = [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == target_name
]

assert len(matches) == 1, (
    "Expected exactly one integrity validation function."
)

function = matches[0]

print("\nFUNCTION:", target_name)
print("=" * 65)

print(ast.get_source_segment(source, function))

print("\nINSPECTION SUMMARY")
print("Function found: True")
print("Python syntax: PASS")
print("Source fingerprint: PASS")

print("\nSAFETY")
print("Notebook 15A modified: False")
print("Notebook 15B modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nINTEGRITY FUNCTION INSPECTION: COMPLETE")


# COMMAND ----------


# Step 15.79X: Inspect prediction probability quality checks
# Read-only — no inference or Delta writes

import ast
import hashlib

print("=" * 65)
print("PREDICTION PROBABILITY QUALITY AUDIT")
print("=" * 65)

source = GUARDED_PRODUCTION_SOURCE

expected_hash = (
    "f6f64d25801e1156512b78c74b3242581bb79506e095affc108788796ef85e0c"
)

assert hashlib.sha256(
    source.encode("utf-8")
).hexdigest() == expected_hash

tree = ast.parse(source)

target_name = "validate_batch_predictions"

matches = [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == target_name
]

assert len(matches) == 1

function = matches[0]

print("\nFUNCTION:", target_name)
print("=" * 65)

print(ast.get_source_segment(source, function))

print("\nSAFETY")
print("Notebook 15A modified: False")
print("Notebook 15B modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nPROBABILITY QUALITY AUDIT: COMPLETE")


# COMMAND ----------


# Step 15.79Y: Inspect metadata NULL handling
# Static inspection only — no inference or Delta writes

import ast
import hashlib

print("=" * 65)
print("BATCH METADATA NULL-SAFETY AUDIT")
print("=" * 65)

source = GUARDED_PRODUCTION_SOURCE

expected_hash = (
    "f6f64d25801e1156512b78c74b3242581bb79506e095affc108788796ef85e0c"
)

assert hashlib.sha256(
    source.encode("utf-8")
).hexdigest() == expected_hash

tree = ast.parse(source)

function = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "validate_batch_predictions"
)

print("\nMETADATA VALIDATION EXPRESSIONS")
print("=" * 65)

targets = {
    "metadata_columns",
    "missing_condition",
    "metadata_mismatches",
    "batch_metadata_consistency"
}

for node in function.body:

    if isinstance(node, (ast.Assign, ast.For)):
        segment = ast.get_source_segment(source, node)

        if any(
            target in segment
            for target in targets
        ):
            print("\n" + segment)

print("\nAUDIT SUMMARY")
print("Source fingerprint: PASS")
print("Python syntax: PASS")

print("\nSAFETY")
print("Notebook 15A modified: False")
print("Notebook 15B modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nMETADATA NULL-SAFETY INSPECTION: COMPLETE")


# COMMAND ----------


# Step 15.79Z: Strengthen metadata consistency validation
# In-memory patch only — no inference or Delta writes

import ast
import hashlib

print("=" * 65)
print("NULL-SAFE METADATA VALIDATION PATCH")
print("=" * 65)

source = GUARDED_PRODUCTION_SOURCE

expected_hash = (
    "f6f64d25801e1156512b78c74b3242581bb79506e095affc108788796ef85e0c"
)

assert hashlib.sha256(
    source.encode("utf-8")
).hexdigest() == expected_hash, (
    "Source fingerprint mismatch. Patch stopped."
)

old_block = '''            (F.col("batch_id") != batch_id)
            | (F.col("model_name") != model_name)
            | (F.col("model_version") != model_version)
            | (
                F.col("classification_threshold")
                != threshold
            )
            | (F.col("source_table") != source_table)'''

new_block = '''            ~F.col("batch_id").eqNullSafe(F.lit(batch_id))
            | ~F.col("model_name").eqNullSafe(F.lit(model_name))
            | ~F.col("model_version").eqNullSafe(F.lit(model_version))
            | ~F.col("classification_threshold").eqNullSafe(
                F.lit(threshold)
            )
            | ~F.col("source_table").eqNullSafe(
                F.lit(source_table)
            )'''

assert source.count(old_block) == 1, (
    "Expected metadata block not found exactly once."
)

NULL_SAFE_PRODUCTION_SOURCE = source.replace(
    old_block,
    new_block,
    1
)

tree = ast.parse(NULL_SAFE_PRODUCTION_SOURCE)

function = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "validate_batch_predictions"
)

function_source = ast.get_source_segment(
    NULL_SAFE_PRODUCTION_SOURCE,
    function
)

assert function_source.count("eqNullSafe") == 5

print("\nPATCH VALIDATION")
print("Metadata comparisons updated:", 5)
print("Python syntax: PASS")
print("Original guarded source unchanged: True")

print("\nNEW SOURCE SHA-256")
print(
    hashlib.sha256(
        NULL_SAFE_PRODUCTION_SOURCE.encode("utf-8")
    ).hexdigest()
)

print("\nSAFETY")
print("Notebook 15A modified: False")
print("Notebook 15B modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nNULL-SAFE METADATA PATCH: PASS")


# COMMAND ----------


# Step 15.80A: Verify final patched production source
# Read-only — no inference or Delta writes

import ast
import hashlib

print("=" * 65)
print("FINAL PATCHED SOURCE VERIFICATION")
print("=" * 65)

assert "NULL_SAFE_PRODUCTION_SOURCE" in globals()

source = NULL_SAFE_PRODUCTION_SOURCE

expected_hash = (
    "3979f576572b7578962b0abc0d1a994c0ff3847b37b9b6208714670a795dc2e4"
)

actual_hash = hashlib.sha256(
    source.encode("utf-8")
).hexdigest()

assert actual_hash == expected_hash

tree = ast.parse(source)

functions = {
    node.name: node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
}

assert len(functions) == 10

print("\nFUNCTION VALIDATION")
print("Production functions:", len(functions))

# Confirm the three destination guards remain present.
expected_tables = {
    "workspace.default.telco_15_job_batch_registry",
    "workspace.default.telco_15_job_dev_predictions"
}

for table_name in sorted(expected_tables):
    assert table_name in source
    print("Expected table present:", table_name)

# Confirm all five NULL-safe comparisons.
validation_function = functions["validate_batch_predictions"]

validation_source = ast.get_source_segment(
    source,
    validation_function
)

assert validation_source.count("eqNullSafe") == 5

print("\nMETADATA VALIDATION")
print("NULL-safe comparisons:", 5)

# Confirm the controlled entry point is retained.
execution_calls = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "execute_snapshot_protected_batch_job"
]

assert len(execution_calls) == 1

print("\nEXECUTION CONFIGURATION")
print("Controlled execution calls:", len(execution_calls))

print("\nSOURCE INTEGRITY")
print("SHA-256 fingerprint: PASS")
print("Python syntax: PASS")

print("\nSAFETY")
print("Notebook 15A modified: False")
print("Notebook 15B modified: False")
print("Inference executed: False")
print("Delta tables modified: False")

print("\nFINAL PATCHED SOURCE VERIFICATION: PASS")


# COMMAND ----------


# Step 15.80B: Create separate guarded recovery candidate
# Workspace import only — DO NOT execute candidate

import ast
import base64
import hashlib

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import (
    ExportFormat,
    ImportFormat,
    Language
)

print("=" * 65)
print("GUARDED RECOVERY CANDIDATE CREATION")
print("=" * 65)

source = NULL_SAFE_PRODUCTION_SOURCE

expected_hash = (
    "3979f576572b7578962b0abc0d1a994c0ff3847b37b9b6208714670a795dc2e4"
)

assert hashlib.sha256(
    source.encode("utf-8")
).hexdigest() == expected_hash

ast.parse(source)

candidate_path = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

w = WorkspaceClient()

# Fail safely if the candidate already exists.
existing_paths = {
    item.path
    for item in w.workspace.list(
        "/Telco_Customer_Churn_ML"
    )
}

assert candidate_path not in existing_paths, (
    "Candidate already exists. No overwrite permitted."
)

encoded_source = base64.b64encode(
    source.encode("utf-8")
).decode("utf-8")

w.workspace.import_(
    path=candidate_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=encoded_source,
    overwrite=False
)

# Verify imported content without executing it.
exported = w.workspace.export(
    path=candidate_path,
    format=ExportFormat.SOURCE
)

exported_source = base64.b64decode(
    exported.content
).decode("utf-8")

assert exported_source == source, (
    "Imported candidate does not match validated source."
)

print("\nCANDIDATE DETAILS")
print("Candidate path:", candidate_path)
print("Source fingerprint: PASS")
print("Python syntax: PASS")
print("Exported content matches: True")

print("\nSAFETY")
print("New candidate created: True")
print("Notebook 15A overwritten: False")
print("Notebook 15B overwritten: False")
print("Batch inference executed: False")
print("Delta tables modified by this cell: False")

print("\nGUARDED RECOVERY CANDIDATE: CREATED")


# COMMAND ----------


# Step 15.80C: Independent guarded candidate verification
# Read-only — do not execute Notebook 15C

import ast
import base64
import hashlib

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

print("=" * 65)
print("INDEPENDENT GUARDED CANDIDATE VERIFICATION")
print("=" * 65)

candidate_path = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

expected_hash = (
    "3979f576572b7578962b0abc0d1a994c0ff3847b37b9b6208714670a795dc2e4"
)

w = WorkspaceClient()

exported = w.workspace.export(
    path=candidate_path,
    format=ExportFormat.SOURCE
)

source = base64.b64decode(
    exported.content
).decode("utf-8")

actual_hash = hashlib.sha256(
    source.encode("utf-8")
).hexdigest()

assert actual_hash == expected_hash, (
    "Candidate source fingerprint mismatch."
)

tree = ast.parse(source)

functions = {
    node.name: node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
}

assert len(functions) == 10

# Verify controlled execution call.
execution_calls = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "execute_snapshot_protected_batch_job"
]

assert len(execution_calls) == 1

# Verify expected development configuration.
required_values = {
    "workspace.default.telco_15_job_dev_predictions",
    "workspace.default.telco_15_job_batch_registry",
    "workspace.default.telco_test_raw"
}

source_constants = {
    node.value
    for node in ast.walk(tree)
    if isinstance(node, ast.Constant)
    and isinstance(node.value, str)
}

assert required_values.issubset(source_constants)

# Verify protected destination checks exist.
for function_name in (
    "register_batch_identity",
    "publish_quality_approved_batch",
    "execute_snapshot_protected_batch_job"
):
    function = functions[function_name]

    assert any(
        isinstance(node, ast.Raise)
        for node in ast.walk(function)
    ), f"Missing validation in {function_name}"

# Verify NULL-safe metadata comparisons.
validation_source = ast.get_source_segment(
    source,
    functions["validate_batch_predictions"]
)

assert validation_source.count("eqNullSafe") == 5

print("\nCANDIDATE VERIFICATION")
print("Candidate path:", candidate_path)
print("Source fingerprint: PASS")
print("Python syntax: PASS")
print("Production functions:", len(functions))
print("Controlled execution calls:", len(execution_calls))
print("Required table configuration: PRESENT")
print("Destination validation statements: PRESENT")
print("NULL-safe metadata comparisons: 5")

print("\nSAFETY")
print("Candidate executed: False")
print("Notebook 15A modified: False")
print("Notebook 15B modified: False")
print("Notebook 15C modified: False")
print("Delta tables modified: False")

print("\nINDEPENDENT CANDIDATE VERIFICATION: PASS")


# COMMAND ----------


# Step 15.80D: Inspect batch registry idempotency
# Read-only — no inference or Delta writes

import ast
import base64
import hashlib

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

print("=" * 65)
print("BATCH REGISTRY IDEMPOTENCY AUDIT")
print("=" * 65)

candidate_path = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

expected_hash = (
    "3979f576572b7578962b0abc0d1a994c0ff3847b37b9b6208714670a795dc2e4"
)

w = WorkspaceClient()

exported = w.workspace.export(
    path=candidate_path,
    format=ExportFormat.SOURCE
)

source = base64.b64decode(
    exported.content
).decode("utf-8")

assert hashlib.sha256(
    source.encode("utf-8")
).hexdigest() == expected_hash

tree = ast.parse(source)

matches = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "register_batch_identity"
]

assert len(matches) == 1

print("\nFUNCTION: register_batch_identity")
print("=" * 65)

print(ast.get_source_segment(source, matches[0]))

print("\nSAFETY")
print("Notebook 15A modified: False")
print("Notebook 15B modified: False")
print("Notebook 15C modified: False")
print("Batch inference executed: False")
print("Delta tables modified: False")

print("\nREGISTRY IDEMPOTENCY INSPECTION: COMPLETE")


# COMMAND ----------


# Step 15.80E: Inspect batch registry table metadata
# Read-only — no inference or Delta writes

print("=" * 65)
print("BATCH REGISTRY TABLE METADATA AUDIT")
print("=" * 65)

registry_table = (
    "workspace.default.telco_15_job_batch_registry"
)

assert spark.catalog.tableExists(registry_table), (
    "Batch registry table does not exist."
)

print("\nTABLE SCHEMA")
spark.table(registry_table).printSchema()

print("\nTABLE DETAILS")
details = spark.sql(
    f"DESCRIBE DETAIL {registry_table}"
)

details.select(
    "format",
    "numFiles",
    "properties"
).show(truncate=False)

print("\nTABLE CONSTRAINTS")

properties = details.select("properties").first()["properties"]

constraint_properties = {
    key: value
    for key, value in (properties or {}).items()
    if "constraint" in key.lower()
}

print("Constraint properties:", constraint_properties)

print("\nSAFETY")
print("Notebook 15A modified: False")
print("Notebook 15B modified: False")
print("Notebook 15C modified: False")
print("Batch inference executed: False")
print("Delta tables modified: False")

print("\nREGISTRY METADATA AUDIT: COMPLETE")


# COMMAND ----------


# Step 15.80F: Audit existing batch registry records
# Read-only — no inference or Delta writes

from pyspark.sql import functions as F

print("=" * 65)
print("BATCH REGISTRY DATA INTEGRITY AUDIT")
print("=" * 65)

registry_table = (
    "workspace.default.telco_15_job_batch_registry"
)

df = spark.table(registry_table)

total_records = df.count()

distinct_batch_ids = (
    df.select("batch_id").distinct().count()
)

null_batch_ids = (
    df.filter(
        F.col("batch_id").isNull()
        | (F.trim(F.col("batch_id")) == "")
    ).count()
)

duplicate_batches = (
    df.groupBy("batch_id")
    .count()
    .filter(F.col("count") > 1)
)

duplicate_batch_count = duplicate_batches.count()

print("\nREGISTRY SUMMARY")
print("Total records:", total_records)
print("Distinct batch IDs:", distinct_batch_ids)
print("Null or empty batch IDs:", null_batch_ids)
print("Duplicate batch IDs:", duplicate_batch_count)

print("\nDUPLICATE DETAILS")
duplicate_batches.show(truncate=False)

print("\nINTEGRITY STATUS")

if (
    null_batch_ids == 0
    and duplicate_batch_count == 0
):
    print("Existing registry data: PASS")
else:
    print("Existing registry data: REVIEW REQUIRED")

print("\nSAFETY")
print("Notebook 15A modified: False")
print("Notebook 15B modified: False")
print("Notebook 15C modified: False")
print("Batch inference executed: False")
print("Delta tables modified: False")

print("\nREGISTRY DATA AUDIT: COMPLETE")


# COMMAND ----------


# Step 15.80G: Audit controlled registry configuration
# Read-only — no inference or Delta writes

import ast
import base64
import hashlib

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

print("=" * 65)
print("CONTROLLED REGISTRY CONFIGURATION AUDIT")
print("=" * 65)

candidate_path = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

expected_hash = (
    "3979f576572b7578962b0abc0d1a994c0ff3847b37b9b6208714670a795dc2e4"
)

w = WorkspaceClient()

exported = w.workspace.export(
    path=candidate_path,
    format=ExportFormat.SOURCE
)

source = base64.b64decode(
    exported.content
).decode("utf-8")

assert hashlib.sha256(
    source.encode("utf-8")
).hexdigest() == expected_hash

tree = ast.parse(source)

expected_configuration = {
    "JOB_BATCH_REGISTRY_TABLE":
        "workspace.default.telco_15_job_batch_registry",
    "JOB_DEV_OUTPUT_TABLE":
        "workspace.default.telco_15_job_dev_predictions",
    "JOB_BATCH_ID":
        "d972dd16-029d-489f-a406-3c6c6e1e6265"
}

assignments = {}

for node in tree.body:
    if not isinstance(node, ast.Assign):
        continue

    if len(node.targets) != 1:
        continue

    target = node.targets[0]

    if not isinstance(target, ast.Name):
        continue

    if target.id not in expected_configuration:
        continue

    assignments[target.id] = ast.literal_eval(node.value)

print("\nCONTROLLED CONFIGURATION")

for name, expected_value in expected_configuration.items():
    actual_value = assignments.get(name)

    assert actual_value == expected_value, (
        f"Unexpected configuration: {name}"
    )

    print(f"{name}: {actual_value}")
    print("Status: PASS")

execution_calls = [
    node
    for node in tree.body
    if isinstance(node, ast.Assign)
    and isinstance(node.value, ast.Call)
    and isinstance(node.value.func, ast.Name)
    and node.value.func.id == "execute_snapshot_protected_batch_job"
]

assert len(execution_calls) == 1

call = execution_calls[0].value

keyword_arguments = {
    keyword.arg: keyword.value
    for keyword in call.keywords
}

assert isinstance(
    keyword_arguments["registry_table"],
    ast.Name
)

assert (
    keyword_arguments["registry_table"].id
    == "JOB_BATCH_REGISTRY_TABLE"
)

assert isinstance(
    keyword_arguments["destination_table"],
    ast.Name
)

assert (
    keyword_arguments["destination_table"].id
    == "JOB_DEV_OUTPUT_TABLE"
)

print("\nEXECUTION VALIDATION")
print("Top-level execution calls:", len(execution_calls))
print("Registry argument: PASS")
print("Development destination argument: PASS")

print("\nSAFETY")
print("Notebook 15A modified: False")
print("Notebook 15B modified: False")
print("Notebook 15C modified: False")
print("Batch inference executed: False")
print("Delta tables modified: False")

print("\nREGISTRY CONFIGURATION AUDIT: PASS")


# COMMAND ----------


# Step 15.80H: Inspect planned job concurrency settings
# Read-only — no job creation, inference, or Delta writes

print("=" * 65)
print("PLANNED JOB CONCURRENCY AUDIT")
print("=" * 65)

assert "JOB_DEFINITION" in globals(), (
    "JOB_DEFINITION is not available."
)

job = JOB_DEFINITION

print("\nJOB DEFINITION TYPE")
print(type(job).__name__)

# Support either a dictionary or an SDK object.
if isinstance(job, dict):
    job_data = job
elif hasattr(job, "as_dict"):
    job_data = job.as_dict()
else:
    raise TypeError(
        "Unsupported JOB_DEFINITION type."
    )

print("\nCONCURRENCY CONFIGURATION")

max_concurrent = job_data.get("max_concurrent_runs")
queue_settings = job_data.get("queue", {})

print("Maximum concurrent runs:", max_concurrent)
print("Queue settings:", queue_settings)

print("\nTASK NOTEBOOK PATHS")

tasks = job_data.get("tasks", [])

for task in tasks:
    notebook_task = task.get("notebook_task") or {}

    print("Task key:", task.get("task_key"))
    print(
        "Notebook path:",
        notebook_task.get("notebook_path")
    )

print("\nCONFIGURATION ASSESSMENT")

if max_concurrent == 1:
    print("Single-job concurrency limit: PASS")
else:
    print("Single-job concurrency limit: REVIEW REQUIRED")

expected_candidate = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

candidate_targeted = any(
    (task.get("notebook_task") or {}).get("notebook_path")
    == expected_candidate
    for task in tasks
)

print("Notebook 15C targeted:", candidate_targeted)

if not candidate_targeted:
    print(
        "Job target requires review before job creation."
    )

print("\nSAFETY")
print("Databricks Job created: False")
print("Databricks Job executed: False")
print("Notebook 15C executed: False")
print("Delta tables modified: False")

print("\nJOB CONCURRENCY AUDIT: COMPLETE")


# COMMAND ----------


# Step 15.80I: Prepare corrected job definition
# In-memory only — no job creation or execution

import copy

print("=" * 65)
print("CORRECTED DATABRICKS JOB DEFINITION")
print("=" * 65)

assert isinstance(JOB_DEFINITION, dict)

CORRECTED_JOB_DEFINITION = copy.deepcopy(
    JOB_DEFINITION
)

expected_candidate = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

tasks = CORRECTED_JOB_DEFINITION.get("tasks", [])

assert len(tasks) == 1, (
    "Expected exactly one job task."
)

task = tasks[0]

assert task.get("task_key") == (
    "telco_snapshot_batch_inference"
)

assert isinstance(task.get("notebook_task"), dict)

old_path = task["notebook_task"]["notebook_path"]

assert old_path == (
    "/Telco_Customer_Churn_ML/"
    "15_Telco_Production_Model_Deployment"
), "Unexpected original notebook path."

task["notebook_task"]["notebook_path"] = (
    expected_candidate
)

assert (
    CORRECTED_JOB_DEFINITION["max_concurrent_runs"]
    == 1
)

assert (
    CORRECTED_JOB_DEFINITION["queue"]["enabled"]
    is True
)

assert (
    CORRECTED_JOB_DEFINITION["tasks"][0]
    ["notebook_task"]["notebook_path"]
    == expected_candidate
)

assert (
    JOB_DEFINITION["tasks"][0]
    ["notebook_task"]["notebook_path"]
    == old_path
)

print("\nNOTEBOOK TARGET")
print("Original:", old_path)
print("Corrected:", expected_candidate)

print("\nJOB SETTINGS")
print(
    "Maximum concurrent runs:",
    CORRECTED_JOB_DEFINITION["max_concurrent_runs"]
)
print(
    "Queue enabled:",
    CORRECTED_JOB_DEFINITION["queue"]["enabled"]
)
print("Task count:", len(tasks))

print("\nVALIDATION")
print("Original definition preserved: PASS")
print("Corrected notebook target: PASS")
print("Concurrency configuration: PASS")

print("\nSAFETY")
print("Databricks Job created: False")
print("Databricks Job executed: False")
print("Notebook 15A modified: False")
print("Notebook 15B modified: False")
print("Notebook 15C modified: False")
print("Delta tables modified: False")

print("\nCORRECTED JOB DEFINITION: PREPARED")


# COMMAND ----------


# Step 15.80J: Audit registry write references
# Read-only — no inference, jobs, or Delta writes

import ast
import base64
import hashlib

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

print("=" * 65)
print("BATCH REGISTRY WRITE-PATH AUDIT")
print("=" * 65)

candidate_path = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

expected_hash = (
    "3979f576572b7578962b0abc0d1a994c0ff3847b37b9b6208714670a795dc2e4"
)

w = WorkspaceClient()

exported = w.workspace.export(
    path=candidate_path,
    format=ExportFormat.SOURCE
)

source = base64.b64decode(
    exported.content
).decode("utf-8")

actual_hash = hashlib.sha256(
    source.encode("utf-8")
).hexdigest()

assert actual_hash == expected_hash, (
    "Candidate source fingerprint changed."
)

tree = ast.parse(source)

registry_references = []

for node in ast.walk(tree):

    if isinstance(node, ast.Constant):
        if (
            isinstance(node.value, str)
            and "telco_15_job_batch_registry" in node.value
        ):
            registry_references.append(node.lineno)

write_methods = {
    "saveAsTable",
    "insertInto",
    "execute",
    "sql",
    "save"
}

write_calls = []

for node in ast.walk(tree):

    if not isinstance(node, ast.Call):
        continue

    if not isinstance(node.func, ast.Attribute):
        continue

    if node.func.attr not in write_methods:
        continue

    write_calls.append(
        (node.lineno, node.func.attr)
    )

print("\nREGISTRY TABLE REFERENCES")
print("Reference lines:", sorted(registry_references))

print("\nPOTENTIAL WRITE/SQL CALLS")

source_lines = source.splitlines()

for lineno, method in sorted(write_calls):

    print(f"\nLine {lineno}: {method}")

    start = max(0, lineno - 3)
    end = min(len(source_lines), lineno + 2)

    for i in range(start, end):
        print(f"{i + 1}: {source_lines[i]}")

print("\nSUMMARY")
print("Registry literal references:", len(registry_references))
print("Potential write/SQL calls:", len(write_calls))
print("Source fingerprint: PASS")

print("\nSAFETY")
print("Notebook 15C modified: False")
print("Job created: False")
print("Batch inference executed: False")
print("Delta tables modified: False")

print("\nREGISTRY WRITE-PATH AUDIT: COMPLETE")


# COMMAND ----------


# Step 15.80K: Inspect registry initialization and writer
# Read-only — no inference or Delta writes

import ast
import base64
import hashlib

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

print("=" * 65)
print("REGISTRY INITIALIZATION AND WRITE AUDIT")
print("=" * 65)

candidate_path = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

expected_hash = (
    "3979f576572b7578962b0abc0d1a994c0ff3847b37b9b6208714670a795dc2e4"
)

exported = WorkspaceClient().workspace.export(
    path=candidate_path,
    format=ExportFormat.SOURCE
)

source = base64.b64decode(
    exported.content
).decode("utf-8")

assert hashlib.sha256(
    source.encode("utf-8")
).hexdigest() == expected_hash

lines = source.splitlines()

print("\nREGISTRY-RELATED SOURCE: LINES 1–100")

for i in range(min(100, len(lines))):
    print(f"{i + 1:4}: {lines[i]}")

print("\nREGISTRATION IMPLEMENTATION: LINES 210–325")

for i in range(209, min(325, len(lines))):
    print(f"{i + 1:4}: {lines[i]}")

print("\nSAFETY")
print("Source fingerprint: PASS")
print("Notebook 15C modified: False")
print("Job created: False")
print("Batch inference executed: False")
print("Delta tables modified: False")

print("\nREGISTRY IMPLEMENTATION INSPECTION: COMPLETE")


# COMMAND ----------


# Step 15.80L: Inspect existing Databricks Job references
# Read-only — no job creation, execution, or Delta writes

from databricks.sdk import WorkspaceClient

print("=" * 65)
print("EXISTING DATABRICKS JOB REFERENCE AUDIT")
print("=" * 65)

w = WorkspaceClient()

candidate_path = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

development_path = (
    "/Telco_Customer_Churn_ML/"
    "15_Telco_Production_Model_Deployment"
)

matching_jobs = []
total_jobs = 0

for job in w.jobs.list(expand_tasks=True):
    total_jobs += 1

    settings = job.settings

    if settings is None:
        continue

    for task in settings.tasks or []:

        notebook_task = task.notebook_task

        if notebook_task is None:
            continue

        path = notebook_task.notebook_path or ""

        if path in (candidate_path, development_path):
            matching_jobs.append({
                "job_id": job.job_id,
                "job_name": settings.name,
                "task_key": task.task_key,
                "notebook_path": path
            })

print("\nJOB INVENTORY")
print("Jobs inspected:", total_jobs)

print("\nMATCHING NOTEBOOK REFERENCES")
print("Matches found:", len(matching_jobs))

for match in matching_jobs:
    print("-" * 50)
    for key, value in match.items():
        print(f"{key}: {value}")

print("\nASSESSMENT")

if not matching_jobs:
    print(
        "No existing jobs directly reference "
        "the development or guarded notebook."
    )
else:
    print(
        "Existing job references require review."
    )

print("\nLIMITATION")
print(
    "This scan cannot detect every indirect registry "
    "writer or manual notebook execution."
)

print("\nSAFETY")
print("Job created or updated: False")
print("Job executed: False")
print("Notebook 15C modified: False")
print("Batch inference executed: False")
print("Delta tables modified: False")

print("\nJOB REFERENCE AUDIT: COMPLETE")


# COMMAND ----------


# Step 15.80M: Verify controlled batch ID is unused
# Read-only — no inference or Delta writes

from pyspark.sql import functions as F

print("=" * 65)
print("CONTROLLED BATCH ID PRE-EXECUTION AUDIT")
print("=" * 65)

batch_id = "d972dd16-029d-489f-a406-3c6c6e1e6265"

tables = {
    "Registry":
        "workspace.default.telco_15_job_batch_registry",
    "Development predictions":
        "workspace.default.telco_15_job_dev_predictions",
    "Approved predictions":
        "workspace.default.telco_15_approved_predictions"
}

results = {}

for label, table_name in tables.items():

    assert spark.catalog.tableExists(table_name), (
        f"Expected table missing: {table_name}"
    )

    df = spark.table(table_name)

    assert "batch_id" in df.columns, (
        f"batch_id column missing: {table_name}"
    )

    matching_count = (
        df.filter(F.col("batch_id") == batch_id)
        .count()
    )

    results[label] = matching_count

    print(f"\n{label}")
    print("Table:", table_name)
    print("Matching batch records:", matching_count)

print("\nASSESSMENT")

if all(count == 0 for count in results.values()):
    print("Controlled batch ID: UNUSED")
    print("Pre-execution identity check: PASS")
else:
    print("Controlled batch ID: ALREADY PRESENT")
    print("Pre-execution identity check: REVIEW REQUIRED")

print("\nSAFETY")
print("Registry modified: False")
print("Development predictions modified: False")
print("Approved predictions modified: False")
print("Notebook 15C executed: False")
print("Databricks Job created: False")

print("\nBATCH ID PRE-EXECUTION AUDIT: COMPLETE")


# COMMAND ----------


# Step 15.80N: Inspect registry table permissions
# Read-only — no inference or Delta writes

print("=" * 65)
print("BATCH REGISTRY ACCESS CONTROL AUDIT")
print("=" * 65)

registry_table = (
    "workspace.default.telco_15_job_batch_registry"
)

print("\nCURRENT USER")

current_user = spark.sql(
    "SELECT current_user() AS user_name"
).first()["user_name"]

print("User:", current_user)

print("\nREGISTRY TABLE GRANTS")

try:
    grants = spark.sql(
        f"SHOW GRANTS ON TABLE {registry_table}"
    )

    grants.show(100, truncate=False)

    print("\nGrant inspection: COMPLETE")

except Exception as exc:
    print("Grant inspection unavailable.")
    print("Error type:", type(exc).__name__)
    print("Error:", str(exc)[:500])

print("\nIMPORTANT")
print(
    "Table grants alone do not establish exclusive "
    "write access. Schema-level privileges, ownership, "
    "group membership, and other access paths may matter."
)

print("\nSAFETY")
print("Registry modified: False")
print("Development predictions modified: False")
print("Approved predictions modified: False")
print("Notebook 15C executed: False")
print("Databricks Job created: False")

print("\nREGISTRY ACCESS CONTROL AUDIT: COMPLETE")


# COMMAND ----------


# Step 15.80O: Inspect registry ownership and schema grants
# Read-only — no inference, job creation, or Delta writes

print("=" * 65)
print("REGISTRY OWNERSHIP AND INHERITED ACCESS AUDIT")
print("=" * 65)

registry_table = (
    "workspace.default.telco_15_job_batch_registry"
)

schema_name = "workspace.default"

print("\nTABLE OWNERSHIP")

try:
    table_info = spark.sql(
        f"DESCRIBE TABLE EXTENDED {registry_table}"
    )

    owner_rows = (
        table_info
        .filter("lower(col_name) = 'owner'")
        .collect()
    )

    if owner_rows:
        for row in owner_rows:
            print("Table owner:", row["data_type"])
    else:
        print("Table owner: Not returned")

except Exception as exc:
    print("Ownership inspection unavailable.")
    print("Error:", str(exc)[:300])

print("\nSCHEMA GRANTS")

try:
    spark.sql(
        f"SHOW GRANTS ON SCHEMA {schema_name}"
    ).show(100, truncate=False)

except Exception as exc:
    print("Schema grant inspection unavailable.")
    print("Error:", str(exc)[:300])

print("\nCATALOG GRANTS")

try:
    spark.sql(
        "SHOW GRANTS ON CATALOG workspace"
    ).show(100, truncate=False)

except Exception as exc:
    print("Catalog grant inspection unavailable.")
    print("Error:", str(exc)[:300])

print("\nSAFETY")
print("Registry modified: False")
print("Development predictions modified: False")
print("Approved predictions modified: False")
print("Notebook 15C modified: False")
print("Databricks Job created: False")
print("Batch inference executed: False")

print("\nOWNERSHIP AND ACCESS AUDIT: COMPLETE")


# COMMAND ----------


# Step 15.80P: Prepare atomic registration candidate
# In-memory source only — no Delta writes

import ast

print("=" * 65)
print("ATOMIC REGISTRATION PATCH PREPARATION")
print("=" * 65)

ATOMIC_REGISTRATION_SOURCE = '''
def register_batch_identity_atomic_candidate(
    batch_id,
    source_table,
    source_fingerprint,
    record_count,
    model_name,
    model_version,
    threshold,
    registry_table
):
    """
    Candidate implementation only.
    Not approved for execution.
    """

    if registry_table != (
        "workspace.default.telco_15_job_batch_registry"
    ):
        raise ValueError("Unauthorized registry destination.")

    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ValueError("Invalid batch ID.")

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold.")

    identity = {
        "batch_id": batch_id,
        "source_table": source_table,
        "source_fingerprint": source_fingerprint,
        "record_count": int(record_count),
        "model_name": model_name,
        "model_version": str(model_version),
        "classification_threshold": float(threshold)
    }

    existing = (
        spark.table(registry_table)
        .filter(F.col("batch_id") == batch_id)
        .limit(2)
        .collect()
    )

    if len(existing) > 1:
        raise RuntimeError("Duplicate registry entries.")

    if existing:
        stored = existing[0].asDict()

        for field, expected in identity.items():
            if stored[field] != expected:
                raise ValueError(
                    f"Batch identity conflict: {field}"
                )

        timestamp = stored["batch_timestamp"]

        if timestamp is not None and timestamp.tzinfo is None:
            timestamp = timestamp.replace(
                tzinfo=timezone.utc
            )

        return {
            "batch_id": batch_id,
            "registration_status": "RETRY",
            "batch_timestamp": timestamp
        }

    timestamp = datetime.now(timezone.utc)

    new_record = {
        **identity,
        "batch_timestamp": timestamp,
        "registered_at": timestamp
    }

    source_df = spark.createDataFrame(
        [new_record],
        schema=registry_schema
    )

    delta_registry = DeltaTable.forName(
        spark,
        registry_table
    )

    (
        delta_registry.alias("target")
        .merge(
            source_df.alias("source"),
            "target.batch_id = source.batch_id"
        )
        .whenNotMatchedInsertAll()
        .execute()
    )

    matching = (
        spark.table(registry_table)
        .filter(F.col("batch_id") == batch_id)
        .limit(2)
        .collect()
    )

    if len(matching) != 1:
        raise RuntimeError(
            "Registry post-write uniqueness check failed."
        )

    stored = matching[0].asDict()

    for field, expected in identity.items():
        if stored[field] != expected:
            raise ValueError(
                f"Post-write identity conflict: {field}"
            )

    stored_timestamp = stored["batch_timestamp"]

    if (
        stored_timestamp is not None
        and stored_timestamp.tzinfo is None
    ):
        stored_timestamp = stored_timestamp.replace(
            tzinfo=timezone.utc
        )

    return {
        "batch_id": batch_id,
        "registration_status": (
            "NEW"
            if stored_timestamp == timestamp
            else "RETRY"
        ),
        "batch_timestamp": stored_timestamp
    }
'''

tree = ast.parse(ATOMIC_REGISTRATION_SOURCE)

functions = [
    node.name
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
]

assert functions == [
    "register_batch_identity_atomic_candidate"
]

print("Candidate function:", functions[0])
print("Python syntax: PASS")
print("Candidate executed: False")
print("Existing registry modified: False")
print("Notebook 15C modified: False")
print("Batch inference executed: False")

print("\nPATCH PREPARATION: COMPLETE")


# COMMAND ----------


# Step 15.80Q: Static audit of atomic registration candidate
# Read-only — no Delta writes or inference

import ast
import hashlib

print("=" * 65)
print("ATOMIC REGISTRATION CANDIDATE SAFETY AUDIT")
print("=" * 65)

assert "ATOMIC_REGISTRATION_SOURCE" in globals(), (
    "Run Step 15.80P first."
)

source = ATOMIC_REGISTRATION_SOURCE
tree = ast.parse(source)

functions = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
]

assert len(functions) == 1

function = functions[0]

assert (
    function.name
    == "register_batch_identity_atomic_candidate"
)

methods = [
    node.func.attr
    for node in ast.walk(function)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
]

merge_count = methods.count("merge")
execute_count = methods.count("execute")
collect_count = methods.count("collect")

timestamp_comparisons = [
    node for node in ast.walk(function)
    if isinstance(node, ast.Compare)
    and any(
        isinstance(op, ast.Eq)
        for op in node.ops
    )
    and any(
        isinstance(child, ast.Name)
        and child.id in ("stored_timestamp", "timestamp")
        for child in ast.walk(node)
    )
]

print("\nSOURCE")
print(
    "SHA-256:",
    hashlib.sha256(source.encode()).hexdigest()
)

print("\nOPERATIONS")
print("Delta MERGE calls:", merge_count)
print("Execute calls:", execute_count)
print("Collect calls:", collect_count)
print(
    "Timestamp equality comparisons:",
    len(timestamp_comparisons)
)

assert merge_count == 1
assert execute_count == 1
assert collect_count == 2

print("\nDESIGN REVIEW")
print("MERGE-based insertion: PRESENT")
print("Pre-write identity inspection: PRESENT")
print("Post-write identity inspection: PRESENT")
print("Atomic uniqueness guarantee: NOT ESTABLISHED")
print("Reliable NEW/RETRY classification: NOT ESTABLISHED")
print("Concurrency retry handling: NOT IMPLEMENTED")

print("\nSAFETY")
print("Candidate executed: False")
print("Registry modified: False")
print("Notebook 15C modified: False")
print("Databricks Job created: False")
print("Batch inference executed: False")
print("Approved predictions modified: False")

print("\nCANDIDATE SAFETY AUDIT: COMPLETE")


# COMMAND ----------


# Step 15.80R: Prepare controlled single-writer policy
# In-memory only — no inference, job creation, or Delta writes

print("=" * 65)
print("CONTROLLED SINGLE-WRITER EXECUTION POLICY")
print("=" * 65)

SINGLE_WRITER_POLICY = {
    "execution_mode": "controlled_development",
    "authorized_notebook": (
        "/Telco_Customer_Churn_ML/"
        "15C_Telco_Guarded_Batch_Job_Candidate"
    ),
    "registry_table": (
        "workspace.default.telco_15_job_batch_registry"
    ),
    "prediction_destination": (
        "workspace.default.telco_15_job_dev_predictions"
    ),
    "protected_table": (
        "workspace.default.telco_15_approved_predictions"
    ),
    "max_concurrent_runs": 1,
    "queue_enabled": True,
    "manual_parallel_execution_permitted": False,
    "external_registry_writers_permitted": False,
    "pre_execution_registry_check_required": True,
    "post_execution_registry_check_required": True,
    "atomic_uniqueness_guaranteed": False
}

assert (
    SINGLE_WRITER_POLICY["max_concurrent_runs"] == 1
)

assert SINGLE_WRITER_POLICY["queue_enabled"] is True

assert (
    SINGLE_WRITER_POLICY["prediction_destination"]
    != SINGLE_WRITER_POLICY["protected_table"]
)

assert (
    SINGLE_WRITER_POLICY["atomic_uniqueness_guaranteed"]
    is False
)

for key, value in SINGLE_WRITER_POLICY.items():
    print(f"{key}: {value}")

print("\nPOLICY STATUS")
print("Policy structure: PASS")
print("Operational restrictions enforced: NOT YET")
print("Concurrent-writer safety guaranteed: NO")

print("\nSAFETY")
print("Notebook 15C modified: False")
print("Job created: False")
print("Batch inference executed: False")
print("Registry modified: False")
print("Approved predictions modified: False")

print("\nSINGLE-WRITER POLICY: PREPARED")


# COMMAND ----------


# Step 15.80S: Validate corrected job against execution policy
# Read-only — no job creation, inference, or Delta writes

print("=" * 65)
print("JOB AND SINGLE-WRITER POLICY ALIGNMENT AUDIT")
print("=" * 65)

assert "CORRECTED_JOB_DEFINITION" in globals()
assert "SINGLE_WRITER_POLICY" in globals()

job = CORRECTED_JOB_DEFINITION
policy = SINGLE_WRITER_POLICY

assert isinstance(job, dict)
assert isinstance(policy, dict)

tasks = job.get("tasks", [])

assert len(tasks) == 1, (
    "Expected exactly one task."
)

task = tasks[0]

actual_path = (
    task.get("notebook_task") or {}
).get("notebook_path")

assert actual_path == policy["authorized_notebook"]

assert (
    job.get("max_concurrent_runs")
    == policy["max_concurrent_runs"]
    == 1
)

assert (
    job.get("queue", {}).get("enabled")
    is True
)

assert (
    policy["prediction_destination"]
    != policy["protected_table"]
)

assert (
    policy["manual_parallel_execution_permitted"]
    is False
)

assert (
    policy["external_registry_writers_permitted"]
    is False
)

assert (
    policy["atomic_uniqueness_guaranteed"]
    is False
)

print("\nJOB CONFIGURATION")
print("Task count:", len(tasks))
print("Task key:", task.get("task_key"))
print("Notebook path:", actual_path)
print(
    "Maximum concurrent runs:",
    job["max_concurrent_runs"]
)
print(
    "Queue enabled:",
    job["queue"]["enabled"]
)

print("\nPOLICY VALIDATION")
print("Authorized notebook: PASS")
print("Single-job concurrency: PASS")
print("Queue configuration: PASS")
print("Protected destination separation: PASS")
print("Single-writer policy structure: PASS")

print("\nIMPORTANT LIMITATIONS")
print("External writer exclusion verified: NO")
print("Manual execution exclusion enforced: NO")
print("Atomic uniqueness guaranteed: NO")

print("\nSAFETY")
print("Job created: False")
print("Job executed: False")
print("Notebook 15C modified: False")
print("Batch inference executed: False")
print("Delta tables modified: False")

print("\nJOB-POLICY ALIGNMENT AUDIT: COMPLETE")


# COMMAND ----------


# Step 15.80U: Final pre-execution safety verification
# Read-only — no job creation, inference, or Delta writes

import base64
import hashlib

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat
from pyspark.sql import functions as F

print("=" * 65)
print("FINAL PRE-EXECUTION SAFETY VERIFICATION")
print("=" * 65)

candidate_path = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

expected_hash = (
    "3979f576572b7578962b0abc0d1a994c0ff3847b37b9b6208714670a795dc2e4"
)

registry = "workspace.default.telco_15_job_batch_registry"
development = "workspace.default.telco_15_job_dev_predictions"
approved = "workspace.default.telco_15_approved_predictions"

batch_id = "d972dd16-029d-489f-a406-3c6c6e1e6265"

# 1. Verify candidate source
exported = WorkspaceClient().workspace.export(
    path=candidate_path,
    format=ExportFormat.SOURCE
)

source = base64.b64decode(
    exported.content
).decode("utf-8")

actual_hash = hashlib.sha256(
    source.encode("utf-8")
).hexdigest()

assert actual_hash == expected_hash

print("\nCANDIDATE SOURCE")
print("Fingerprint: PASS")

# 2. Verify corrected job definition
assert CORRECTED_JOB_DEFINITION["max_concurrent_runs"] == 1
assert CORRECTED_JOB_DEFINITION["queue"]["enabled"] is True

tasks = CORRECTED_JOB_DEFINITION["tasks"]
assert len(tasks) == 1

assert (
    tasks[0]["notebook_task"]["notebook_path"]
    == candidate_path
)

print("\nJOB CONFIGURATION")
print("Notebook target: PASS")
print("Single-run concurrency: PASS")
print("Queue enabled: PASS")

# 3. Verify registry and batch identity
registry_df = spark.table(registry)

registry_count = registry_df.count()

duplicate_count = (
    registry_df.groupBy("batch_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

invalid_count = registry_df.filter(
    F.col("batch_id").isNull()
    | (F.trim(F.col("batch_id")) == "")
).count()

assert duplicate_count == 0
assert invalid_count == 0

print("\nREGISTRY")
print("Records:", registry_count)
print("Duplicate batch IDs:", duplicate_count)
print("Invalid batch IDs:", invalid_count)
print("Integrity: PASS")

# 4. Verify controlled batch ID is unused
for table_name in (registry, development, approved):
    matching = spark.table(table_name).filter(
        F.col("batch_id") == batch_id
    ).count()

    assert matching == 0, (
        f"Controlled batch already exists: {table_name}"
    )

print("\nCONTROLLED BATCH")
print("Batch ID unused: PASS")

# 5. Capture approved predictions baseline
approved_count = spark.table(approved).count()

approved_version = spark.sql(
    f"DESCRIBE HISTORY {approved}"
).select("version").first()["version"]

print("\nPROTECTED APPROVED PREDICTIONS")
print("Record count:", approved_count)
print("Current Delta version:", approved_version)
print("Baseline captured: PASS")

print("\nSAFETY")
print("Job created: False")
print("Job executed: False")
print("Notebook 15C executed: False")
print("Registry modified: False")
print("Development predictions modified: False")
print("Approved predictions modified: False")

print("\nFINAL PRE-EXECUTION VERIFICATION: PASS")


# COMMAND ----------


# Step 15.80V: Validate Databricks Job creation request
# In-memory only — no job creation or inference

from databricks.sdk.service.jobs import CreateJob

print("=" * 65)
print("DATABRICKS JOB CREATION REQUEST VALIDATION")
print("=" * 65)

assert "CORRECTED_JOB_DEFINITION" in globals()

job_config = CORRECTED_JOB_DEFINITION

expected_notebook = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

assert job_config["max_concurrent_runs"] == 1
assert job_config["queue"]["enabled"] is True

assert len(job_config["tasks"]) == 1

assert (
    job_config["tasks"][0]["notebook_task"]["notebook_path"]
    == expected_notebook
)

# Convert the validated dictionary to a Databricks SDK object.
# This does NOT submit the request.

PREPARED_JOB_REQUEST = CreateJob.from_dict(
    job_config
)

assert PREPARED_JOB_REQUEST.max_concurrent_runs == 1

assert (
    PREPARED_JOB_REQUEST.tasks[0]
    .notebook_task.notebook_path
    == expected_notebook
)

print("\nJOB REQUEST")
print("Job name:", PREPARED_JOB_REQUEST.name)
print(
    "Maximum concurrent runs:",
    PREPARED_JOB_REQUEST.max_concurrent_runs
)
print("Task count:", len(PREPARED_JOB_REQUEST.tasks))
print(
    "Notebook target:",
    PREPARED_JOB_REQUEST.tasks[0]
    .notebook_task.notebook_path
)

print("\nVALIDATION")
print("SDK request structure: PASS")
print("Notebook target: PASS")
print("Single-run concurrency: PASS")

print("\nSAFETY")
print("Job creation API called: False")
print("Job created: False")
print("Job executed: False")
print("Notebook 15C executed: False")
print("Delta tables modified: False")

print("\nJOB CREATION REQUEST: READY FOR REVIEW")


# COMMAND ----------


# Step 15.80V.1: Correct Databricks SDK validation
# In-memory only — no job creation or execution

from databricks.sdk.service.jobs import JobSettings

print("=" * 65)
print("DATABRICKS JOB SDK COMPATIBILITY VALIDATION")
print("=" * 65)

assert "CORRECTED_JOB_DEFINITION" in globals()

job_config = CORRECTED_JOB_DEFINITION

expected_notebook = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

# Validate the original dictionary
assert job_config["max_concurrent_runs"] == 1
assert job_config["queue"]["enabled"] is True
assert len(job_config["tasks"]) == 1

assert (
    job_config["tasks"][0]["notebook_task"]["notebook_path"]
    == expected_notebook
)

# Deserialize job settings using the installed SDK.
# This does not call the Databricks Jobs API.
PREPARED_JOB_SETTINGS = JobSettings.from_dict(
    job_config
)

assert PREPARED_JOB_SETTINGS.max_concurrent_runs == 1

assert len(PREPARED_JOB_SETTINGS.tasks) == 1

assert (
    PREPARED_JOB_SETTINGS.tasks[0]
    .notebook_task.notebook_path
    == expected_notebook
)

# Prevent accidental automatic execution.
assert not job_config.get("schedule")
assert not job_config.get("trigger")
assert not job_config.get("continuous")

print("\nJOB SETTINGS")
print("Job name:", PREPARED_JOB_SETTINGS.name)
print(
    "Maximum concurrent runs:",
    PREPARED_JOB_SETTINGS.max_concurrent_runs
)
print("Task count:", len(PREPARED_JOB_SETTINGS.tasks))
print(
    "Notebook target:",
    PREPARED_JOB_SETTINGS.tasks[0]
    .notebook_task.notebook_path
)

print("\nEXECUTION SAFETY")
print("Active schedule configured: False")
print("Trigger configured: False")
print("Continuous execution configured: False")

print("\nVALIDATION")
print("SDK compatibility: PASS")
print("Job settings structure: PASS")
print("Notebook target: PASS")
print("Single-run concurrency: PASS")

print("\nSAFETY")
print("Job creation API called: False")
print("Job created: False")
print("Job executed: False")
print("Notebook 15C executed: False")
print("Delta tables modified: False")

print("\nJOB SETTINGS VALIDATION: PASS")


# COMMAND ----------


# Step 15.80W: Create the guarded Databricks Job
# Creates job definition ONLY — does not run inference

from databricks.sdk import WorkspaceClient

print("=" * 65)
print("GUARDED DATABRICKS JOB CREATION")
print("=" * 65)

assert "PREPARED_JOB_SETTINGS" in globals()

settings = PREPARED_JOB_SETTINGS

expected_path = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

assert settings.max_concurrent_runs == 1
assert len(settings.tasks) == 1

assert (
    settings.tasks[0].notebook_task.notebook_path
    == expected_path
)

assert settings.schedule is None
assert settings.trigger is None
assert settings.continuous is None

w = WorkspaceClient()

# Check whether a job with this name already exists.
existing = [
    job
    for job in w.jobs.list()
    if job.settings is not None
    and job.settings.name == settings.name
]

if existing:
    print("\nJOB CREATION BLOCKED")
    print("A job with this name already exists.")

    for job in existing:
        print("Existing Job ID:", job.job_id)

    print("No new job created.")

else:
    # This API call creates the job definition.
    # It does NOT execute the job.
    created = w.jobs.create(
        **settings.as_dict()
    )

    print("\nJOB CREATED SUCCESSFULLY")
    print("Job ID:", created.job_id)
    print("Job name:", settings.name)
    print(
        "Notebook target:",
        settings.tasks[0].notebook_task.notebook_path
    )

print("\nEXECUTION SAFETY")
print("Job run API called: False")
print("Notebook 15C executed by this cell: False")
print("Batch inference executed by this cell: False")
print("Delta writes performed by this cell: False")

print("\nJOB CREATION STEP: COMPLETE")


# COMMAND ----------


# Step 15.80W.1: Verify job creation status
# Read-only — no job creation, execution, or Delta writes

from databricks.sdk import WorkspaceClient

print("=" * 65)
print("DATABRICKS JOB CREATION STATUS CHECK")
print("=" * 65)

w = WorkspaceClient()

expected_name = (
    "Telco_Churn_Snapshot_Protected_Inference"
)

expected_path = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

matching_jobs = []

for job in w.jobs.list(expand_tasks=True):

    if job.settings is None:
        continue

    if job.settings.name != expected_name:
        continue

    matching_jobs.append(job)

print("\nMATCHING JOBS")
print("Count:", len(matching_jobs))

for job in matching_jobs:

    print("\nJob ID:", job.job_id)
    print("Job name:", job.settings.name)

    print(
        "Maximum concurrent runs:",
        job.settings.max_concurrent_runs
    )

    for task in job.settings.tasks or []:

        notebook_path = (
            task.notebook_task.notebook_path
            if task.notebook_task is not None
            else None
        )

        print("Task:", task.task_key)
        print("Notebook:", notebook_path)
        print(
            "Expected notebook:",
            notebook_path == expected_path
        )

    print("Schedule:", job.settings.schedule)
    print("Trigger:", job.settings.trigger)
    print("Continuous:", job.settings.continuous)

print("\nASSESSMENT")

if len(matching_jobs) == 0:
    print("No matching job found.")
    print("Creation retry may be prepared after SDK correction.")

elif len(matching_jobs) == 1:
    print("One matching job exists.")
    print("Do not retry creation until its settings are reviewed.")

else:
    print("Multiple matching jobs exist.")
    print("Do not retry creation.")

print("\nSAFETY")
print("Job creation API called by this cell: False")
print("Job run API called by this cell: False")
print("Notebook 15C executed by this cell: False")
print("Delta writes performed by this cell: False")

print("\nJOB CREATION STATUS CHECK: COMPLETE")


# COMMAND ----------


# Step 15.80W.2: Prepare typed SDK job creation arguments
# In-memory only — no job creation or inference

from databricks.sdk.service.jobs import JobSettings

print("=" * 65)
print("TYPED DATABRICKS JOB REQUEST VALIDATION")
print("=" * 65)

assert "CORRECTED_JOB_DEFINITION" in globals()

job_config = CORRECTED_JOB_DEFINITION

expected_path = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

# Deserialize all nested configuration objects.
TYPED_JOB_SETTINGS = JobSettings.from_dict(
    job_config
)

assert TYPED_JOB_SETTINGS.max_concurrent_runs == 1
assert len(TYPED_JOB_SETTINGS.tasks) == 1

assert (
    TYPED_JOB_SETTINGS.tasks[0]
    .notebook_task.notebook_path
    == expected_path
)

assert TYPED_JOB_SETTINGS.schedule is None
assert TYPED_JOB_SETTINGS.trigger is None
assert TYPED_JOB_SETTINGS.continuous is None

# Prepare typed keyword arguments for the Jobs API.
# Do not convert nested objects back into dictionaries.
TYPED_JOB_CREATE_ARGS = {
    "name": TYPED_JOB_SETTINGS.name,
    "tasks": TYPED_JOB_SETTINGS.tasks,
    "max_concurrent_runs":
        TYPED_JOB_SETTINGS.max_concurrent_runs,
    "queue": TYPED_JOB_SETTINGS.queue,
    "timeout_seconds":
        TYPED_JOB_SETTINGS.timeout_seconds,
    "environments":
        TYPED_JOB_SETTINGS.environments
}

# Remove only absent optional fields.
TYPED_JOB_CREATE_ARGS = {
    key: value
    for key, value in TYPED_JOB_CREATE_ARGS.items()
    if value is not None
}

print("\nJOB CONFIGURATION")
print("Name:", TYPED_JOB_SETTINGS.name)
print("Task count:", len(TYPED_JOB_SETTINGS.tasks))
print(
    "Notebook:",
    TYPED_JOB_SETTINGS.tasks[0]
    .notebook_task.notebook_path
)
print(
    "Maximum concurrent runs:",
    TYPED_JOB_SETTINGS.max_concurrent_runs
)

print("\nSDK OBJECT TYPES")
print("Task:", type(TYPED_JOB_SETTINGS.tasks[0]).__name__)
print("Queue:", type(TYPED_JOB_SETTINGS.queue).__name__)

for environment in TYPED_JOB_SETTINGS.environments or []:
    print("Environment:", type(environment).__name__)
    assert hasattr(environment, "as_dict")

assert all(
    hasattr(task, "as_dict")
    for task in TYPED_JOB_SETTINGS.tasks
)

assert hasattr(TYPED_JOB_SETTINGS.queue, "as_dict")

print("\nVALIDATION")
print("Typed task objects: PASS")
print("Typed environment objects: PASS")
print("Typed queue configuration: PASS")
print("Notebook target: PASS")
print("No automatic execution configuration: PASS")

print("\nSAFETY")
print("Job creation API called: False")
print("Job executed: False")
print("Notebook 15C executed: False")
print("Registry modified: False")
print("Approved predictions modified: False")

print("\nTYPED JOB REQUEST: READY FOR REVIEW")


# COMMAND ----------


# Step 15.80W.3: Create guarded Databricks Job
# Creates the job definition — DOES NOT RUN IT

from databricks.sdk import WorkspaceClient

print("=" * 65)
print("GUARDED DATABRICKS JOB CREATION")
print("=" * 65)

assert "TYPED_JOB_CREATE_ARGS" in globals()

args = TYPED_JOB_CREATE_ARGS

expected_name = "Telco_Churn_Snapshot_Protected_Inference"

expected_path = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

assert args["name"] == expected_name
assert args["max_concurrent_runs"] == 1
assert len(args["tasks"]) == 1

assert (
    args["tasks"][0].notebook_task.notebook_path
    == expected_path
)

assert "schedule" not in args
assert "trigger" not in args
assert "continuous" not in args

w = WorkspaceClient()

existing_jobs = [
    job
    for job in w.jobs.list()
    if job.settings is not None
    and job.settings.name == expected_name
]

if existing_jobs:

    print("\nCREATION BLOCKED")
    print("A job with this name already exists.")

    for job in existing_jobs:
        print("Existing Job ID:", job.job_id)

    print("No new job created.")

else:

    # Create the job definition only.
    # This does not invoke run_now or submit.

    created_job = w.jobs.create(**args)

    print("\nJOB CREATED SUCCESSFULLY")
    print("Job ID:", created_job.job_id)
    print("Job name:", expected_name)
    print("Notebook target:", expected_path)

print("\nEXECUTION SAFETY")
print("Job run API called by this cell: False")
print("Notebook 15C executed by this cell: False")
print("Batch inference executed by this cell: False")
print("Delta writes performed by this cell: False")

print("\nJOB CREATION STEP: COMPLETE")


# COMMAND ----------


# Step 15.80X: Verify created Databricks Job
# Read-only — no job execution or Delta writes

from databricks.sdk import WorkspaceClient

print("=" * 65)
print("CREATED DATABRICKS JOB VERIFICATION")
print("=" * 65)

JOB_ID = 752436180383454

expected_path = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

w = WorkspaceClient()

job = w.jobs.get(job_id=JOB_ID)
settings = job.settings

assert settings is not None
assert settings.name == (
    "Telco_Churn_Snapshot_Protected_Inference"
)
assert settings.max_concurrent_runs == 1
assert len(settings.tasks or []) == 1

task = settings.tasks[0]

assert task.notebook_task is not None
assert task.notebook_task.notebook_path == expected_path

assert settings.queue is not None
assert settings.queue.enabled is True

assert settings.schedule is None
assert settings.trigger is None
assert settings.continuous is None

print("\nSAVED JOB CONFIGURATION")
print("Job ID:", JOB_ID)
print("Job name:", settings.name)
print("Task key:", task.task_key)
print("Notebook:", task.notebook_task.notebook_path)
print("Maximum concurrent runs:", settings.max_concurrent_runs)
print("Queue enabled:", settings.queue.enabled)

print("\nAUTOMATIC EXECUTION")
print("Schedule:", settings.schedule)
print("Trigger:", settings.trigger)
print("Continuous:", settings.continuous)

print("\nRECENT JOB RUNS")

runs = list(
    w.jobs.list_runs(
        job_id=JOB_ID,
        limit=10
    )
)

print("Runs found:", len(runs))

for run in runs:
    print(
        "Run ID:",
        run.run_id,
        "| State:",
        run.state
    )

print("\nVERIFICATION")

if len(runs) == 0:
    print("Job has no recorded runs: PASS")
else:
    print("Job has recorded runs: REVIEW REQUIRED")

print("\nSAFETY")
print("Job creation API called by this cell: False")
print("Job execution API called by this cell: False")
print("Notebook 15C executed by this cell: False")
print("Delta writes performed by this cell: False")

print("\nCREATED JOB VERIFICATION: COMPLETE")


# COMMAND ----------


# Step 15.80Y: Final pre-run Delta safety check
# READ-ONLY — does not execute the Databricks Job

from pyspark.sql import functions as F

print("=" * 65)
print("FINAL PRE-RUN DELTA SAFETY CHECK")
print("=" * 65)

approved_table = (
    "workspace.default.telco_15_approved_predictions"
)

dev_table = (
    "workspace.default.telco_15_job_dev_predictions"
)

registry_table = (
    "workspace.default.telco_15_job_batch_registry"
)

batch_id = "d972dd16-029d-489f-a406-3c6c6e1e6265"

# 1. Protected table baseline
approved_count = spark.table(approved_table).count()

approved_version = (
    spark.sql(f"DESCRIBE HISTORY {approved_table}")
    .select("version")
    .first()["version"]
)

assert approved_count == 1409
assert approved_version == 3

print("\nPROTECTED APPROVED PREDICTIONS")
print("Rows:", approved_count)
print("Delta version:", approved_version)
print("Baseline: PASS")

# 2. Development output
dev_count = spark.table(dev_table).count()

dev_batch_matches = (
    spark.table(dev_table)
    .filter(F.col("batch_id") == batch_id)
    .count()
)

print("\nDEVELOPMENT PREDICTIONS")
print("Rows:", dev_count)
print("Controlled batch matches:", dev_batch_matches)

assert dev_batch_matches == 0

# 3. Registry
registry_count = spark.table(registry_table).count()

registry_batch_matches = (
    spark.table(registry_table)
    .filter(F.col("batch_id") == batch_id)
    .count()
)

print("\nBATCH REGISTRY")
print("Rows:", registry_count)
print("Controlled batch matches:", registry_batch_matches)

assert registry_batch_matches == 0

print("\nFINAL ASSESSMENT")
print("Approved predictions baseline: PASS")
print("Controlled batch ID unused: PASS")
print("Development destination check: PASS")
print("Registry check: PASS")

print("\nSAFETY")
print("Databricks Job executed: False")
print("Notebook 15C executed by this cell: False")
print("Delta writes performed by this cell: False")

print("\nFINAL PRE-RUN SAFETY CHECK: PASS")


# COMMAND ----------


# Step 15.80Z.1: Submit ONE authorized development run
# WARNING: This starts batch inference and may write
# to the development predictions table and batch registry.
# It must NOT write to the approved predictions table.

from databricks.sdk import WorkspaceClient

print("=" * 65)
print("FIRST CONTROLLED DEVELOPMENT JOB EXECUTION")
print("=" * 65)

JOB_ID = 752436180383454

EXPECTED_NOTEBOOK = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

w = WorkspaceClient()

# Verify saved job configuration.
job = w.jobs.get(job_id=JOB_ID)
settings = job.settings

assert settings is not None
assert len(settings.tasks or []) == 1
assert settings.max_concurrent_runs == 1
assert settings.tasks[0].notebook_task is not None
assert (
    settings.tasks[0].notebook_task.notebook_path
    == EXPECTED_NOTEBOOK
)

assert settings.schedule is None
assert settings.trigger is None
assert settings.continuous is None

# This is the first run: prevent accidental resubmission.
existing_runs = list(
    w.jobs.list_runs(job_id=JOB_ID, limit=10)
)

assert len(existing_runs) == 0, (
    "Execution blocked: this job already has a run. "
    "Review its history before proceeding."
)

print("\nPRE-SUBMISSION VALIDATION")
print("Job ID:", JOB_ID)
print("Notebook:", EXPECTED_NOTEBOOK)
print("Existing runs:", len(existing_runs))
print("Maximum concurrent runs:", settings.max_concurrent_runs)
print("Configuration: PASS")

# Submit exactly one run.
# This returns after submission; it does not wait
# for the batch inference to finish.
submitted_run = w.jobs.run_now(
    job_id=JOB_ID
)

print("\nRUN SUBMITTED")
print("Run ID:", submitted_run.run_id)
print("Job ID:", JOB_ID)

print("\nIMPORTANT")
print("Execution has been requested.")
print("Successful inference is NOT yet confirmed.")
print("Do not submit another run.")
print("Do not manually execute Notebook 15C.")

print("\nSTEP 15.80Z.1: SUBMISSION COMPLETE")


# COMMAND ----------


# Step 15.80Z.2: Inspect controlled job execution
# READ-ONLY — no new job run or Delta writes

from databricks.sdk import WorkspaceClient

print("=" * 65)
print("CONTROLLED DATABRICKS JOB RUN STATUS")
print("=" * 65)

JOB_ID = 752436180383454
RUN_ID = 767116018436128

w = WorkspaceClient()

run = w.jobs.get_run(run_id=RUN_ID)

assert run.job_id == JOB_ID, (
    "Run does not belong to the expected job."
)

print("\nRUN IDENTIFICATION")
print("Job ID:", run.job_id)
print("Run ID:", run.run_id)
print("Run name:", run.run_name)

print("\nEXECUTION STATE")

state = run.state

life_cycle = (
    state.life_cycle_state.value
    if state and state.life_cycle_state
    else "UNKNOWN"
)

result = (
    state.result_state.value
    if state and state.result_state
    else "NOT_AVAILABLE"
)

message = (
    state.state_message
    if state
    else None
)

print("Lifecycle state:", life_cycle)
print("Result state:", result)
print("State message:", message)

print("\nTASK INFORMATION")

for task in run.tasks or []:
    print("Task key:", task.task_key)
    print("Task run ID:", task.run_id)

    if task.state:
        print(
            "Task lifecycle:",
            task.state.life_cycle_state
        )
        print(
            "Task result:",
            task.state.result_state
        )
        print(
            "Task message:",
            task.state.state_message
        )

print("\nASSESSMENT")

if life_cycle == "TERMINATED" and result == "SUCCESS":
    print("JOB EXECUTION: SUCCESS")
    print("Next step: Validate Delta outputs.")

elif life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
    print("JOB EXECUTION: REVIEW REQUIRED")
    print("Do not submit another run.")

else:
    print("JOB EXECUTION: NOT YET COMPLETE")
    print("Do not submit another run.")

print("\nSAFETY")
print("New job run submitted: False")
print("Notebook executed by this cell: False")
print("Delta writes performed by this cell: False")

print("\nRUN STATUS CHECK: COMPLETE")


# COMMAND ----------


# Step 15.80Z.3: Wait for existing Databricks Job run
# READ-ONLY — does not submit another run

from databricks.sdk import WorkspaceClient
from datetime import timedelta

print("=" * 65)
print("WAIT FOR CONTROLLED DATABRICKS JOB COMPLETION")
print("=" * 65)

JOB_ID = 752436180383454
RUN_ID = 767116018436128

w = WorkspaceClient()

print("\nMonitoring existing run:", RUN_ID)
print("No new execution will be submitted.")

try:
    # Wait up to 10 minutes for the existing run.
    completed_run = w.jobs.wait_get_run_job_terminated_or_skipped(
        run_id=RUN_ID,
        timeout=timedelta(minutes=10)
    )

    assert completed_run.job_id == JOB_ID

    state = completed_run.state

    lifecycle = (
        state.life_cycle_state.value
        if state and state.life_cycle_state
        else "UNKNOWN"
    )

    result = (
        state.result_state.value
        if state and state.result_state
        else "NOT_AVAILABLE"
    )

    print("\nFINAL RUN STATE")
    print("Lifecycle:", lifecycle)
    print("Result:", result)
    print("Message:", state.state_message)

    if lifecycle == "TERMINATED" and result == "SUCCESS":
        print("\nJOB EXECUTION: SUCCESS")
        print("Next step: Validate registry and Delta outputs.")
    else:
        print("\nJOB EXECUTION: REVIEW REQUIRED")
        print("Do not submit another run.")

except TimeoutError:
    print("\nWAIT TIMED OUT")
    print("The job may still be running.")
    print("Do not submit another run.")

except Exception as exc:
    print("\nMONITORING ERROR")
    print("Error type:", type(exc).__name__)
    print("Message:", str(exc))
    print("The run outcome has not been established.")
    print("Do not submit another run.")

print("\nSAFETY")
print("New job submitted: False")
print("Delta writes performed by this cell: False")

print("\nSTEP 15.80Z.3: COMPLETE")


# COMMAND ----------


# Step 15.80Z.4: Validate controlled batch inference outputs
# READ-ONLY — no job execution or Delta writes

from pyspark.sql import functions as F

print("=" * 65)
print("CONTROLLED BATCH INFERENCE OUTPUT VALIDATION")
print("=" * 65)

BATCH_ID = "d972dd16-029d-489f-a406-3c6c6e1e6265"

APPROVED_TABLE = (
    "workspace.default.telco_15_approved_predictions"
)

DEV_TABLE = (
    "workspace.default.telco_15_job_dev_predictions"
)

REGISTRY_TABLE = (
    "workspace.default.telco_15_job_batch_registry"
)

# 1. Verify approved predictions remain unchanged.
approved_count = spark.table(APPROVED_TABLE).count()

approved_version = (
    spark.sql(f"DESCRIBE HISTORY {APPROVED_TABLE}")
    .select("version")
    .first()["version"]
)

approved_pass = (
    approved_count == 1409
    and approved_version == 3
)

print("\n1. PROTECTED APPROVED PREDICTIONS")
print("Row count:", approved_count)
print("Delta version:", approved_version)
print("Protection check:", "PASS" if approved_pass else "FAIL")

# 2. Verify development predictions.
dev_df = spark.table(DEV_TABLE)

dev_count = dev_df.count()

batch_predictions = dev_df.filter(
    F.col("batch_id") == BATCH_ID
)

batch_prediction_count = batch_predictions.count()

distinct_predictions = (
    batch_predictions.select("customerID").distinct().count()
)

dev_pass = (
    dev_count == 5636
    and batch_prediction_count == 1409
    and distinct_predictions == 1409
)

print("\n2. DEVELOPMENT PREDICTIONS")
print("Total rows:", dev_count)
print("Controlled batch rows:", batch_prediction_count)
print("Distinct customers:", distinct_predictions)
print("Development output check:", "PASS" if dev_pass else "FAIL")

# 3. Verify batch registry.
registry_df = spark.table(REGISTRY_TABLE)

registry_count = registry_df.count()

batch_registry_count = registry_df.filter(
    F.col("batch_id") == BATCH_ID
).count()

registry_pass = (
    registry_count == 4
    and batch_registry_count == 1
)

print("\n3. BATCH REGISTRY")
print("Total registry rows:", registry_count)
print("Controlled batch entries:", batch_registry_count)
print("Registry check:", "PASS" if registry_pass else "FAIL")

# 4. Final assessment.
print("\nFINAL VALIDATION")

print("Approved table:", "PASS" if approved_pass else "FAIL")
print("Development predictions:", "PASS" if dev_pass else "FAIL")
print("Batch registry:", "PASS" if registry_pass else "FAIL")

if approved_pass and dev_pass and registry_pass:
    print("\nCONTROLLED DEVELOPMENT TEST: PASS")
else:
    print("\nCONTROLLED DEVELOPMENT TEST: REVIEW REQUIRED")
    print("Do not submit another job run.")

print("\nSAFETY")
print("New Databricks Job submitted: False")
print("Notebook 15C executed by this cell: False")
print("Delta writes performed by this cell: False")

print("\nSTEP 15.80Z.4: COMPLETE")


# COMMAND ----------


# Step 15.80Z.5: Inspect controlled batch prediction quality
# READ-ONLY — no Databricks Job execution or Delta writes

from pyspark.sql import functions as F

print("=" * 65)
print("CONTROLLED BATCH PREDICTION QUALITY VALIDATION")
print("=" * 65)

BATCH_ID = "d972dd16-029d-489f-a406-3c6c6e1e6265"

DEV_TABLE = (
    "workspace.default.telco_15_job_dev_predictions"
)

batch_df = spark.table(DEV_TABLE).filter(
    F.col("batch_id") == BATCH_ID
)

print("\nBATCH STRUCTURE")
print("Rows:", batch_df.count())
print("Columns:", len(batch_df.columns))
print("Column names:", batch_df.columns)

print("\nNULL VALUE CHECK")

null_counts = batch_df.select([
    F.sum(
        F.when(F.col(c).isNull(), 1).otherwise(0)
    ).alias(c)
    for c in batch_df.columns
]).first().asDict()

for column, count in null_counts.items():
    print(f"{column}: {count}")

print("\nDUPLICATE CUSTOMER CHECK")

duplicates = (
    batch_df.groupBy("customerID")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

print("Duplicate customer IDs:", duplicates)

print("\nBATCH ID CONSISTENCY")

distinct_batch_ids = (
    batch_df.select("batch_id").distinct().count()
)

print("Distinct batch IDs:", distinct_batch_ids)

print("\nASSESSMENT")

assert batch_df.count() == 1409
assert duplicates == 0
assert distinct_batch_ids == 1

print("Expected prediction count: PASS")
print("Customer uniqueness: PASS")
print("Batch identity consistency: PASS")

if all(value == 0 for value in null_counts.values()):
    print("Null value check: PASS")
else:
    print("Null value check: REVIEW REQUIRED")
    print("Some columns contain null values.")

print("\nSAFETY")
print("Job executed by this cell: False")
print("Delta writes performed by this cell: False")

print("\nSTEP 15.80Z.5: COMPLETE")


# COMMAND ----------


# Step 15.81A: Inspect batch retry safety
# READ-ONLY — no job execution or Delta writes

from pyspark.sql import functions as F

print("=" * 65)
print("BATCH RETRY SAFETY INSPECTION")
print("=" * 65)

BATCH_ID = "d972dd16-029d-489f-a406-3c6c6e1e6265"

REGISTRY_TABLE = (
    "workspace.default.telco_15_job_batch_registry"
)

DEV_TABLE = (
    "workspace.default.telco_15_job_dev_predictions"
)

APPROVED_TABLE = (
    "workspace.default.telco_15_approved_predictions"
)

registry = spark.table(REGISTRY_TABLE)
predictions = spark.table(DEV_TABLE)

registered_batch = registry.filter(
    F.col("batch_id") == BATCH_ID
)

published_batch = predictions.filter(
    F.col("batch_id") == BATCH_ID
)

print("\nREGISTRY")
print("Matching entries:", registered_batch.count())
registered_batch.show(truncate=False)

print("\nPUBLISHED PREDICTIONS")
print("Matching predictions:", published_batch.count())

print("\nCUSTOMER UNIQUENESS")
duplicate_count = (
    published_batch
    .groupBy("customerID")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

print("Duplicate customers:", duplicate_count)

print("\nPROTECTED TABLE")
approved_count = spark.table(APPROVED_TABLE).count()

approved_version = (
    spark.sql(f"DESCRIBE HISTORY {APPROVED_TABLE}")
    .select("version")
    .first()["version"]
)

print("Approved rows:", approved_count)
print("Approved Delta version:", approved_version)

assert registered_batch.count() == 1
assert published_batch.count() == 1409
assert duplicate_count == 0
assert approved_count == 1409
assert approved_version == 3

print("\nASSESSMENT")
print("Existing batch registration: PASS")
print("Existing published predictions: PASS")
print("Customer uniqueness: PASS")
print("Protected table baseline: PASS")

print("\nNOTE")
print("This checks the current state only.")
print("It does not prove that concurrent retries are safe.")

print("\nSAFETY")
print("New job run submitted: False")
print("Delta writes performed: False")

print("\nSTEP 15.81A: COMPLETE")


# COMMAND ----------


# Step 15.81B: Inspect retry configuration and batch integrity
# READ-ONLY — no job execution or Delta writes

from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F

print("=" * 65)
print("JOB RETRY CONFIGURATION AND BATCH INTEGRITY")
print("=" * 65)

JOB_ID = 752436180383454

REGISTRY_TABLE = (
    "workspace.default.telco_15_job_batch_registry"
)

DEV_TABLE = (
    "workspace.default.telco_15_job_dev_predictions"
)

w = WorkspaceClient()

job = w.jobs.get(job_id=JOB_ID)
settings = job.settings

assert settings is not None
assert len(settings.tasks or []) == 1

task = settings.tasks[0]

print("\n1. JOB RETRY CONFIGURATION")
print("Job ID:", JOB_ID)
print("Task:", task.task_key)
print("Maximum retries:", task.max_retries)
print("Minimum retry interval:", task.min_retry_interval_millis)
print("Retry on timeout:", task.retry_on_timeout)
print("Task timeout:", task.timeout_seconds)
print("Maximum concurrent runs:", settings.max_concurrent_runs)

print("\n2. REGISTRY BATCH UNIQUENESS")

registry = spark.table(REGISTRY_TABLE)

duplicate_registry_ids = (
    registry
    .groupBy("batch_id")
    .count()
    .filter(F.col("count") > 1)
)

registry_duplicates = duplicate_registry_ids.count()

print("Registry rows:", registry.count())
print("Duplicate batch IDs:", registry_duplicates)

if registry_duplicates:
    duplicate_registry_ids.show(truncate=False)

print("\n3. DEVELOPMENT PREDICTION UNIQUENESS")

predictions = spark.table(DEV_TABLE)

duplicate_predictions = (
    predictions
    .groupBy("batch_id", "customerID")
    .count()
    .filter(F.col("count") > 1)
)

prediction_duplicates = duplicate_predictions.count()

print("Development prediction rows:", predictions.count())
print(
    "Duplicate batch/customer combinations:",
    prediction_duplicates
)

if prediction_duplicates:
    duplicate_predictions.show(truncate=False)

print("\n4. PROTECTED TABLE BASELINE")

approved_table = (
    "workspace.default.telco_15_approved_predictions"
)

approved_count = spark.table(approved_table).count()

approved_version = (
    spark.sql(f"DESCRIBE HISTORY {approved_table}")
    .select("version")
    .first()["version"]
)

print("Approved rows:", approved_count)
print("Approved version:", approved_version)

assert approved_count == 1409
assert approved_version == 3

print("\nASSESSMENT")
print(
    "Registry uniqueness:",
    "PASS" if registry_duplicates == 0 else "REVIEW REQUIRED"
)
print(
    "Prediction uniqueness:",
    "PASS" if prediction_duplicates == 0 else "REVIEW REQUIRED"
)
print("Protected table baseline: PASS")

print("\nNOTE")
print("These checks inspect existing data only.")
print("They do not prove retry safety under concurrency.")

print("\nSAFETY")
print("New job submitted: False")
print("Notebook 15C executed by this cell: False")
print("Delta writes performed by this cell: False")

print("\nSTEP 15.81B: COMPLETE")


# COMMAND ----------


# Step 15.81C: Inspect deployed notebook retry logic
# READ-ONLY — source inspection only

import base64
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

print("=" * 65)
print("DEPLOYED NOTEBOOK RETRY LOGIC INSPECTION")
print("=" * 65)

NOTEBOOK_PATH = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

w = WorkspaceClient()

exported = w.workspace.export(
    path=NOTEBOOK_PATH,
    format=ExportFormat.SOURCE
)

assert exported.content is not None

source = base64.b64decode(
    exported.content
).decode("utf-8")

lines = source.splitlines()

function_names = [
    "register_batch_identity",
    "publish_batch_predictions",
    "execute_snapshot_protected_batch_job"
]

for function_name in function_names:

    print("\n" + "=" * 65)
    print("FUNCTION:", function_name)
    print("=" * 65)

    matches = [
        i for i, line in enumerate(lines)
        if line.lstrip().startswith(
            f"def {function_name}("
        )
    ]

    if not matches:
        print("Function not found.")
        continue

    start = matches[0]

    # Show up to 90 lines from the function definition.
    end = min(start + 90, len(lines))

    for i in range(start, end):
        print(f"{i + 1:04d}: {lines[i]}")

print("\nSAFETY")
print("Job executed: False")
print("Notebook 15C modified: False")
print("Delta writes performed: False")

print("\nSTEP 15.81C: SOURCE INSPECTION COMPLETE")


# COMMAND ----------


# Step 15.81D: Locate publisher and Delta write operations
# READ-ONLY — no notebook execution or Delta writes

import base64
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

print("=" * 65)
print("DEPLOYED NOTEBOOK WRITE-PATH INSPECTION")
print("=" * 65)

NOTEBOOK_PATH = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

w = WorkspaceClient()

exported = w.workspace.export(
    path=NOTEBOOK_PATH,
    format=ExportFormat.SOURCE
)

source = base64.b64decode(
    exported.content
).decode("utf-8")

lines = source.splitlines()

print("\n1. FUNCTION DEFINITIONS")

for i, line in enumerate(lines):
    if line.lstrip().startswith("def "):
        print(f"{i + 1:04d}: {line.strip()}")

print("\n2. DELTA WRITE OPERATIONS")

write_patterns = [
    ".saveAsTable(",
    ".insertInto(",
    ".mode(",
    "MERGE INTO",
    "DeltaTable.forName",
    ".merge(",
    ".execute()"
]

for i, line in enumerate(lines):
    if any(pattern in line for pattern in write_patterns):
        print(f"\nLINE {i + 1}")

        start = max(0, i - 5)
        end = min(len(lines), i + 7)

        for j in range(start, end):
            print(f"{j + 1:04d}: {lines[j]}")

print("\n3. REGISTRATION AND PUBLICATION CALLS")

for i, line in enumerate(lines):
    if (
        "register_batch_identity(" in line
        or "publish_" in line
        or "registration_status" in line
    ):
        print(f"{i + 1:04d}: {line.strip()}")

print("\nSAFETY")
print("Job executed: False")
print("Notebook modified: False")
print("Delta writes performed: False")

print("\nSTEP 15.81D: COMPLETE")


# COMMAND ----------


# Step 15.81E: Inspect strict publication and orchestration
# READ-ONLY — no execution or Delta writes

import base64
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

print("=" * 65)
print("STRICT PUBLICATION AND RETRY FLOW INSPECTION")
print("=" * 65)

NOTEBOOK_PATH = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

w = WorkspaceClient()

exported = w.workspace.export(
    path=NOTEBOOK_PATH,
    format=ExportFormat.SOURCE
)

source = base64.b64decode(
    exported.content
).decode("utf-8")

lines = source.splitlines()

# Display only the relevant code sections.
sections = [
    ("REGISTRY RETRY RETURN", 260, 325),
    ("PUBLISHER", 883, 1030),
    ("ORCHESTRATOR", 1118, 1235),
    ("STRICT PUBLISHER WRAPPER", 1252, 1325)
]

for title, start, end in sections:

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    for line_number in range(start, min(end, len(lines)) + 1):
        print(
            f"{line_number:04d}: "
            f"{lines[line_number - 1]}"
        )

print("\nSAFETY")
print("Databricks Job executed: False")
print("Notebook 15C modified: False")
print("Delta writes performed: False")

print("\nSTEP 15.81E: COMPLETE")


# COMMAND ----------


# Step 15.81F: Inspect strict publication integrity logic
# READ-ONLY — no job execution or Delta writes

import base64
import hashlib

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat

print("=" * 65)
print("STRICT PUBLICATION INTEGRITY VALIDATOR INSPECTION")
print("=" * 65)

NOTEBOOK_PATH = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

EXPECTED_HASH = (
    "3979f576572b7578962b0abc0d1a994c0ff3847"
    "b37b9b6208714670a795dc2e4"
)

w = WorkspaceClient()

exported = w.workspace.export(
    path=NOTEBOOK_PATH,
    format=ExportFormat.SOURCE
)

assert exported.content is not None

source = base64.b64decode(
    exported.content
).decode("utf-8")

source_hash = hashlib.sha256(
    source.encode("utf-8")
).hexdigest()

print("\nSOURCE INTEGRITY")
print("Notebook:", NOTEBOOK_PATH)
print("SHA256:", source_hash)

assert source_hash == EXPECTED_HASH, (
    "Notebook source changed. Review before proceeding."
)

print("Source integrity: PASS")

lines = source.splitlines()

print("\nSTRICT INTEGRITY VALIDATOR")

start = next(
    i for i, line in enumerate(lines)
    if line.lstrip().startswith(
        "def validate_strict_publication_integrity("
    )
)

end = next(
    (
        i for i in range(start + 1, len(lines))
        if lines[i].startswith("def ")
        or lines[i].startswith("# COMMAND ----------")
    ),
    len(lines)
)

for i in range(start, end):
    print(f"{i + 1:04d}: {lines[i]}")

print("\nSAFETY")
print("Databricks Job executed: False")
print("Notebook 15C modified: False")
print("Delta writes performed: False")

print("\nSTEP 15.81F: COMPLETE")


# COMMAND ----------


# Step 15.81G: Read-only retry recovery assessment

from pyspark.sql import functions as F

print("=" * 65)
print("BATCH RETRY RECOVERY ASSESSMENT")
print("=" * 65)

BATCH_ID = "d972dd16-029d-489f-a406-3c6c6e1e6265"

REGISTRY_TABLE = (
    "workspace.default.telco_15_job_batch_registry"
)

DEV_TABLE = (
    "workspace.default.telco_15_job_dev_predictions"
)

APPROVED_TABLE = (
    "workspace.default.telco_15_approved_predictions"
)

registry_rows = (
    spark.table(REGISTRY_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
    .collect()
)

assert len(registry_rows) == 1

registered = registry_rows[0]

published = (
    spark.table(DEV_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

published_count = published.count()

unique_customers = (
    published.select("customerID").distinct().count()
)

registered_count = registered["record_count"]

print("\nREGISTERED BATCH")
print("Expected records:", registered_count)
print("Source:", registered["source_table"])
print("Model:", registered["model_name"])
print("Model version:", registered["model_version"])

print("\nPUBLISHED BATCH")
print("Published records:", published_count)
print("Unique customers:", unique_customers)

print("\nRETRY STATE")

if published_count == 0:
    retry_state = "REGISTERED_NOT_PUBLISHED"
elif (
    published_count == registered_count
    and unique_customers == published_count
):
    retry_state = "COMPLETE_CANDIDATE"
else:
    retry_state = "INCOMPLETE_OR_DUPLICATED"

print("Current state:", retry_state)

print("\nRECOVERY INTERPRETATION")

if retry_state == "COMPLETE_CANDIDATE":
    print(
        "Counts and customer uniqueness are consistent "
        "with a completed batch."
    )
    print(
        "Prediction-value equivalence still requires "
        "comparison with regenerated predictions."
    )
elif retry_state == "REGISTERED_NOT_PUBLISHED":
    print(
        "Registry exists but no predictions are published."
    )
else:
    print(
        "Published records are incomplete or duplicated."
    )
    print(
        "Current strict validator may block recovery."
    )

print("\nPROTECTED TABLE")

approved_count = spark.table(APPROVED_TABLE).count()

approved_version = (
    spark.sql(f"DESCRIBE HISTORY {APPROVED_TABLE}")
    .select("version")
    .first()["version"]
)

print("Approved rows:", approved_count)
print("Approved version:", approved_version)

assert approved_count == 1409
assert approved_version == 3

print("\nSAFETY")
print("Job executed: False")
print("Inference executed: False")
print("Delta writes performed: False")

print("\nSTEP 15.81G: COMPLETE")


# COMMAND ----------


# Step 15.81H: Verify published customer population
# READ-ONLY — no inference, job execution, or Delta writes

from pyspark.sql import functions as F

print("=" * 65)
print("PINNED SOURCE AND PUBLISHED CUSTOMER VERIFICATION")
print("=" * 65)

BATCH_ID = "d972dd16-029d-489f-a406-3c6c6e1e6265"

SOURCE_TABLE = "workspace.default.telco_test_raw"
SOURCE_VERSION = 2

DEV_TABLE = (
    "workspace.default.telco_15_job_dev_predictions"
)

APPROVED_TABLE = (
    "workspace.default.telco_15_approved_predictions"
)

# Read the exact source snapshot used by the job.
source_df = (
    spark.read
    .format("delta")
    .option("versionAsOf", SOURCE_VERSION)
    .table(SOURCE_TABLE)
)

published_df = (
    spark.table(DEV_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

source_ids = source_df.select("customerID")
published_ids = published_df.select("customerID")

print("\n1. RECORD COUNTS")
print("Source records:", source_df.count())
print("Published records:", published_df.count())

print("\n2. CUSTOMER ID VALIDATION")

source_duplicates = (
    source_ids.groupBy("customerID")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

published_duplicates = (
    published_ids.groupBy("customerID")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

missing_published = (
    source_ids.join(
        published_ids,
        "customerID",
        "left_anti"
    ).count()
)

unexpected_published = (
    published_ids.join(
        source_ids,
        "customerID",
        "left_anti"
    ).count()
)

print("Source duplicate IDs:", source_duplicates)
print("Published duplicate IDs:", published_duplicates)
print("Missing published customers:", missing_published)
print("Unexpected published customers:", unexpected_published)

population_pass = (
    source_df.count() == 1409
    and published_df.count() == 1409
    and source_duplicates == 0
    and published_duplicates == 0
    and missing_published == 0
    and unexpected_published == 0
)

print("\n3. ASSESSMENT")
print(
    "Customer population:",
    "PASS" if population_pass else "REVIEW REQUIRED"
)

assert population_pass, (
    "Published customer population does not match "
    "the pinned source snapshot."
)

print("\n4. PROTECTED TABLE BASELINE")

approved_count = spark.table(APPROVED_TABLE).count()

approved_version = (
    spark.sql(f"DESCRIBE HISTORY {APPROVED_TABLE}")
    .select("version")
    .first()["version"]
)

print("Approved rows:", approved_count)
print("Approved Delta version:", approved_version)

assert approved_count == 1409
assert approved_version == 3

print("\nSAFETY")
print("New job submitted: False")
print("Inference executed: False")
print("Delta writes performed: False")

print("\nSTEP 15.81H: COMPLETE")


# COMMAND ----------


# Step 15.81I: Published prediction metadata integrity
# READ-ONLY — no inference, job execution, or Delta writes

from pyspark.sql import functions as F

print("=" * 65)
print("PUBLISHED PREDICTION METADATA INTEGRITY")
print("=" * 65)

BATCH_ID = "d972dd16-029d-489f-a406-3c6c6e1e6265"

REGISTRY_TABLE = "workspace.default.telco_15_job_batch_registry"
DEV_TABLE = "workspace.default.telco_15_job_dev_predictions"
APPROVED_TABLE = "workspace.default.telco_15_approved_predictions"

registry_rows = (
    spark.table(REGISTRY_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
    .collect()
)

assert len(registry_rows) == 1
registered = registry_rows[0]

published = (
    spark.table(DEV_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

expected = {
    "model_name": registered["model_name"],
    "model_version": str(registered["model_version"]),
    "classification_threshold": float(
        registered["classification_threshold"]
    ),
    "source_table": registered["source_table"]
}

print("\n1. METADATA CONSISTENCY")

all_pass = True

for column_name, expected_value in expected.items():

    if column_name == "classification_threshold":
        mismatch = (
            F.col(column_name).isNull()
            | (F.abs(F.col(column_name) - F.lit(expected_value)) > 1e-12)
        )
    else:
        mismatch = ~F.col(column_name).cast("string").eqNullSafe(
            F.lit(str(expected_value))
        )

    mismatch_count = published.filter(mismatch).count()

    status = "PASS" if mismatch_count == 0 else "FAIL"
    all_pass = all_pass and mismatch_count == 0

    print(
        f"{column_name}: {status} "
        f"(mismatches={mismatch_count})"
    )

print("\n2. PREDICTION TIMESTAMP")

timestamp_mismatches = published.filter(
    ~F.col("prediction_timestamp").eqNullSafe(
        F.lit(registered["batch_timestamp"]).cast("timestamp")
    )
).count()

print("Timestamp mismatches:", timestamp_mismatches)

all_pass = all_pass and timestamp_mismatches == 0

print("\n3. PROBABILITY AND CLASS VALIDITY")

invalid_probabilities = published.filter(
    F.col("churn_probability").isNull()
    | F.isnan(F.col("churn_probability"))
    | (F.col("churn_probability") < 0)
    | (F.col("churn_probability") > 1)
).count()

invalid_classes = published.filter(
    F.col("predicted_class").isNull()
    | ~F.col("predicted_class").isin(0, 1)
    | F.col("predicted_churn").isNull()
    | ~F.col("predicted_churn").isin(0, 1)
).count()

print("Invalid probabilities:", invalid_probabilities)
print("Invalid predicted classes:", invalid_classes)

all_pass = (
    all_pass
    and invalid_probabilities == 0
    and invalid_classes == 0
)

print("\n4. PROTECTED TABLE BASELINE")

approved_count = spark.table(APPROVED_TABLE).count()
approved_version = (
    spark.sql(f"DESCRIBE HISTORY {APPROVED_TABLE}")
    .select("version")
    .first()["version"]
)

print("Approved rows:", approved_count)
print("Approved Delta version:", approved_version)

assert approved_count == 1409
assert approved_version == 3

print("\nASSESSMENT")
print(
    "Metadata integrity:",
    "PASS" if all_pass else "REVIEW REQUIRED"
)

assert all_pass, "Published prediction metadata integrity failed."

print("\nSAFETY")
print("New job submitted: False")
print("Inference executed: False")
print("Delta writes performed: False")

print("\nSTEP 15.81I: COMPLETE")


# COMMAND ----------


# Step 15.81I.1 — Corrected prediction validity checks
# READ-ONLY: No job execution, inference, or Delta writes

from pyspark.sql import functions as F

print("=" * 65)
print("CORRECTED PREDICTION VALIDITY CHECK")
print("=" * 65)

BATCH_ID = "d972dd16-029d-489f-a406-3c6c6e1e6265"

DEV_TABLE = "workspace.default.telco_15_job_dev_predictions"
APPROVED_TABLE = "workspace.default.telco_15_approved_predictions"

published = (
    spark.table(DEV_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

print("\n1. COLUMN TYPES")

for name in [
    "churn_probability",
    "predicted_class",
    "predicted_churn"
]:
    print(f"{name}: {published.schema[name].dataType}")

print("\n2. PROBABILITY VALIDATION")

invalid_probabilities = published.filter(
    F.col("churn_probability").isNull()
    | F.isnan(F.col("churn_probability").cast("double"))
    | (F.col("churn_probability") < 0)
    | (F.col("churn_probability") > 1)
).count()

print("Invalid probabilities:", invalid_probabilities)

print("\n3. CLASS VALIDATION")

invalid_numeric_classes = published.filter(
    F.col("predicted_class").isNull()
    | ~F.col("predicted_class").isin(0, 1)
).count()

invalid_churn_labels = published.filter(
    F.col("predicted_churn").isNull()
    | ~F.col("predicted_churn").isin("Yes", "No")
).count()

print("Invalid numeric classes:", invalid_numeric_classes)
print("Invalid churn labels:", invalid_churn_labels)

print("\n4. CLASS-LABEL CONSISTENCY")

class_label_mismatches = published.filter(
    (
        (F.col("predicted_class") == 1)
        & (F.col("predicted_churn") != "Yes")
    )
    | (
        (F.col("predicted_class") == 0)
        & (F.col("predicted_churn") != "No")
    )
).count()

print("Class-label mismatches:", class_label_mismatches)

print("\n5. PROTECTED TABLE BASELINE")

approved_count = spark.table(APPROVED_TABLE).count()

approved_version = (
    spark.sql(f"DESCRIBE HISTORY {APPROVED_TABLE}")
    .select("version")
    .first()["version"]
)

print("Approved rows:", approved_count)
print("Approved Delta version:", approved_version)

assert approved_count == 1409
assert approved_version == 3

print("\n6. ASSESSMENT")

checks = {
    "Probability validity": invalid_probabilities == 0,
    "Numeric class validity": invalid_numeric_classes == 0,
    "Churn label validity": invalid_churn_labels == 0,
    "Class-label consistency": class_label_mismatches == 0
}

for check_name, passed in checks.items():
    print(
        f"{check_name}: "
        f"{'PASS' if passed else 'FAIL'}"
    )

assert all(checks.values()), (
    "One or more prediction validity checks failed."
)

print("\nSAFETY")
print("New job submitted: False")
print("Inference executed: False")
print("Delta writes performed: False")

print("\nSTEP 15.81I.1: COMPLETE")


# COMMAND ----------


# Step 15.81J — Classification threshold consistency
# READ-ONLY: No inference, job execution, or Delta writes

from pyspark.sql import functions as F

print("=" * 65)
print("CLASSIFICATION THRESHOLD CONSISTENCY")
print("=" * 65)

BATCH_ID = "d972dd16-029d-489f-a406-3c6c6e1e6265"

REGISTRY_TABLE = "workspace.default.telco_15_job_batch_registry"
DEV_TABLE = "workspace.default.telco_15_job_dev_predictions"
APPROVED_TABLE = "workspace.default.telco_15_approved_predictions"

registry_rows = (
    spark.table(REGISTRY_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
    .collect()
)

assert len(registry_rows) == 1

threshold = float(
    registry_rows[0]["classification_threshold"]
)

published = (
    spark.table(DEV_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

print("\n1. REGISTERED THRESHOLD")
print("Threshold:", threshold)

print("\n2. EXPECTED PREDICTION CLASS")

checked = published.withColumn(
    "expected_class",
    F.when(
        F.col("churn_probability") >= F.lit(threshold),
        F.lit(1)
    ).otherwise(F.lit(0))
)

class_mismatches = checked.filter(
    ~F.col("predicted_class").eqNullSafe(
        F.col("expected_class")
    )
).count()

print("Class mismatches:", class_mismatches)

print("\n3. EXPECTED CHURN LABEL")

checked = checked.withColumn(
    "expected_churn",
    F.when(
        F.col("expected_class") == 1,
        F.lit("Yes")
    ).otherwise(F.lit("No"))
)

label_mismatches = checked.filter(
    ~F.col("predicted_churn").eqNullSafe(
        F.col("expected_churn")
    )
).count()

print("Label mismatches:", label_mismatches)

print("\n4. PREDICTION DISTRIBUTION")

checked.groupBy(
    "predicted_class",
    "predicted_churn"
).count().orderBy(
    "predicted_class"
).show(truncate=False)

print("\n5. PROTECTED TABLE BASELINE")

approved_count = spark.table(APPROVED_TABLE).count()

approved_version = (
    spark.sql(f"DESCRIBE HISTORY {APPROVED_TABLE}")
    .select("version")
    .first()["version"]
)

print("Approved rows:", approved_count)
print("Approved Delta version:", approved_version)

assert approved_count == 1409
assert approved_version == 3

print("\n6. ASSESSMENT")

print(
    "Threshold consistency:",
    "PASS" if class_mismatches == 0 else "FAIL"
)

print(
    "Label consistency:",
    "PASS" if label_mismatches == 0 else "FAIL"
)

assert class_mismatches == 0
assert label_mismatches == 0

print("\nSAFETY")
print("New job submitted: False")
print("Inference executed: False")
print("Delta writes performed: False")

print("\nSTEP 15.81J: COMPLETE")


# COMMAND ----------


# Step 15.81K — Retry safety assessment
# No Spark reads, inference, job execution, or Delta writes

print("=" * 65)
print("BATCH RETRY SAFETY ASSESSMENT")
print("=" * 65)

scenarios = [
    (
        "Identical completed retry",
        "SUPPORTED BY CODE",
        "Registry identity check and strict publication comparison"
    ),
    (
        "Retry after registration but before publication",
        "SUPPORTED BY CODE",
        "Existing registration reused; destination has no batch rows"
    ),
    (
        "Retry after complete Delta MERGE",
        "SUPPORTED BY CODE",
        "Existing matching predictions are not reinserted"
    ),
    (
        "Existing incomplete publication",
        "BLOCKED",
        "Strict validator rejects differing customer populations"
    ),
    (
        "Concurrent first registration",
        "NOT GUARANTEED",
        "Read-then-append is not an atomic uniqueness operation"
    ),
    (
        "Concurrent prediction publication",
        "NOT GUARANTEED",
        "Pre-validation and MERGE are separate operations"
    )
]

for name, status, explanation in scenarios:
    print(f"\nScenario: {name}")
    print(f"Assessment: {status}")
    print(f"Reason: {explanation}")

print("\nRECOMMENDED NEXT DESIGN TASK")
print(
    "Design controlled single-writer retry handling, "
    "registry concurrency protection, and explicit "
    "recovery rules for incomplete batches."
)

print("\nSAFETY")
print("Job executed: False")
print("Notebook modified: False")
print("Delta writes performed: False")

print("\nSTEP 15.81K: COMPLETE")


# COMMAND ----------


# Step 15.81L — Proposed retry state machine
# DESIGN ONLY — no Spark operations or Delta writes

print("=" * 65)
print("PROPOSED PRODUCTION BATCH RETRY STATE MACHINE")
print("=" * 65)

states = [
    {
        "state": "NEW",
        "condition": "No registered batch identity",
        "action": "Acquire exclusive registration ownership",
        "write_policy": "Only the authorized owner may register"
    },
    {
        "state": "REGISTERED_NOT_PUBLISHED",
        "condition": "Registry exists; zero published rows",
        "action": "Validate identity, then publish",
        "write_policy": "Reuse original batch timestamp"
    },
    {
        "state": "COMPLETE_IDENTICAL",
        "condition": "All expected rows exist and values match",
        "action": "Return successful retry result",
        "write_policy": "No prediction changes required"
    },
    {
        "state": "INCOMPLETE_CONSISTENT",
        "condition": "Published rows are a matching subset",
        "action": "Permit controlled recovery",
        "write_policy": "Insert missing rows only"
    },
    {
        "state": "CONFLICT",
        "condition": "Identity, customer population or values conflict",
        "action": "Reject execution and investigate",
        "write_policy": "No writes permitted"
    }
]

for item in states:
    print("\n" + "-" * 65)
    print("STATE:", item["state"])
    print("Condition:", item["condition"])
    print("Action:", item["action"])
    print("Write policy:", item["write_policy"])

print("\nDESIGN CONSTRAINTS")
print("1. No concurrent writers for the same batch.")
print("2. Existing conflicting predictions must never be overwritten.")
print("3. Recovery may insert only verified missing predictions.")
print("4. Batch identity and original timestamp must remain stable.")
print("5. Registry and publication require failure recovery handling.")
print("6. Approved predictions table must remain untouched.")
print("7. Recovery requires explicit authorization before testing.")

print("\nSAFETY")
print("Databricks Job executed: False")
print("Notebook 15C modified: False")
print("Delta writes performed: False")

print("\nSTEP 15.81L: DESIGN COMPLETE")
