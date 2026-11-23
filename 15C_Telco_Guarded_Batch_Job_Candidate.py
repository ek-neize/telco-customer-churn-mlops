# Databricks notebook source
# DBTITLE 1,Production snapshot-protected batch job source
# Databricks notebook source
# Telco Churn - Snapshot-Protected Batch Inference
# Production notebook - initialization and functions


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


from pyspark.sql.types import StructType

registry_schema = StructType.fromJson({'type': 'struct', 'fields': [{'name': 'batch_id', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'source_table', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'source_fingerprint', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'record_count', 'type': 'long', 'nullable': False, 'metadata': {}}, {'name': 'model_name', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'model_version', 'type': 'string', 'nullable': False, 'metadata': {}}, {'name': 'classification_threshold', 'type': 'double', 'nullable': False, 'metadata': {}}, {'name': 'batch_timestamp', 'type': 'timestamp', 'nullable': False, 'metadata': {}}, {'name': 'registered_at', 'type': 'timestamp', 'nullable': False, 'metadata': {}}]})



# Initialize production preprocessing pipeline

job_preprocessing_pipeline = PipelineModel.load(
    JOB_PIPELINE_PATH
)

# Initialize registered MLflow Spark model

# Serverless and shared clusters require a UC
# volume temp directory for SparkML model loading
import os

os.environ["MLFLOW_DFS_TMP"] = (
    "/Volumes/workspace/default/telco_data/mlflow_tmp"
)

job_registered_model = mlflow.spark.load_model(
    JOB_MODEL_URI
)



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

    if registry_table != 'workspace.default.telco_15_job_batch_registry':
        raise ValueError(
            'Write blocked: unauthorized table destination.'
        )

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
            ~F.col("batch_id").eqNullSafe(F.lit(batch_id))
            | ~F.col("model_name").eqNullSafe(F.lit(model_name))
            | ~F.col("model_version").eqNullSafe(F.lit(model_version))
            | ~F.col("classification_threshold").eqNullSafe(
                F.lit(threshold)
            )
            | ~F.col("source_table").eqNullSafe(
                F.lit(source_table)
            )
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

    if destination_table != 'workspace.default.telco_15_job_dev_predictions':
        raise ValueError(
            'Write blocked: unauthorized table destination.'
        )

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

    allowed_destination = (
        "workspace.default.telco_15_job_dev_predictions"
    )

    if destination_table != allowed_destination:
        raise ValueError(
            "Publication blocked: only the development "
            "predictions table is permitted."
        )

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



# Preserve the original Delta publisher.
_original_publish_quality_approved_batch = (
    publish_quality_approved_batch
)

# Activate strict publication validation.
publish_quality_approved_batch = (
    publish_quality_approved_batch_strict
)

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
