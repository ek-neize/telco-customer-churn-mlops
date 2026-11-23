# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC %md
# MAGIC # Notebook 14 — Telco Model Calibration and Reliability
# MAGIC
# MAGIC ## Project
# MAGIC Telco Customer Churn Prediction — End-to-End ML/MLOps
# MAGIC
# MAGIC ## Objective
# MAGIC Evaluate whether predicted customer churn probabilities
# MAGIC accurately reflect observed churn frequencies.
# MAGIC
# MAGIC ## Model
# MAGIC Registered Logistic Regression model:
# MAGIC
# MAGIC `workspace.default.telco_churn_logistic_regression`
# MAGIC
# MAGIC ## Evaluation Dataset
# MAGIC `workspace.default.telco_test_raw`
# MAGIC
# MAGIC ## Key Evaluation Methods
# MAGIC 1. Predicted probability distribution
# MAGIC 2. Reliability diagram
# MAGIC 3. Brier score
# MAGIC 4. Log loss
# MAGIC 5. Calibration assessment
# MAGIC 6. Optional calibration improvement
# MAGIC
# MAGIC ## Methodology
# MAGIC Use the existing fitted preprocessing pipeline and
# MAGIC registered Logistic Regression model.
# MAGIC
# MAGIC Evaluate probability reliability on held-out test data.
# MAGIC
# MAGIC Do not fit probability calibration transformations
# MAGIC on the evaluation dataset.
# MAGIC
# MAGIC ## Expected Outcome
# MAGIC Determine whether the model's predicted churn
# MAGIC probabilities are sufficiently reliable for
# MAGIC risk-based customer retention decisions.
# MAGIC

# COMMAND ----------


# Step 14.4: Load model, preprocessing pipeline, and test data

import os
import mlflow
import mlflow.spark

from pyspark.ml import PipelineModel

print("=" * 65)
print("NOTEBOOK 14 — MODEL AND DATA LOADING")
print("=" * 65)

# Unity Catalog model registry
mlflow.set_registry_uri("databricks-uc")

# Registered Logistic Regression model
MODEL_URI = (
    "models:/workspace.default."
    "telco_churn_logistic_regression/1"
)

# Saved fitted preprocessing pipeline
PIPELINE_PATH = (
    "/Volumes/workspace/default/telco_data/"
    "telco_preprocessing_pipeline"
)

# Temporary location for MLflow Spark model loading
MLFLOW_TMP_PATH = (
    "/Volumes/workspace/default/telco_data/"
    "mlflow_spark_tmp"
)

os.environ["MLFLOW_DFS_TMP"] = MLFLOW_TMP_PATH

# Load registered model
churn_model = mlflow.spark.load_model(
    MODEL_URI,
    dfs_tmpdir=MLFLOW_TMP_PATH
)

print("Registered model loaded successfully.")

# Load preprocessing pipeline
preprocessing_pipeline = PipelineModel.load(
    PIPELINE_PATH
)

print("Preprocessing pipeline loaded successfully.")

# Load held-out test dataset
test_df = spark.table(
    "workspace.default.telco_test_raw"
)

print("Test dataset loaded successfully.")

# Validate model and pipeline
print("\nModel stages:", len(churn_model.stages))
print("Preprocessing stages:", len(preprocessing_pipeline.stages))
print("Test rows:", test_df.count())

assert len(churn_model.stages) == 1, (
    "Expected one Logistic Regression model stage."
)

assert len(preprocessing_pipeline.stages) == 20, (
    "Expected 20 preprocessing stages."
)

assert test_df.count() == 1409, (
    "Expected 1,409 test records."
)

print("\nMODEL AND DATA LOADING: PASS")


# COMMAND ----------


# Step 14.5: Inspect churn labels and model configuration

from pyspark.sql import functions as F

print("=" * 65)
print("CHURN LABEL AND MODEL CONFIGURATION VALIDATION")
print("=" * 65)

# Extract the trained Logistic Regression model
lr_model = churn_model.stages[0]

print("Model type:", type(lr_model).__name__)
print("Label column:", lr_model.getLabelCol())
print("Features column:", lr_model.getFeaturesCol())
print("Probability column:", lr_model.getProbabilityCol())
print("Number of classes:", lr_model.numClasses)

# Inspect original churn labels
print("\nORIGINAL CHURN LABEL DISTRIBUTION")

test_df.groupBy("Churn").count().orderBy("Churn").show()

# Inspect target label column availability
label_col = lr_model.getLabelCol()

print("\nLabel column present in raw test data:",
      label_col in test_df.columns)

# Inspect the model's expected class count
assert lr_model.numClasses == 2, (
    "Expected a binary classification model."
)

print("\nMODEL CONFIGURATION INSPECTION: COMPLETE")


# COMMAND ----------


# Step 14.6: Verify positive-class mapping

from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

print("=" * 65)
print("POSITIVE CLASS MAPPING VALIDATION")
print("=" * 65)

# Apply the existing fitted preprocessing pipeline
transformed_test_df = preprocessing_pipeline.transform(test_df)

# Generate model predictions
scored_test_df = lr_model.transform(transformed_test_df)

# Extract class-1 probability and predicted class
mapping_df = (
    scored_test_df
    .withColumn(
        "probability_class_1",
        vector_to_array(F.col("probability"))[1]
    )
    .withColumn(
        "actual_churn",
        F.when(F.col("Churn") == "Yes", 1)
         .when(F.col("Churn") == "No", 0)
    )
    .select(
        "customerID",
        "Churn",
        "actual_churn",
        "prediction",
        "probability_class_1"
    )
)

# Verify valid actual outcomes
assert mapping_df.filter(
    F.col("actual_churn").isNull()
).count() == 0, "Unexpected Churn label found."

print("\nAVERAGE CLASS-1 PROBABILITY BY ACTUAL OUTCOME")

mapping_df.groupBy("Churn").agg(
    F.count("*").alias("customers"),
    F.avg("probability_class_1").alias(
        "avg_probability_class_1"
    )
).orderBy("Churn").show(truncate=False)

print("\nPREDICTED CLASS VS ACTUAL CHURN")

mapping_df.groupBy(
    "Churn", "prediction"
).count().orderBy(
    "Churn", "prediction"
).show()

print("\nPOSITIVE CLASS MAPPING INSPECTION: COMPLETE")


# COMMAND ----------


# Step 14.7: Prepare calibration evaluation dataset

from pyspark.sql import functions as F

print("=" * 65)
print("CALIBRATION EVALUATION DATA PREPARATION")
print("=" * 65)

calibration_df = (
    mapping_df
    .select(
        "customerID",
        F.col("actual_churn").cast("int"),
        F.col("probability_class_1")
         .cast("double")
         .alias("churn_probability")
    )
)

# Validate row count
row_count = calibration_df.count()

assert row_count == 1409, (
    f"Expected 1409 rows, found {row_count}"
)

# Validate probability range and missing values
invalid_rows = calibration_df.filter(
    F.col("actual_churn").isNull()
    | F.col("churn_probability").isNull()
    | F.isnan("churn_probability")
    | (F.col("churn_probability") < 0)
    | (F.col("churn_probability") > 1)
).count()

assert invalid_rows == 0, (
    f"Found {invalid_rows} invalid records"
)

# Verify customer IDs are unique
unique_customers = calibration_df.select(
    "customerID"
).distinct().count()

assert unique_customers == row_count, (
    "Duplicate customer IDs detected"
)

# Summarize evaluation data
calibration_df.agg(
    F.count("*").alias("total_customers"),
    F.sum("actual_churn").alias("actual_churners"),
    F.avg("actual_churn").alias("observed_churn_rate"),
    F.avg("churn_probability").alias(
        "mean_predicted_churn_probability"
    ),
    F.min("churn_probability").alias("min_probability"),
    F.max("churn_probability").alias("max_probability")
).show(truncate=False)

print("\nCALIBRATION DATA PREPARATION: PASS")


# COMMAND ----------


# Step 14.8: Predicted churn probability distribution

import matplotlib.pyplot as plt
import numpy as np

print("=" * 65)
print("PREDICTED CHURN PROBABILITY DISTRIBUTION")
print("=" * 65)

# Collect only the two columns needed for plotting
calibration_pd = calibration_df.select(
    "actual_churn",
    "churn_probability"
).toPandas()

# Validate local dataset
assert len(calibration_pd) == 1409

# Extract probabilities
probabilities = calibration_pd[
    "churn_probability"
].to_numpy()

# Create histogram
plt.figure(figsize=(11, 6))

plt.hist(
    probabilities,
    bins=np.linspace(0, 1, 21),
    edgecolor="black",
    alpha=0.75
)

# Add reference lines
plt.axvline(
    probabilities.mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean predicted: {probabilities.mean():.2%}"
)

plt.axvline(
    calibration_pd["actual_churn"].mean(),
    color="green",
    linestyle="--",
    linewidth=2,
    label=(
        "Observed churn rate: "
        f"{calibration_pd['actual_churn'].mean():.2%}"
    )
)

plt.title(
    "Distribution of Predicted Customer Churn Probabilities",
    fontsize=14
)

plt.xlabel("Predicted Churn Probability")
plt.ylabel("Number of Customers")

plt.xlim(0, 1)
plt.xticks(np.arange(0, 1.01, 0.1))

plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("\nPROBABILITY DISTRIBUTION SUMMARY")
print(calibration_pd["churn_probability"].describe())

print("\nPROBABILITY DISTRIBUTION ANALYSIS: COMPLETE")


# COMMAND ----------


# Step 14.9: Reliability diagram and calibration table

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 65)
print("MODEL CALIBRATION — RELIABILITY DIAGRAM")
print("=" * 65)

# Use the evaluation data prepared in Step 14.8
calibration_analysis_pd = calibration_pd.copy()

# Divide predicted probabilities into 10 fixed-width bins
bin_edges = np.linspace(0, 1, 11)

calibration_analysis_pd["probability_bin"] = pd.cut(
    calibration_analysis_pd["churn_probability"],
    bins=bin_edges,
    include_lowest=True
)

# Calculate observed and predicted churn by bin
calibration_summary = (
    calibration_analysis_pd
    .groupby("probability_bin", observed=False)
    .agg(
        customers=("actual_churn", "size"),
        mean_predicted_probability=(
            "churn_probability", "mean"
        ),
        observed_churn_rate=("actual_churn", "mean")
    )
    .reset_index()
)

# Remove empty bins from the plotted results
calibration_summary = calibration_summary[
    calibration_summary["customers"] > 0
].copy()

# Calculate calibration gap
calibration_summary["calibration_gap"] = (
    calibration_summary["observed_churn_rate"]
    - calibration_summary["mean_predicted_probability"]
)

print("\nCALIBRATION SUMMARY BY PROBABILITY BIN")
print(calibration_summary.to_string(index=False))

# Plot reliability diagram
plt.figure(figsize=(9, 7))

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Perfect calibration"
)

plt.plot(
    calibration_summary["mean_predicted_probability"],
    calibration_summary["observed_churn_rate"],
    marker="o",
    linewidth=2,
    markersize=7,
    label="Logistic Regression"
)

plt.title(
    "Telco Churn Model — Reliability Diagram",
    fontsize=14
)

plt.xlabel("Mean Predicted Churn Probability")
plt.ylabel("Observed Churn Rate")

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("\nRELIABILITY DIAGRAM ANALYSIS: COMPLETE")


# COMMAND ----------


# Step 14.10: Brier Score and Log Loss

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

print("=" * 65)
print("MODEL PROBABILITY QUALITY METRICS")
print("=" * 65)

# Actual churn outcomes and predicted probabilities
y_true = calibration_pd["actual_churn"].to_numpy()
y_prob = calibration_pd["churn_probability"].to_numpy()

# Calculate model metrics
model_brier = brier_score_loss(y_true, y_prob)

model_log_loss = log_loss(
    y_true,
    y_prob,
    labels=[0, 1]
)

# Constant-probability reference benchmark
baseline_probability = y_true.mean()

baseline_predictions = np.full(
    len(y_true),
    baseline_probability
)

baseline_brier = brier_score_loss(
    y_true,
    baseline_predictions
)

baseline_log_loss = log_loss(
    y_true,
    baseline_predictions,
    labels=[0, 1]
)

# Relative Brier improvement
brier_improvement = (
    (baseline_brier - model_brier)
    / baseline_brier
) * 100

print(f"Total customers: {len(y_true)}")
print(f"Observed churn rate: {baseline_probability:.6f}")

print("\nMODEL METRICS")
print(f"Brier Score: {model_brier:.6f}")
print(f"Log Loss:    {model_log_loss:.6f}")

print("\nCONSTANT-PROBABILITY BASELINE")
print(f"Baseline Brier Score: {baseline_brier:.6f}")
print(f"Baseline Log Loss:    {baseline_log_loss:.6f}")

print("\nRELATIVE IMPROVEMENT")
print(f"Brier Score improvement: {brier_improvement:.2f}%")

# Basic validation
assert np.isfinite(model_brier)
assert np.isfinite(model_log_loss)
assert 0 <= model_brier <= 1

print("\nPROBABILITY QUALITY METRICS: PASS")


# COMMAND ----------


# Step 14.11: Expected Calibration Error (ECE)

import numpy as np

print("=" * 65)
print("EXPECTED CALIBRATION ERROR ANALYSIS")
print("=" * 65)

# Reuse the 10-bin calibration summary from Step 14.9
ece_df = calibration_summary.copy()

# Calculate the weight of each probability bin
ece_df["bin_weight"] = (
    ece_df["customers"] / len(calibration_pd)
)

# Absolute calibration gap
ece_df["absolute_calibration_gap"] = (
    ece_df["calibration_gap"].abs()
)

# Weighted calibration error
ece_df["weighted_calibration_error"] = (
    ece_df["bin_weight"]
    * ece_df["absolute_calibration_gap"]
)

# Expected Calibration Error
ece = ece_df["weighted_calibration_error"].sum()

# Maximum Calibration Error
mce = ece_df["absolute_calibration_gap"].max()

print(f"Total customers: {len(calibration_pd)}")
print(f"Nonempty probability bins: {len(ece_df)}")

print("\nCALIBRATION METRICS")
print(f"Expected Calibration Error: {ece:.6f}")
print(f"ECE as percentage points: {ece * 100:.2f}")
print(f"Maximum Calibration Error: {mce:.6f}")
print(f"MCE as percentage points: {mce * 100:.2f}")

print("\nCALIBRATION ERROR CONTRIBUTIONS")

print(
    ece_df[
        [
            "probability_bin",
            "customers",
            "absolute_calibration_gap",
            "weighted_calibration_error"
        ]
    ].to_string(index=False)
)

# Validation
assert np.isfinite(ece)
assert 0 <= ece <= 1
assert ece_df["customers"].sum() == 1409

print("\nEXPECTED CALIBRATION ERROR VALIDATION: PASS")


# COMMAND ----------


# Step 14.12: Inspect training data for calibration experiment

from pyspark.sql import functions as F

print("=" * 65)
print("CALIBRATION EXPERIMENT — TRAINING DATA VALIDATION")
print("=" * 65)

train_df = spark.table(
    "workspace.default.telco_train_raw"
)

print("\nTraining rows:", train_df.count())
print("Test rows:", test_df.count())

print("\nTraining columns:")
print(train_df.columns)

print("\nTraining churn distribution:")
train_df.groupBy("Churn").count().orderBy("Churn").show()

print("\nPreprocessing pipeline label-related stages:")

for index, stage in enumerate(preprocessing_pipeline.stages):
    stage_type = type(stage).__name__

    if hasattr(stage, "getOutputCol"):
        output_col = stage.getOutputCol()
    else:
        output_col = None

    if (
        "Index" in stage_type
        or output_col == "label"
    ):
        print(
            f"Stage {index}: {stage_type}, "
            f"output={output_col}"
        )

        if hasattr(stage, "labels"):
            print("  Labels:", list(stage.labels))

# Validate train/test separation
train_ids = set(
    row.customerID
    for row in train_df.select("customerID").collect()
)

test_ids = set(
    row.customerID
    for row in test_df.select("customerID").collect()
)

overlap = train_ids.intersection(test_ids)

print("\nOverlapping customer IDs:", len(overlap))

assert len(overlap) == 0, (
    "Training and test datasets overlap."
)

assert train_df.count() == 5634, (
    "Unexpected training dataset size."
)

print("\nTRAINING DATA VALIDATION: PASS")


# COMMAND ----------


# Step 14.13: Create stratified fitting/calibration subsets

from pyspark.sql import functions as F
from pyspark.sql.window import Window

print("=" * 65)
print("STRATIFIED CALIBRATION DATA SPLIT")
print("=" * 65)

# Explicit binary target encoding
train_labeled_df = train_df.withColumn(
    "label",
    F.when(F.col("Churn") == "Yes", F.lit(1.0))
     .when(F.col("Churn") == "No", F.lit(0.0))
)

# Deterministic ordering within each class
split_window = Window.partitionBy("label").orderBy(
    F.sha2(
        F.concat(
            F.lit("telco_calibration_2026_"),
            F.col("customerID")
        ),
        256
    )
)

count_window = Window.partitionBy("label")

train_ranked_df = (
    train_labeled_df
    .withColumn(
        "_split_rank",
        F.row_number().over(split_window)
    )
    .withColumn(
        "_class_count",
        F.count("*").over(count_window)
    )
    .withColumn(
        "_calibration_cutoff",
        F.floor(F.col("_class_count") * 0.20)
    )
)

# 20% calibration, 80% model fitting
calibration_fit_df = (
    train_ranked_df
    .filter(
        F.col("_split_rank") > F.col("_calibration_cutoff")
    )
    .drop(
        "_split_rank",
        "_class_count",
        "_calibration_cutoff"
    )
)

calibration_holdout_df = (
    train_ranked_df
    .filter(
        F.col("_split_rank") <= F.col("_calibration_cutoff")
    )
    .drop(
        "_split_rank",
        "_class_count",
        "_calibration_cutoff"
    )
)

print("\nMODEL-FITTING SUBSET")
calibration_fit_df.groupBy("Churn").count().show()

print("\nCALIBRATION SUBSET")
calibration_holdout_df.groupBy("Churn").count().show()

fit_count = calibration_fit_df.count()
holdout_count = calibration_holdout_df.count()

print("\nTOTAL COUNTS")
print("Model-fitting customers:", fit_count)
print("Calibration customers:", holdout_count)
print("Combined customers:", fit_count + holdout_count)

# Validate split integrity
assert fit_count + holdout_count == 5634

assert (
    calibration_fit_df.select("customerID")
    .intersect(
        calibration_holdout_df.select("customerID")
    )
    .count()
) == 0, "Fitting/calibration overlap detected."

print("\nSTRATIFIED CALIBRATION SPLIT: PASS")


# COMMAND ----------


# Step 14.14: Inspect fitted preprocessing pipeline

from pyspark.ml import PipelineModel

print("=" * 65)
print("PREPROCESSING PIPELINE STRUCTURE INSPECTION")
print("=" * 65)

for index, stage in enumerate(preprocessing_pipeline.stages):

    stage_type = type(stage).__name__

    print(f"\nStage {index}: {stage_type}")

    if hasattr(stage, "getInputCol"):
        try:
            print("  Input column:", stage.getInputCol())
        except Exception:
            pass

    if hasattr(stage, "getInputCols"):
        try:
            print("  Input columns:", stage.getInputCols())
        except Exception:
            pass

    if hasattr(stage, "getOutputCol"):
        try:
            print("  Output column:", stage.getOutputCol())
        except Exception:
            pass

    if hasattr(stage, "getOutputCols"):
        try:
            print("  Output columns:", stage.getOutputCols())
        except Exception:
            pass

    if hasattr(stage, "getHandleInvalid"):
        try:
            print("  Handle invalid:", stage.getHandleInvalid())
        except Exception:
            pass

    if hasattr(stage, "getDropLast"):
        try:
            print("  Drop last:", stage.getDropLast())
        except Exception:
            pass

    if hasattr(stage, "getWithMean"):
        try:
            print("  With mean:", stage.getWithMean())
        except Exception:
            pass

    if hasattr(stage, "getWithStd"):
        try:
            print("  With std:", stage.getWithStd())
        except Exception:
            pass

print("\nPIPELINE STRUCTURE INSPECTION: COMPLETE")


# COMMAND ----------


# Step 14.15: Reconstruct preprocessing pipeline

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)

print("=" * 65)
print("LEAKAGE-SAFE PREPROCESSING PIPELINE RECONSTRUCTION")
print("=" * 65)

# 1. Numerical imputation
original_imputer = preprocessing_pipeline.stages[0]

imputer = Imputer(
    inputCols=["TotalCharges"],
    outputCols=["TotalCharges_imputed"],
    strategy=original_imputer.getStrategy()
)

# 2. Categorical indexing
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

indexers = [
    StringIndexer(
        inputCol=col_name,
        outputCol=f"{col_name}_index",
        handleInvalid="keep"
    )
    for col_name in categorical_columns
]

# 3. One-hot encoding
encoder = OneHotEncoder(
    inputCols=[
        f"{col_name}_index"
        for col_name in categorical_columns
    ],
    outputCols=[
        f"{col_name}_encoded"
        for col_name in categorical_columns
    ],
    handleInvalid="keep",
    dropLast=True
)

# 4. Numerical feature assembly
numerical_assembler = VectorAssembler(
    inputCols=[
        "tenure",
        "MonthlyCharges",
        "TotalCharges_imputed",
        "SeniorCitizen"
    ],
    outputCol="numerical_features",
    handleInvalid="error"
)

# 5. Numerical standardization
scaler = StandardScaler(
    inputCol="numerical_features",
    outputCol="scaled_numerical_features",
    withMean=True,
    withStd=True
)

# 6. Final feature assembly
final_assembler = VectorAssembler(
    inputCols=[
        "scaled_numerical_features"
    ] + [
        f"{col_name}_encoded"
        for col_name in categorical_columns
    ],
    outputCol="features",
    handleInvalid="error"
)

# Build new unfitted pipeline
candidate_preprocessing_pipeline = Pipeline(
    stages=(
        [imputer]
        + indexers
        + [
            encoder,
            numerical_assembler,
            scaler,
            final_assembler
        ]
    )
)

# Validate stage count
stage_count = len(
    candidate_preprocessing_pipeline.getStages()
)

print("Candidate preprocessing stages:", stage_count)
print("Original preprocessing stages:",
      len(preprocessing_pipeline.stages))

assert stage_count == 20

print("\nPipeline stage types:")

for index, stage in enumerate(
    candidate_preprocessing_pipeline.getStages()
):
    print(f"Stage {index}: {type(stage).__name__}")

print("\nPREPROCESSING PIPELINE RECONSTRUCTION: PASS")


# COMMAND ----------


# Step 14.16: Fit leakage-safe preprocessing pipeline

print("=" * 65)
print("CANDIDATE PREPROCESSING PIPELINE FITTING")
print("=" * 65)

# Fit preprocessing using ONLY model-fitting customers
candidate_fitted_pipeline = (
    candidate_preprocessing_pipeline.fit(
        calibration_fit_df
    )
)

print("Candidate preprocessing pipeline fitted successfully.")

# Transform model-fitting data
candidate_fit_transformed_df = (
    candidate_fitted_pipeline.transform(
        calibration_fit_df
    )
)

print("Model-fitting dataset transformed successfully.")

# Validate stage count
fitted_stage_count = len(
    candidate_fitted_pipeline.stages
)

print("\nFitted preprocessing stages:", fitted_stage_count)

assert fitted_stage_count == 20, (
    "Expected 20 fitted preprocessing stages."
)

# Validate transformed row count
transformed_count = candidate_fit_transformed_df.count()

print("Transformed training customers:", transformed_count)

assert transformed_count == 4508, (
    "Unexpected transformed training row count."
)

# Inspect feature-vector dimension
first_row = candidate_fit_transformed_df.select(
    "features"
).first()

feature_count = len(first_row["features"])

print("Feature vector dimension:", feature_count)

assert feature_count > 0, (
    "Feature vector must not be empty."
)

print("\nCANDIDATE PREPROCESSING FITTING: PASS")


# COMMAND ----------


# Step 14.17: Train candidate Logistic Regression model

from pyspark.ml.classification import LogisticRegression

print("=" * 65)
print("CANDIDATE LOGISTIC REGRESSION MODEL TRAINING")
print("=" * 65)

# Inspect the registered model's training configuration
print("\nORIGINAL MODEL CONFIGURATION")

print("Regularization parameter:", lr_model.getRegParam())
print("Elastic net parameter:", lr_model.getElasticNetParam())
print("Maximum iterations:", lr_model.getMaxIter())
print("Convergence tolerance:", lr_model.getTol())
print("Fit intercept:", lr_model.getFitIntercept())
print("Standardization:", lr_model.getStandardization())
print("Threshold:", lr_model.getThreshold())

# Create candidate model using the original settings
candidate_lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    regParam=lr_model.getRegParam(),
    elasticNetParam=lr_model.getElasticNetParam(),
    maxIter=lr_model.getMaxIter(),
    tol=lr_model.getTol(),
    fitIntercept=lr_model.getFitIntercept(),
    standardization=lr_model.getStandardization(),
    threshold=lr_model.getThreshold()
)

# Train ONLY on the 4,508 model-fitting customers
candidate_lr_model = candidate_lr.fit(
    candidate_fit_transformed_df
)

print("\nCandidate Logistic Regression trained successfully.")

# Validate trained model
print("\nTRAINED MODEL DETAILS")
print("Model type:", type(candidate_lr_model).__name__)
print("Number of coefficients:", len(candidate_lr_model.coefficients))
print("Intercept:", candidate_lr_model.intercept)
print("Number of classes:", candidate_lr_model.numClasses)

assert candidate_lr_model.numClasses == 2

assert len(candidate_lr_model.coefficients) == feature_count

print("\nCANDIDATE MODEL TRAINING: PASS")


# COMMAND ----------


# Step 14.18: Generate calibration holdout predictions

from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

print("=" * 65)
print("CALIBRATION HOLDOUT PREDICTION GENERATION")
print("=" * 65)

# Apply preprocessing fitted on model-fitting customers
calibration_transformed_df = (
    candidate_fitted_pipeline.transform(
        calibration_holdout_df
    )
)

print("Calibration preprocessing completed.")

# Generate predictions using the candidate model
calibration_scored_df = (
    candidate_lr_model.transform(
        calibration_transformed_df
    )
)

print("Calibration predictions generated.")

# Extract churn probabilities
calibration_predictions_df = (
    calibration_scored_df
    .select(
        "customerID",
        F.col("label").cast("int").alias("actual_churn"),
        F.col("prediction").cast("int").alias("prediction"),
        vector_to_array(
            F.col("probability")
        )[1].alias("churn_probability")
    )
)

# Validate predictions
calibration_count = calibration_predictions_df.count()

invalid_count = calibration_predictions_df.filter(
    F.col("actual_churn").isNull()
    | F.col("churn_probability").isNull()
    | F.isnan("churn_probability")
    | (F.col("churn_probability") < 0)
    | (F.col("churn_probability") > 1)
).count()

assert calibration_count == 1126
assert invalid_count == 0

print("\nCALIBRATION PREDICTION SUMMARY")

calibration_predictions_df.agg(
    F.count("*").alias("customers"),
    F.sum("actual_churn").alias("actual_churners"),
    F.avg("actual_churn").alias("observed_churn_rate"),
    F.avg("churn_probability").alias(
        "mean_predicted_probability"
    ),
    F.min("churn_probability").alias("min_probability"),
    F.max("churn_probability").alias("max_probability")
).show(truncate=False)

print("\nCALIBRATION HOLDOUT PREDICTIONS: PASS")


# COMMAND ----------


# Step 14.19: Fit sigmoid probability calibration

import numpy as np

from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

print("=" * 65)
print("SIGMOID PROBABILITY CALIBRATION")
print("=" * 65)

# Extract class-1 raw decision scores and actual outcomes
calibration_scores_df = (
    calibration_scored_df
    .select(
        "customerID",
        F.col("label").cast("int").alias("actual_churn"),
        vector_to_array(
            F.col("rawPrediction")
        )[1].alias("raw_score")
    )
)

# Convert small calibration dataset to Pandas
calibration_scores_pd = (
    calibration_scores_df.toPandas()
)

# Prepare inputs
X_calibration = (
    calibration_scores_pd[["raw_score"]].to_numpy()
)

y_calibration = (
    calibration_scores_pd["actual_churn"].to_numpy()
)

# Validate inputs
assert len(X_calibration) == 1126
assert len(np.unique(y_calibration)) == 2
assert np.isfinite(X_calibration).all()

# Fit sigmoid calibration model
sigmoid_calibrator = SklearnLogisticRegression(
    penalty=None,
    solver="lbfgs",
    max_iter=1000
)

sigmoid_calibrator.fit(
    X_calibration,
    y_calibration
)

print("\nSigmoid calibration model fitted successfully.")

print("\nCALIBRATION PARAMETERS")
print(
    "Sigmoid coefficient:",
    sigmoid_calibrator.coef_[0][0]
)
print(
    "Sigmoid intercept:",
    sigmoid_calibrator.intercept_[0]
)

# Generate calibrated probabilities on calibration data
calibrated_probabilities = (
    sigmoid_calibrator.predict_proba(
        X_calibration
    )[:, 1]
)

print("\nCALIBRATION HOLDOUT SUMMARY")
print(
    "Observed churn rate:",
    round(y_calibration.mean(), 6)
)
print(
    "Mean calibrated probability:",
    round(calibrated_probabilities.mean(), 6)
)

# Validate probabilities
assert np.all(
    (calibrated_probabilities >= 0)
    & (calibrated_probabilities <= 1)
)

print("\nSIGMOID CALIBRATION FITTING: PASS")


# COMMAND ----------


# Step 14.20: Independent test evaluation

import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array
from sklearn.metrics import brier_score_loss, log_loss

print("=" * 65)
print("INDEPENDENT TEST — CALIBRATION COMPARISON")
print("=" * 65)

# 1. Prepare test labels using the same target mapping
candidate_test_df = test_df.withColumn(
    "label",
    F.when(F.col("Churn") == "Yes", F.lit(1.0))
     .when(F.col("Churn") == "No", F.lit(0.0))
)

# 2. Apply candidate preprocessing (fitted on 4,508)
candidate_test_transformed_df = (
    candidate_fitted_pipeline.transform(candidate_test_df)
)

# 3. Score using the candidate Logistic Regression model
candidate_test_scored_df = (
    candidate_lr_model.transform(
        candidate_test_transformed_df
    )
)

# 4. Extract actual labels, probabilities, and raw scores
candidate_test_scores_pd = (
    candidate_test_scored_df
    .select(
        "customerID",
        F.col("label").cast("int").alias("actual_churn"),
        vector_to_array(
            F.col("probability")
        )[1].alias("uncalibrated_probability"),
        vector_to_array(
            F.col("rawPrediction")
        )[1].alias("raw_score")
    )
    .toPandas()
)

# 5. Apply sigmoid calibrator to independent test scores
X_test_scores = (
    candidate_test_scores_pd[["raw_score"]].to_numpy()
)

candidate_test_scores_pd["calibrated_probability"] = (
    sigmoid_calibrator.predict_proba(X_test_scores)[:, 1]
)

# 6. Prepare evaluation arrays
y_test = candidate_test_scores_pd["actual_churn"].to_numpy()

p_uncalibrated = (
    candidate_test_scores_pd["uncalibrated_probability"].to_numpy()
)

p_calibrated = (
    candidate_test_scores_pd["calibrated_probability"].to_numpy()
)

# 7. Calculate metrics
uncalibrated_brier = brier_score_loss(
    y_test, p_uncalibrated
)

calibrated_brier = brier_score_loss(
    y_test, p_calibrated
)

uncalibrated_logloss = log_loss(
    y_test, p_uncalibrated, labels=[0, 1]
)

calibrated_logloss = log_loss(
    y_test, p_calibrated, labels=[0, 1]
)

# 8. Display comparison
comparison_df = pd.DataFrame({
    "Model": [
        "Uncalibrated Candidate",
        "Sigmoid-Calibrated Candidate"
    ],
    "Brier Score": [
        uncalibrated_brier,
        calibrated_brier
    ],
    "Log Loss": [
        uncalibrated_logloss,
        calibrated_logloss
    ],
    "Mean Predicted Probability": [
        p_uncalibrated.mean(),
        p_calibrated.mean()
    ]
})

print("\nTEST DATASET SUMMARY")
print("Customers:", len(y_test))
print("Actual churners:", int(y_test.sum()))
print("Observed churn rate:", round(y_test.mean(), 6))

print("\nMODEL COMPARISON")
print(comparison_df.to_string(index=False))

print("\nCALIBRATION IMPACT")
print(
    "Brier Score change:",
    round(calibrated_brier - uncalibrated_brier, 6)
)
print(
    "Log Loss change:",
    round(calibrated_logloss - uncalibrated_logloss, 6)
)

# 9. Validate results
assert len(y_test) == 1409
assert int(y_test.sum()) == 383

assert np.isfinite(
    [uncalibrated_brier, calibrated_brier,
     uncalibrated_logloss, calibrated_logloss]
).all()

assert np.all((p_calibrated >= 0) & (p_calibrated <= 1))

print("\nINDEPENDENT TEST CALIBRATION COMPARISON: PASS")


# COMMAND ----------


# Step 14.21: Reliability curves and calibration error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 65)
print("RELIABILITY CURVES AND CALIBRATION ERROR")
print("=" * 65)

# Fixed probability intervals from 0 to 1
bin_edges = np.linspace(0, 1, 11)

def calculate_reliability(y_true, probabilities, model_name):
    rows = []

    for i in range(10):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        # Include the right boundary in the final bin
        if i == 9:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        mean_probability = float(
            probabilities[mask].mean()
        )

        observed_rate = float(
            y_true[mask].mean()
        )

        gap = observed_rate - mean_probability

        rows.append({
            "Model": model_name,
            "Bin": f"{lower:.1f}-{upper:.1f}",
            "Customers": count,
            "Mean Probability": mean_probability,
            "Observed Churn": observed_rate,
            "Calibration Gap": gap,
            "Absolute Gap": abs(gap)
        })

    return pd.DataFrame(rows)


# Calculate reliability summaries
uncalibrated_reliability = calculate_reliability(
    y_test,
    p_uncalibrated,
    "Uncalibrated"
)

calibrated_reliability = calculate_reliability(
    y_test,
    p_calibrated,
    "Sigmoid-Calibrated"
)

reliability_comparison = pd.concat(
    [
        uncalibrated_reliability,
        calibrated_reliability
    ],
    ignore_index=True
)

# Calculate ECE and MCE
def calculate_calibration_errors(reliability_df):
    total = reliability_df["Customers"].sum()

    ece = (
        reliability_df["Customers"]
        * reliability_df["Absolute Gap"]
    ).sum() / total

    mce = reliability_df["Absolute Gap"].max()

    return float(ece), float(mce)


uncalibrated_ece, uncalibrated_mce = (
    calculate_calibration_errors(
        uncalibrated_reliability
    )
)

calibrated_ece, calibrated_mce = (
    calculate_calibration_errors(
        calibrated_reliability
    )
)

# Display comparison
calibration_error_df = pd.DataFrame({
    "Model": [
        "Uncalibrated Candidate",
        "Sigmoid-Calibrated Candidate"
    ],
    "ECE": [
        uncalibrated_ece,
        calibrated_ece
    ],
    "MCE": [
        uncalibrated_mce,
        calibrated_mce
    ]
})

print("\nCALIBRATION ERROR COMPARISON")
print(
    calibration_error_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nRELIABILITY BIN DETAILS")
print(
    reliability_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# Plot reliability curves
plt.figure(figsize=(9, 7))

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    color="gray",
    label="Perfect Calibration"
)

plt.plot(
    uncalibrated_reliability["Mean Probability"],
    uncalibrated_reliability["Observed Churn"],
    marker="o",
    label="Uncalibrated Candidate"
)

plt.plot(
    calibrated_reliability["Mean Probability"],
    calibrated_reliability["Observed Churn"],
    marker="s",
    label="Sigmoid-Calibrated Candidate"
)

plt.xlabel("Mean Predicted Churn Probability")
plt.ylabel("Observed Churn Rate")
plt.title("Telco Churn — Probability Reliability Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Validate bin coverage
assert uncalibrated_reliability["Customers"].sum() == 1409
assert calibrated_reliability["Customers"].sum() == 1409

assert np.isfinite([
    uncalibrated_ece,
    uncalibrated_mce,
    calibrated_ece,
    calibrated_mce
]).all()

print("\nRELIABILITY AND CALIBRATION ERROR: PASS")


# COMMAND ----------


# Step 14.22: Fit isotonic probability calibration

import numpy as np
from sklearn.isotonic import IsotonicRegression

print("=" * 65)
print("ISOTONIC PROBABILITY CALIBRATION")
print("=" * 65)

# Use the existing calibration holdout predictions
# These were generated without fitting on holdout customers

isotonic_calibration_pd = (
    calibration_predictions_df
    .select(
        "actual_churn",
        "churn_probability"
    )
    .toPandas()
)

X_isotonic = (
    isotonic_calibration_pd["churn_probability"]
    .to_numpy()
)

y_isotonic = (
    isotonic_calibration_pd["actual_churn"]
    .to_numpy()
)

# Validate calibration data
assert len(X_isotonic) == 1126
assert len(np.unique(y_isotonic)) == 2
assert np.isfinite(X_isotonic).all()

# Fit isotonic calibration
isotonic_calibrator = IsotonicRegression(
    y_min=0.0,
    y_max=1.0,
    increasing=True,
    out_of_bounds="clip"
)

isotonic_calibrator.fit(
    X_isotonic,
    y_isotonic
)

print("\nIsotonic calibration fitted successfully.")

# Generate fitted probabilities on calibration holdout
isotonic_calibration_probabilities = (
    isotonic_calibrator.predict(X_isotonic)
)

print("\nCALIBRATION HOLDOUT SUMMARY")

print(
    "Customers:",
    len(y_isotonic)
)

print(
    "Observed churn rate:",
    round(y_isotonic.mean(), 6)
)

print(
    "Mean isotonic probability:",
    round(isotonic_calibration_probabilities.mean(), 6)
)

print(
    "Minimum isotonic probability:",
    round(isotonic_calibration_probabilities.min(), 6)
)

print(
    "Maximum isotonic probability:",
    round(isotonic_calibration_probabilities.max(), 6)
)

print(
    "Learned isotonic thresholds:",
    len(isotonic_calibrator.X_thresholds_)
)

# Validate results
assert np.isfinite(
    isotonic_calibration_probabilities
).all()

assert np.all(
    (isotonic_calibration_probabilities >= 0)
    & (isotonic_calibration_probabilities <= 1)
)

print("\nISOTONIC CALIBRATION FITTING: PASS")


# COMMAND ----------


# Step 14.23: Compare three probability calibration methods

import numpy as np
import pandas as pd

from sklearn.metrics import brier_score_loss, log_loss

print("=" * 65)
print("THREE-MODEL PROBABILITY CALIBRATION COMPARISON")
print("=" * 65)

# 1. Generate isotonic probabilities on the test dataset
p_isotonic = isotonic_calibrator.predict(
    p_uncalibrated
)

# 2. Validate test probabilities
assert len(p_isotonic) == 1409
assert np.isfinite(p_isotonic).all()
assert np.all((p_isotonic >= 0) & (p_isotonic <= 1))

# 3. Reuse the reliability functions from Step 14.21
isotonic_reliability = calculate_reliability(
    y_test,
    p_isotonic,
    "Isotonic-Calibrated"
)

isotonic_ece, isotonic_mce = (
    calculate_calibration_errors(
        isotonic_reliability
    )
)

# 4. Calculate all comparison metrics
models_to_compare = {
    "Uncalibrated Candidate": p_uncalibrated,
    "Sigmoid-Calibrated Candidate": p_calibrated,
    "Isotonic-Calibrated Candidate": p_isotonic
}

comparison_rows = []

for model_name, probabilities in models_to_compare.items():

    reliability = calculate_reliability(
        y_test,
        probabilities,
        model_name
    )

    ece, mce = calculate_calibration_errors(
        reliability
    )

    comparison_rows.append({
        "Model": model_name,
        "Brier Score": brier_score_loss(
            y_test, probabilities
        ),
        "Log Loss": log_loss(
            y_test, probabilities, labels=[0, 1]
        ),
        "ECE": ece,
        "MCE": mce,
        "Mean Probability": probabilities.mean(),
        "Exact 0 or 1": int(
            np.sum(
                (probabilities == 0)
                | (probabilities == 1)
            )
        )
    })

three_model_comparison_df = pd.DataFrame(
    comparison_rows
)

print("\nMODEL COMPARISON")
print(
    three_model_comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# 5. Compare isotonic against the uncalibrated candidate
print("\nISOTONIC VS UNCALIBRATED")

print(
    "Brier Score change:",
    round(
        brier_score_loss(y_test, p_isotonic)
        - brier_score_loss(y_test, p_uncalibrated),
        6
    )
)

print(
    "Log Loss change:",
    round(
        log_loss(y_test, p_isotonic, labels=[0, 1])
        - log_loss(y_test, p_uncalibrated, labels=[0, 1]),
        6
    )
)

print(
    "ECE change:",
    round(
        isotonic_ece - uncalibrated_ece,
        6
    )
)

# 6. Validate reliability coverage
assert isotonic_reliability["Customers"].sum() == 1409

print("\nTHREE-MODEL CALIBRATION COMPARISON: PASS")


# COMMAND ----------


# Step 14.24: Investigate extreme isotonic probabilities

import numpy as np
import pandas as pd

print("=" * 65)
print("ISOTONIC EXTREME PROBABILITY DIAGNOSTICS")
print("=" * 65)

# Build test prediction diagnostic dataset
isotonic_diagnostics_df = pd.DataFrame({
    "actual_churn": y_test,
    "uncalibrated_probability": p_uncalibrated,
    "isotonic_probability": p_isotonic
})

# Identify probabilities exactly equal to 0 or 1
extreme_mask = (
    (isotonic_diagnostics_df["isotonic_probability"] == 0)
    |
    (isotonic_diagnostics_df["isotonic_probability"] == 1)
)

extreme_predictions_df = (
    isotonic_diagnostics_df.loc[extreme_mask].copy()
)

# Identify confidently incorrect predictions
extreme_predictions_df["incorrect_extreme"] = (
    (
        (extreme_predictions_df["isotonic_probability"] == 0)
        &
        (extreme_predictions_df["actual_churn"] == 1)
    )
    |
    (
        (extreme_predictions_df["isotonic_probability"] == 1)
        &
        (extreme_predictions_df["actual_churn"] == 0)
    )
)

print("\nEXTREME PREDICTION SUMMARY")
print("Total test customers:", len(y_test))
print("Extreme isotonic predictions:", len(extreme_predictions_df))

print(
    "Extreme probability = 0:",
    int((extreme_predictions_df["isotonic_probability"] == 0).sum())
)

print(
    "Extreme probability = 1:",
    int((extreme_predictions_df["isotonic_probability"] == 1).sum())
)

print(
    "Incorrect extreme predictions:",
    int(extreme_predictions_df["incorrect_extreme"].sum())
)

print("\nEXTREME PREDICTION BREAKDOWN")

print(
    pd.crosstab(
        extreme_predictions_df["isotonic_probability"],
        extreme_predictions_df["actual_churn"],
        rownames=["Isotonic Probability"],
        colnames=["Actual Churn"]
    )
)

# Show representative examples
print("\nSAMPLE EXTREME PREDICTIONS")

print(
    extreme_predictions_df.head(10).to_string(
        index=False
    )
)

# Validate count
assert len(extreme_predictions_df) == 60

print("\nISOTONIC EXTREME DIAGNOSTICS: PASS")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Notebook 14 — Model Calibration and Reliability
# MAGIC ## Final Findings and Recommendations
# MAGIC
# MAGIC ### 1. Objective
# MAGIC
# MAGIC Evaluate whether post-training probability calibration improves
# MAGIC the reliability of a Telco Customer Churn Logistic Regression model.
# MAGIC
# MAGIC ### 2. Experimental Design
# MAGIC
# MAGIC - Model-fitting customers: 4,508
# MAGIC - Calibration holdout customers: 1,126
# MAGIC - Test customers: 1,409
# MAGIC - Preprocessing fitted exclusively on model-fitting customers
# MAGIC - Candidate Logistic Regression trained on model-fitting customers
# MAGIC - Sigmoid and isotonic calibrators fitted on calibration holdout
# MAGIC - All three approaches evaluated on the same test dataset
# MAGIC
# MAGIC The original registered MLflow model was not modified.
# MAGIC
# MAGIC ### 3. Model Comparison
# MAGIC
# MAGIC | Metric | Uncalibrated | Sigmoid | Isotonic |
# MAGIC |---|---:|---:|---:|
# MAGIC | Brier Score | 0.139279 | 0.139347 | 0.141487 |
# MAGIC | Log Loss | 0.429929 | 0.428644 | 0.503645 |
# MAGIC | Expected Calibration Error | 0.022194 | 0.024938 | 0.036391 |
# MAGIC | Maximum Calibration Error | 0.152044 | 0.812715 | 0.655142 |
# MAGIC | Mean Predicted Probability | 0.259443 | 0.261973 | 0.262843 |
# MAGIC
# MAGIC Observed test churn rate: 0.271824.
# MAGIC
# MAGIC ### 4. Key Findings
# MAGIC
# MAGIC **Uncalibrated Logistic Regression**
# MAGIC
# MAGIC - Lowest Brier Score among the three candidates.
# MAGIC - Lowest Expected Calibration Error.
# MAGIC - No exact zero or one probability predictions.
# MAGIC - Preferred reference candidate for this experiment.
# MAGIC
# MAGIC **Sigmoid Calibration**
# MAGIC
# MAGIC - Slight improvement in Log Loss.
# MAGIC - Slight deterioration in Brier Score and ECE.
# MAGIC - No compelling overall improvement in probability quality.
# MAGIC
# MAGIC **Isotonic Calibration**
# MAGIC
# MAGIC - Worse Brier Score, Log Loss, and ECE.
# MAGIC - Produced 60 extreme probability predictions.
# MAGIC - Three extreme predictions were incorrect:
# MAGIC   - One actual churner received a 0% churn probability.
# MAGIC   - Two non-churners received 100% churn probabilities.
# MAGIC - Extreme incorrect predictions contributed to the
# MAGIC   deterioration in Log Loss.
# MAGIC
# MAGIC ### 5. Business Implications
# MAGIC
# MAGIC Reliable churn probabilities support:
# MAGIC
# MAGIC - Customer retention prioritization.
# MAGIC - Risk-based marketing outreach.
# MAGIC - Retention budget allocation.
# MAGIC - Expected-value-based decision-making.
# MAGIC
# MAGIC Calibration should be adopted only when it demonstrates
# MAGIC measurable improvements in probability reliability and
# MAGIC business decision quality.
# MAGIC
# MAGIC ### 6. Recommendation
# MAGIC
# MAGIC Retain the uncalibrated Logistic Regression candidate as
# MAGIC the reference approach for this calibration experiment.
# MAGIC
# MAGIC Do not replace the existing registered MLflow model based
# MAGIC on these exploratory results.
# MAGIC
# MAGIC ### 7. Limitations
# MAGIC
# MAGIC - Results are based on one train/calibration split.
# MAGIC - The calibration dataset contains 1,126 customers.
# MAGIC - Isotonic calibration may overfit small calibration samples.
# MAGIC - ECE and MCE depend on the selected probability bins.
# MAGIC - MCE is particularly sensitive to sparsely populated bins.
# MAGIC - The test dataset was inspected repeatedly during experimentation,
# MAGIC   introducing potential model-selection bias.
# MAGIC - The candidate model was trained on fewer customers than
# MAGIC   the original registered model.
# MAGIC
# MAGIC ### 8. Recommended Future Work
# MAGIC
# MAGIC - Evaluate calibration using cross-validation.
# MAGIC - Compare results across multiple random splits.
# MAGIC - Evaluate calibration stability across customer segments.
# MAGIC - Use an untouched validation dataset for final model selection.
# MAGIC - Assess whether calibration improves expected retention value.
# MAGIC - Monitor probability reliability after deployment.
# MAGIC
# MAGIC ### Final Conclusion
# MAGIC
# MAGIC Probability calibration does not automatically improve model
# MAGIC reliability.
# MAGIC
# MAGIC In this experiment, the uncalibrated Logistic Regression
# MAGIC candidate demonstrated the strongest overall probability
# MAGIC quality across Brier Score and Expected Calibration Error.
# MAGIC
# MAGIC The experiment highlights the importance of validating
# MAGIC calibration methods before deploying them in production.
# MAGIC