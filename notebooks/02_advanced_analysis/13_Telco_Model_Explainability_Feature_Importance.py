# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 13. Telco Customer Churn — Model Explainability
# MAGIC
# MAGIC ## Business Objective
# MAGIC
# MAGIC Explain how the trained Logistic Regression model
# MAGIC uses customer characteristics to estimate churn risk.
# MAGIC
# MAGIC The goal is to improve model transparency and support
# MAGIC responsible business interpretation.
# MAGIC
# MAGIC ## Key Questions
# MAGIC
# MAGIC 1. Which transformed features have the strongest
# MAGIC    influence on the model's churn predictions?
# MAGIC
# MAGIC 2. Which features are associated with higher or
# MAGIC    lower predicted churn risk?
# MAGIC
# MAGIC 3. How can individual customer predictions be
# MAGIC    explained using model coefficients?
# MAGIC
# MAGIC 4. What limitations should stakeholders understand
# MAGIC    when interpreting feature importance?
# MAGIC
# MAGIC ## Model Information
# MAGIC
# MAGIC Model type: Logistic Regression
# MAGIC
# MAGIC Registered model:
# MAGIC workspace.default.telco_churn_logistic_regression
# MAGIC
# MAGIC Model version: 1
# MAGIC
# MAGIC Preprocessing pipeline:
# MAGIC `/Volumes/workspace/default/telco_data/telco_preprocessing_pipeline`
# MAGIC
# MAGIC Feature vector dimensions: 60
# MAGIC
# MAGIC ## Planned Analysis
# MAGIC
# MAGIC - Load and inspect the trained model.
# MAGIC - Inspect preprocessing pipeline stages.
# MAGIC - Recover transformed feature names.
# MAGIC - Analyze Logistic Regression coefficients.
# MAGIC - Visualize global feature importance.
# MAGIC - Explain selected individual predictions.
# MAGIC - Document business interpretations and limitations.
# MAGIC
# MAGIC ## Interpretation Guidelines
# MAGIC
# MAGIC Logistic Regression coefficients describe changes
# MAGIC in predicted log-odds, holding other model features
# MAGIC constant.
# MAGIC
# MAGIC Positive coefficients increase predicted churn
# MAGIC log-odds; negative coefficients decrease them.
# MAGIC
# MAGIC Coefficient magnitude depends on feature scaling
# MAGIC and encoding, so absolute coefficient values are
# MAGIC not automatically comparable across all features.
# MAGIC
# MAGIC Model explanations describe statistical associations
# MAGIC and prediction contributions, not causal effects.
# MAGIC
# MAGIC ## Limitations
# MAGIC
# MAGIC - Model explanations do not establish causation.
# MAGIC - Correlated features can complicate interpretation.
# MAGIC - One-hot encoded categories require careful handling.
# MAGIC - Individual predictions may be uncertain.
# MAGIC - Explainability findings should be validated before
# MAGIC   being used to guide customer interventions.
# MAGIC

# COMMAND ----------


# Step 13.3: Load registered model and preprocessing pipeline
# Fix: Configure Unity Catalog Volume for MLflow temporary files

import os
import mlflow
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel

# Registered model information
MODEL_URI = (
    "models:/workspace.default."
    "telco_churn_logistic_regression/1"
)

# Saved preprocessing pipeline
PIPELINE_PATH = (
    "/Volumes/workspace/default/telco_data/"
    "telco_preprocessing_pipeline"
)

# Temporary storage for Spark MLflow operations
MLFLOW_TMP_PATH = (
    "/Volumes/workspace/default/telco_data/"
    "mlflow_spark_tmp"
)

# Configure MLflow temporary directory
os.environ["MLFLOW_DFS_TMP"] = MLFLOW_TMP_PATH

print("=" * 60)
print("MODEL AND PIPELINE INSPECTION")
print("=" * 60)

# Load fitted preprocessing pipeline
preprocessing_pipeline = PipelineModel.load(
    PIPELINE_PATH
)

print(
    "Preprocessing pipeline type:",
    type(preprocessing_pipeline).__name__
)

print(
    "Preprocessing stages:",
    len(preprocessing_pipeline.stages)
)

# Load registered Logistic Regression model
# Use UC Volume for temporary Spark ML files
churn_model = mlflow.spark.load_model(
    MODEL_URI,
    dfs_tmpdir=MLFLOW_TMP_PATH
)

print(
    "Registered model type:",
    type(churn_model).__name__
)

# Inspect Logistic Regression model
if isinstance(churn_model, LogisticRegressionModel):

    print("\nLOGISTIC REGRESSION MODEL")

    print(
        "Number of coefficients:",
        len(churn_model.coefficients)
    )

    print(
        "Intercept:",
        churn_model.intercept
    )

    print(
        "Features column:",
        churn_model.getFeaturesCol()
    )

    print(
        "Probability column:",
        churn_model.getProbabilityCol()
    )

else:
    print(
        "\nLoaded model is not a direct "
        "LogisticRegressionModel instance."
    )

print("\nMODEL LOADING: COMPLETE")


# COMMAND ----------


# Step 13.4: Inspect registered model pipeline stages

from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel

print("=" * 60)
print("REGISTERED MODEL PIPELINE INSPECTION")
print("=" * 60)

# Inspect each stage of the registered pipeline
for index, stage in enumerate(churn_model.stages):

    print(
        f"Stage {index}: "
        f"{type(stage).__name__}"
    )

# Locate Logistic Regression stage
lr_stages = [
    stage
    for stage in churn_model.stages
    if isinstance(stage, LogisticRegressionModel)
]

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION INSPECTION")
print("=" * 60)

print(
    "Logistic Regression stages found:",
    len(lr_stages)
)

assert len(lr_stages) == 1, (
    "Expected exactly one Logistic Regression stage."
)

# Extract the trained Logistic Regression model
lr_model = lr_stages[0]

print(
    "Number of coefficients:",
    len(lr_model.coefficients)
)

print(
    "Intercept:",
    lr_model.intercept
)

print(
    "Features column:",
    lr_model.getFeaturesCol()
)

print(
    "Probability column:",
    lr_model.getProbabilityCol()
)

print("\nLOGISTIC REGRESSION EXTRACTION: PASS")


# COMMAND ----------


# Step 13.5: Inspect preprocessing stages safely

print("=" * 65)
print("PREPROCESSING PIPELINE STAGES")
print("=" * 65)

# Parameters we want to inspect
parameter_names = [
    "inputCol",
    "inputCols",
    "outputCol",
    "outputCols"
]

# Inspect all preprocessing stages
for index, stage in enumerate(preprocessing_pipeline.stages):

    print(
        f"\nStage {index:02d}: "
        f"{type(stage).__name__}"
    )

    for param_name in parameter_names:

        # Check whether the parameter exists AND has a value
        if (
            stage.hasParam(param_name)
            and stage.isDefined(param_name)
        ):

            value = stage.getOrDefault(param_name)

            print(
                f"  {param_name}: {value}"
            )

print("\n" + "=" * 65)
print("TRAINING DATA INSPECTION")
print("=" * 65)

# Load raw training data
train_df = spark.table(
    "workspace.default.telco_train_raw"
)

print("Training columns:")
print(train_df.columns)

# Check for existing feature vector
if "features" in train_df.columns:

    feature_metadata = (
        train_df.schema["features"].metadata
    )

    print("\nExisting feature metadata:")
    print(feature_metadata)

else:

    print(
        "\nRaw training data does not contain "
        "a features column."
    )

print("\nPREPROCESSING INSPECTION: COMPLETE")


# COMMAND ----------


# Step 13.6: Recover transformed feature names

print("=" * 65)
print("TRANSFORMED FEATURE NAME EXTRACTION")
print("=" * 65)

# Apply the fitted preprocessing pipeline
# No fitting or retraining occurs
transformed_train_df = preprocessing_pipeline.transform(
    train_df
)

# Read feature metadata
feature_metadata = (
    transformed_train_df.schema["features"].metadata
)

ml_attr = feature_metadata.get("ml_attr", {})

print(
    "Metadata feature count:",
    ml_attr.get("num_attrs")
)

# Extract feature names by their original vector indices
feature_names = {}

for attribute_type, attributes in (
    ml_attr.get("attrs", {}).items()
):

    for attribute in attributes:

        feature_index = int(attribute["idx"])

        feature_names[feature_index] = (
            attribute.get(
                "name",
                f"feature_{feature_index}"
            )
        )

# Retrieve Logistic Regression coefficients
coefficients = list(lr_model.coefficients)

print(
    "Model coefficient count:",
    len(coefficients)
)

print(
    "Recovered feature names:",
    len(feature_names)
)

# Validate complete feature mapping
assert len(coefficients) == 60, (
    "Expected 60 Logistic Regression coefficients."
)

assert len(feature_names) == len(coefficients), (
    "Feature metadata count does not match "
    "model coefficient count."
)

assert set(feature_names) == set(range(len(coefficients))), (
    "Feature metadata indices are incomplete."
)

# Build a structured coefficient mapping
feature_coefficient_mapping = [
    (
        index,
        feature_names[index],
        float(coefficients[index])
    )
    for index in range(len(coefficients))
]

# Create Spark DataFrame
feature_coefficients_df = spark.createDataFrame(
    feature_coefficient_mapping,
    [
        "feature_index",
        "feature_name",
        "coefficient"
    ]
)

# Display all transformed features
display(
    feature_coefficients_df.orderBy(
        "feature_index"
    )
)

print("\nFEATURE NAME EXTRACTION: PASS")


# COMMAND ----------


# Step 13.7: Feature importance without Spark execution

import math
import pandas as pd

print("=" * 65)
print("LOGISTIC REGRESSION FEATURE IMPORTANCE")
print("=" * 65)

# Use the feature mapping already created in Step 13.6
# This is an existing Python list, not a Spark DataFrame.

feature_records = []

for index, name, coefficient in feature_coefficient_mapping:

    coefficient = float(coefficient)

    feature_records.append({
        "feature_index": index,
        "feature_name": name,
        "coefficient": coefficient,
        "absolute_coefficient": abs(coefficient),
        "odds_ratio": math.exp(coefficient)
    })

# Convert the 60 local records into a Pandas DataFrame
feature_importance_pd = pd.DataFrame(feature_records)

# Exclude zero-coefficient features from rankings
ranked_features_pd = feature_importance_pd[
    feature_importance_pd["absolute_coefficient"] > 1e-10
].copy()

# Sort positive coefficients
top_positive = (
    ranked_features_pd
    .sort_values("coefficient", ascending=False)
    .head(10)
)

# Sort negative coefficients
top_negative = (
    ranked_features_pd
    .sort_values("coefficient", ascending=True)
    .head(10)
)

# Sort by absolute coefficient magnitude
top_absolute = (
    ranked_features_pd
    .sort_values("absolute_coefficient", ascending=False)
    .head(15)
)

# Print results without Spark display()
print("\nTOP 10 POSITIVE COEFFICIENTS")
print(top_positive.to_string(index=False))

print("\nTOP 10 NEGATIVE COEFFICIENTS")
print(top_negative.to_string(index=False))

print("\nTOP 15 BY ABSOLUTE COEFFICIENT")
print(top_absolute.to_string(index=False))

print("\nFEATURE IMPORTANCE ANALYSIS: COMPLETE")


# COMMAND ----------


# Step 13.8: Visualize global model coefficients

import matplotlib.pyplot as plt
import numpy as np

print("=" * 65)
print("GLOBAL FEATURE COEFFICIENT VISUALIZATION")
print("=" * 65)

# Select the 15 largest coefficients by absolute value
plot_df = (
    feature_importance_pd
    .loc[
        feature_importance_pd["absolute_coefficient"] > 1e-10
    ]
    .nlargest(15, "absolute_coefficient")
    .sort_values("coefficient")
    .copy()
)

# Improve numerical feature labels
feature_labels = {
    "scaled_numerical_features_0": "Tenure (scaled)",
    "scaled_numerical_features_1": "Monthly charges (scaled)",
    "scaled_numerical_features_2": "Total charges (scaled)",
    "scaled_numerical_features_3": "Senior citizen (scaled)"
}

plot_df["display_name"] = (
    plot_df["feature_name"]
    .replace(feature_labels)
    .str.replace("_encoded_", ": ", regex=False)
    .str.replace("_", " ", regex=False)
)

# Positive and negative coefficients use different colors
colors = np.where(
    plot_df["coefficient"] >= 0,
    "tomato",
    "steelblue"
)

plt.figure(figsize=(12, 8))

plt.barh(
    plot_df["display_name"],
    plot_df["coefficient"],
    color=colors
)

plt.axvline(
    x=0,
    color="black",
    linewidth=1
)

plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Transformed Feature")

plt.title(
    "Telco Customer Churn — Top 15 Model Coefficients"
)

plt.tight_layout()
plt.show()

print("\nFEATURE COEFFICIENT VISUALIZATION: COMPLETE")


# COMMAND ----------

# DBTITLE 1,Step 13.9: Explain one high-risk customer prediction

# Step 13.9: Explain one high-risk customer prediction

import math
import pandas as pd
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

print("=" * 65)
print("INDIVIDUAL CUSTOMER PREDICTION EXPLAINABILITY")
print("=" * 65)

# Load raw test data
test_df = spark.table(
    "workspace.default.telco_test_raw"
)

# Apply fitted preprocessing pipeline
transformed_test_df = preprocessing_pipeline.transform(
    test_df
)

# Generate predictions using the extracted LR model
scored_test_df = lr_model.transform(
    transformed_test_df
)

# Select customer with highest predicted churn probability
# Only retrieve one record to the Python environment
customer_row = (
    scored_test_df
    .withColumn(
        "churn_probability",
        vector_to_array(F.col("probability"))[1]
    )
    .orderBy(
        F.col("churn_probability").desc(),
        F.col("customerID").asc()
    )
    .select(
        "customerID",
        "features",
        "churn_probability"
    )
    .first()
)

assert customer_row is not None, "No customer was returned."

customer_id = customer_row["customerID"]
customer_probability = float(
    customer_row["churn_probability"]
)

# Extract customer's transformed feature vector
customer_features = customer_row["features"].toArray()

# Retrieve model coefficients and intercept
coefficients = list(lr_model.coefficients)
intercept = float(lr_model.intercept)

assert len(customer_features) == len(coefficients), (
    "Customer feature vector length does not match coefficients."
)

# Calculate individual feature contributions
contributions = []

for index, feature_value in enumerate(customer_features):

    coefficient = float(coefficients[index])
    feature_value = float(feature_value)

    contribution = coefficient * feature_value

    contributions.append({
        "feature_index": index,
        "feature_name": feature_names[index],
        "feature_value": feature_value,
        "coefficient": coefficient,
        "contribution": contribution
    })

contributions_pd = pd.DataFrame(contributions)

# Reconstruct the model's linear score
manual_log_odds = (
    intercept + contributions_pd["contribution"].sum()
)

# Convert log-odds to churn probability
manual_probability = (
    1 / (1 + math.exp(-manual_log_odds))
)

print("Customer ID:", customer_id)

print(
    "Model churn probability:",
    round(customer_probability, 6)
)

print(
    "Manual churn probability:",
    round(manual_probability, 6)
)

print(
    "Absolute difference:",
    abs(customer_probability - manual_probability)
)

# Validate the explanation
assert math.isclose(
    customer_probability,
    manual_probability,
    rel_tol=1e-6,
    abs_tol=1e-6
), "Manual probability does not match model prediction."

print("\nTOP 10 POSITIVE CONTRIBUTIONS")

print(
    contributions_pd
    .sort_values("contribution", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\nTOP 10 NEGATIVE CONTRIBUTIONS")

print(
    contributions_pd
    .sort_values("contribution", ascending=True)
    .head(10)
    .to_string(index=False)
)

print("\nINDIVIDUAL EXPLANATION VALIDATION: PASS")


# COMMAND ----------


# Step 13.10: Visualize individual customer contributions

import matplotlib.pyplot as plt
import numpy as np

print("=" * 65)
print("INDIVIDUAL CUSTOMER CONTRIBUTION VISUALIZATION")
print("=" * 65)

# Select the 15 largest nonzero contributions
plot_df = (
    contributions_pd
    .assign(
        absolute_contribution=lambda df: df["contribution"].abs()
    )
    .query("absolute_contribution > 1e-10")
    .nlargest(15, "absolute_contribution")
    .sort_values("contribution")
    .copy()
)

# Improve numerical feature labels
feature_labels = {
    "scaled_numerical_features_0": "Tenure (scaled)",
    "scaled_numerical_features_1": "Monthly charges (scaled)",
    "scaled_numerical_features_2": "Total charges (scaled)",
    "scaled_numerical_features_3": "Senior citizen (scaled)"
}

plot_df["display_name"] = (
    plot_df["feature_name"]
    .replace(feature_labels)
    .str.replace("_encoded_", ": ", regex=False)
    .str.replace("_", " ", regex=False)
)

# Red = positive contribution, blue = negative
colors = np.where(
    plot_df["contribution"] >= 0,
    "tomato",
    "steelblue"
)

plt.figure(figsize=(12, 8))

plt.barh(
    plot_df["display_name"],
    plot_df["contribution"],
    color=colors
)

plt.axvline(
    0,
    color="black",
    linewidth=1
)

plt.xlabel("Contribution to Predicted Churn Log-Odds")
plt.ylabel("Customer Feature")

plt.title(
    f"Customer {customer_id} — Churn Prediction Explanation\n"
    f"Predicted Churn Probability: {customer_probability:.2%}"
)

plt.tight_layout()
plt.show()

print("\nINDIVIDUAL CONTRIBUTION VISUALIZATION: COMPLETE")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC %md
# MAGIC # Step 13.11 — Model Explainability and Business Findings
# MAGIC
# MAGIC ## 1. Objective
# MAGIC
# MAGIC Explain the trained Telco Customer Churn Logistic Regression
# MAGIC model at both global and individual customer levels.
# MAGIC
# MAGIC The analysis uses the fitted preprocessing pipeline,
# MAGIC the registered Logistic Regression model, and the
# MAGIC original transformed feature metadata.
# MAGIC
# MAGIC ## 2. Global Model Explainability
# MAGIC
# MAGIC All 60 transformed features were successfully mapped
# MAGIC to their corresponding model coefficients.
# MAGIC
# MAGIC ### Features with positive churn associations
# MAGIC
# MAGIC Notable positive coefficients include:
# MAGIC
# MAGIC - Fiber-optic internet: +1.0249
# MAGIC - Total charges (scaled): +0.7636
# MAGIC - Month-to-month contract: +0.6350
# MAGIC - Streaming movies: +0.4337
# MAGIC - Streaming TV: +0.4313
# MAGIC - Electronic-check payment: +0.2484
# MAGIC
# MAGIC ### Features with negative churn associations
# MAGIC
# MAGIC Notable negative coefficients include:
# MAGIC
# MAGIC - Tenure (scaled): -1.4968
# MAGIC - Monthly charges (scaled): -1.2728
# MAGIC - Two-year contract: -0.8565
# MAGIC - DSL internet: -0.8525
# MAGIC
# MAGIC These coefficients represent conditional associations
# MAGIC with predicted churn log-odds.
# MAGIC
# MAGIC They should not be interpreted as causal effects.
# MAGIC
# MAGIC ## 3. Individual Customer Explainability
# MAGIC
# MAGIC Customer ID: 5150-ITWWB
# MAGIC
# MAGIC Predicted churn probability: 84.69%
# MAGIC
# MAGIC Manual probability reconstruction: 84.69%
# MAGIC
# MAGIC The difference between the two calculations was
# MAGIC approximately 2.22 × 10^-16.
# MAGIC
# MAGIC This validates the coefficient-based decomposition
# MAGIC of the customer's predicted churn log-odds.
# MAGIC
# MAGIC ### Largest positive contributions
# MAGIC
# MAGIC - Low tenure: +1.7816
# MAGIC - Fiber-optic internet: +1.0249
# MAGIC - Month-to-month contract: +0.6350
# MAGIC - Streaming movies: +0.4337
# MAGIC - Streaming TV: +0.4313
# MAGIC
# MAGIC ### Largest negative contributions
# MAGIC
# MAGIC - Monthly charges (scaled): -1.2720
# MAGIC - Total charges (scaled): -0.6521
# MAGIC
# MAGIC The feature contributions and intercept collectively
# MAGIC reconstruct the model's predicted probability.
# MAGIC
# MAGIC ## 4. Business Implications
# MAGIC
# MAGIC The model can help prioritize customers for
# MAGIC retention outreach based on predicted churn risk.
# MAGIC
# MAGIC Potential uses include:
# MAGIC
# MAGIC - Identifying high-risk customers for review.
# MAGIC - Prioritizing outreach using predicted risk scores.
# MAGIC - Supporting customer retention analysis.
# MAGIC - Investigating service and contract characteristics
# MAGIC   associated with predicted churn.
# MAGIC
# MAGIC These are potential applications, not demonstrated
# MAGIC causal retention interventions.
# MAGIC
# MAGIC ## 5. Explainability Limitations
# MAGIC
# MAGIC 1. Logistic Regression coefficients describe
# MAGIC    conditional associations, not causation.
# MAGIC
# MAGIC 2. Numerical features were transformed and scaled,
# MAGIC    so coefficient interpretation depends on the
# MAGIC    preprocessing pipeline.
# MAGIC
# MAGIC 3. Correlated predictors can produce coefficients
# MAGIC    that are difficult to interpret individually.
# MAGIC
# MAGIC 4. Full one-hot encoding complicates interpretation
# MAGIC    of categorical coefficients and reference states.
# MAGIC
# MAGIC 5. Coefficient magnitude is not a universal measure
# MAGIC    of predictive feature importance.
# MAGIC
# MAGIC 6. Individual contributions are relative to the
# MAGIC    model's zero-valued feature baseline, not an
# MAGIC    average customer.
# MAGIC
# MAGIC 7. Model predictions should be monitored for
# MAGIC    performance, calibration, drift, and fairness
# MAGIC    before use in operational decisions.
# MAGIC
# MAGIC ## 6. Conclusion
# MAGIC
# MAGIC The explainability analysis successfully connected
# MAGIC the registered churn model to interpretable feature
# MAGIC names and individual customer predictions.
# MAGIC
# MAGIC The manual probability reconstruction confirms
# MAGIC the mathematical consistency of the explanation.
# MAGIC
# MAGIC This provides a foundation for communicating model
# MAGIC behavior to business stakeholders and for further
# MAGIC validation of model reliability.
# MAGIC

# COMMAND ----------


# Step 13.12A: Final explainability validation

import math

print("=" * 65)
print("NOTEBOOK 13 — FINAL VALIDATION")
print("=" * 65)

# 1. Validate feature count
assert len(feature_importance_pd) == 60, (
    "Expected 60 transformed features."
)

print("1. Feature count: PASS (60)")

# 2. Validate unique feature indices
assert feature_importance_pd["feature_index"].nunique() == 60
assert set(feature_importance_pd["feature_index"]) == set(range(60))

print("2. Feature index mapping: PASS")

# 3. Validate coefficient alignment
for row in feature_importance_pd.itertuples(index=False):
    assert math.isclose(
        float(row.coefficient),
        float(lr_model.coefficients[int(row.feature_index)]),
        rel_tol=1e-10,
        abs_tol=1e-10
    )

print("3. Model coefficient alignment: PASS")

# 4. Validate odds ratios
for row in feature_importance_pd.itertuples(index=False):
    assert math.isclose(
        float(row.odds_ratio),
        math.exp(float(row.coefficient)),
        rel_tol=1e-10
    )

print("4. Odds ratio calculations: PASS")

# 5. Validate individual prediction
assert math.isclose(
    customer_probability,
    manual_probability,
    rel_tol=1e-6,
    abs_tol=1e-6
)

print("5. Individual prediction reconstruction: PASS")

# 6. Validate feature contributions
assert math.isclose(
    float(contributions_pd["contribution"].sum()) + intercept,
    manual_log_odds,
    rel_tol=1e-10,
    abs_tol=1e-10
)

print("6. Feature contribution decomposition: PASS")

print("\nALL NOTEBOOK 13 VALIDATIONS: PASS")


# COMMAND ----------


# Step 13.12B: Save feature importance to Unity Catalog

print("=" * 65)
print("SAVE MODEL EXPLAINABILITY ARTIFACT")
print("=" * 65)

TABLE_NAME = "workspace.default.telco_13_feature_importance"

# Select columns for permanent storage
feature_importance_export = feature_importance_pd[
    [
        "feature_index",
        "feature_name",
        "coefficient",
        "absolute_coefficient",
        "odds_ratio"
    ]
].copy()

# Ensure correct data types
feature_importance_export["feature_index"] = (
    feature_importance_export["feature_index"].astype(int)
)

for col in [
    "coefficient",
    "absolute_coefficient",
    "odds_ratio"
]:
    feature_importance_export[col] = (
        feature_importance_export[col].astype(float)
    )

# Convert the small Pandas DataFrame to Spark
feature_importance_spark = spark.createDataFrame(
    feature_importance_export
)

# Save as a managed Delta table
(
    feature_importance_spark.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(TABLE_NAME)
)

print("Saved table:", TABLE_NAME)
print("Expected rows:", len(feature_importance_export))

print("\nFEATURE IMPORTANCE TABLE SAVE: COMPLETE")


# COMMAND ----------


# Step 13.12C: Verify saved feature importance table

table_name = "workspace.default.telco_13_feature_importance"

saved_df = spark.table(table_name)

row_count = saved_df.count()

expected_columns = [
    "feature_index",
    "feature_name",
    "coefficient",
    "absolute_coefficient",
    "odds_ratio"
]

print("=" * 65)
print("FEATURE IMPORTANCE TABLE VERIFICATION")
print("=" * 65)

print("Table:", table_name)
print("Row count:", row_count)
print("Columns:", saved_df.columns)

assert row_count == 60, "Expected 60 feature records."
assert saved_df.columns == expected_columns, (
    "Saved columns do not match expected columns."
)

print("\nFEATURE IMPORTANCE TABLE VALIDATION: PASS")
