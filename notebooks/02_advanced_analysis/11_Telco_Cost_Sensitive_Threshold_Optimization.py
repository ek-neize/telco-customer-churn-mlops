# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Notebook 11 — Cost-Sensitive Churn Threshold Optimization
# MAGIC
# MAGIC ## 1. Business Objective and Economic Assumptions
# MAGIC
# MAGIC ### 1.1 Business Problem
# MAGIC
# MAGIC A telecommunications provider wants to identify customers at risk of churn and determine which customers should receive retention interventions.
# MAGIC
# MAGIC Traditional classification metrics such as accuracy, precision, recall, and F1 score help evaluate predictive performance, but they do not directly measure the financial value of a retention strategy.
# MAGIC
# MAGIC This notebook extends the Telco Customer Churn project by evaluating classification thresholds using an illustrative expected net benefit framework.
# MAGIC
# MAGIC ### 1.2 Objectives
# MAGIC
# MAGIC The objectives are to:
# MAGIC
# MAGIC 1. Load customer churn predictions and actual churn labels.
# MAGIC 2. Evaluate classification performance at multiple thresholds.
# MAGIC 3. Estimate intervention costs and expected retention benefits.
# MAGIC 4. Compare threshold-based targeting strategies.
# MAGIC 5. Identify economically attractive thresholds under stated assumptions.
# MAGIC 6. Perform sensitivity analysis to understand how financial assumptions influence decisions.
# MAGIC
# MAGIC ### 1.3 Economic Assumptions
# MAGIC
# MAGIC | Parameter | Assumed Value |
# MAGIC |---|---:|
# MAGIC | Net customer value preserved | $500 |
# MAGIC | Intervention cost per targeted customer | $50 |
# MAGIC | Probability of preventing churn, conditional on churn | 20% |
# MAGIC | Baseline threshold | 0.50 |
# MAGIC | Previously optimized threshold | 0.40 |
# MAGIC
# MAGIC All financial assumptions are illustrative and are not derived from observed customer retention campaigns.
# MAGIC
# MAGIC ### 1.4 Expected Net Benefit
# MAGIC
# MAGIC For a targeted customer:
# MAGIC
# MAGIC Expected Net Benefit = P(Churn) × P(Prevent Churn | Intervention, Churn) × Value Preserved − Intervention Cost
# MAGIC
# MAGIC For a customer with predicted churn probability of 0.60:
# MAGIC
# MAGIC Expected Net Benefit = 0.60 × 0.20 × $500 − $50
# MAGIC
# MAGIC Expected Net Benefit = $10
# MAGIC
# MAGIC Customers who are not targeted incur no intervention cost and receive no modeled intervention benefit.
# MAGIC
# MAGIC ### 1.5 Important Assumptions and Limitations
# MAGIC
# MAGIC - Churn probabilities are treated as estimates of baseline churn risk.
# MAGIC - The intervention effectiveness rate is hypothetical.
# MAGIC - Customer value and intervention cost are assumed constant across customers.
# MAGIC - The analysis does not establish causal intervention effects.
# MAGIC - Actual campaign outcomes and realized savings are unavailable.
# MAGIC - Calibration should be evaluated before interpreting predicted probabilities as reliable economic inputs.
# MAGIC - Threshold selection on the historical test dataset is exploratory and would require independent validation before deployment.
# MAGIC
# MAGIC ### 1.6 Expected Outcome
# MAGIC
# MAGIC The notebook will produce a comparison of classification thresholds, including predictive performance, number of customers targeted, estimated intervention costs, and simulated expected net benefit.
# MAGIC
# MAGIC The final recommendation will be conditional on the stated economic assumptions rather than presented as a validated production retention policy.
# MAGIC

# COMMAND ----------


from pyspark.sql import functions as F

# Existing prediction tables
prediction_tables = [
    "workspace.default.telco_09_batch_predictions",
    "workspace.default.telco_10_prediction_history"
]

for table_name in prediction_tables:
    print("=" * 80)
    print(f"TABLE: {table_name}")
    print("=" * 80)

    df = spark.table(table_name)

    print("\nTotal records:", df.count())

    print("\nSchema:")
    df.printSchema()

    print("\nColumn names:")
    print(df.columns)

    print("\nSample records:")
    display(df.limit(5))


# COMMAND ----------


# Step 11.4: Inspect actual customer churn labels

test_df = spark.table("workspace.default.telco_test_raw")

print("Total test records:", test_df.count())

print("\nTest dataset schema:")
test_df.printSchema()

print("\nAvailable columns:")
print(test_df.columns)

print("\nCustomer and churn label sample:")

label_columns = [
    col for col in ["customerID", "Churn", "label"]
    if col in test_df.columns
]

if label_columns:
    display(test_df.select(*label_columns).limit(10))
else:
    print("No expected label columns found.")


# COMMAND ----------


from pyspark.sql import functions as F

# -------------------------------------------------
# 1. Load saved predictions
# -------------------------------------------------

BATCH_ID = "telco_788040999362398"

predictions_df = (
    spark.table("workspace.default.telco_10_prediction_history")
    .filter(F.col("batch_id") == BATCH_ID)
    .select(
        "customerID",
        "churn_probability",
        "batch_id",
        "model_version"
    )
)

# -------------------------------------------------
# 2. Load actual churn labels
# -------------------------------------------------

actual_df = (
    spark.table("workspace.default.telco_test_raw")
    .select(
        "customerID",
        F.when(F.col("Churn") == "Yes", 1)
         .when(F.col("Churn") == "No", 0)
         .otherwise(None)
         .cast("int")
         .alias("actual_churn")
    )
)

# -------------------------------------------------
# 3. Join predictions with actual outcomes
# -------------------------------------------------

analysis_df = predictions_df.join(
    actual_df,
    on="customerID",
    how="inner"
)

# -------------------------------------------------
# 4. Validate the joined dataset
# -------------------------------------------------

total_predictions = predictions_df.count()
total_actual = actual_df.count()
total_joined = analysis_df.count()

unique_customers = (
    analysis_df.select("customerID").distinct().count()
)

null_probabilities = analysis_df.filter(
    F.col("churn_probability").isNull()
).count()

null_labels = analysis_df.filter(
    F.col("actual_churn").isNull()
).count()

invalid_probabilities = analysis_df.filter(
    (F.col("churn_probability") < 0) |
    (F.col("churn_probability") > 1)
).count()

print("=" * 60)
print("NOTEBOOK 11 — DATA VALIDATION")
print("=" * 60)

print(f"Prediction records:      {total_predictions}")
print(f"Actual label records:    {total_actual}")
print(f"Joined records:          {total_joined}")
print(f"Unique customers:        {unique_customers}")
print(f"Missing probabilities:   {null_probabilities}")
print(f"Missing actual labels:   {null_labels}")
print(f"Invalid probabilities:   {invalid_probabilities}")

# -------------------------------------------------
# 5. Fail early if validation is unsuccessful
# -------------------------------------------------

assert total_predictions == 1409, "Unexpected prediction count"
assert total_actual == 1409, "Unexpected label count"
assert total_joined == 1409, "Join lost or duplicated records"
assert unique_customers == 1409, "Duplicate customer IDs detected"
assert null_probabilities == 0, "Missing probabilities detected"
assert null_labels == 0, "Missing actual churn labels detected"
assert invalid_probabilities == 0, "Invalid probabilities detected"

print("\nDATA VALIDATION: PASS")

# -------------------------------------------------
# 6. Preview analysis data
# -------------------------------------------------

display(
    analysis_df.select(
        "customerID",
        "churn_probability",
        "actual_churn",
        "batch_id",
        "model_version"
    ).orderBy("customerID").limit(10)
)


# COMMAND ----------


# Step 11.6: Classification threshold comparison

from pyspark.sql import functions as F

# Thresholds from 0.10 to 0.90
thresholds = [round(i / 100, 2) for i in range(10, 91, 5)]

results = []

# Collect only the two columns required for analysis.
# The validated dataset contains 1,409 records.
customer_scores = [
    (float(row["churn_probability"]), int(row["actual_churn"]))
    for row in analysis_df.select(
        "churn_probability", "actual_churn"
    ).collect()
]

for threshold in thresholds:

    tp = sum(
        1 for p, y in customer_scores
        if p >= threshold and y == 1
    )

    fp = sum(
        1 for p, y in customer_scores
        if p >= threshold and y == 0
    )

    tn = sum(
        1 for p, y in customer_scores
        if p < threshold and y == 0
    )

    fn = sum(
        1 for p, y in customer_scores
        if p < threshold and y == 1
    )

    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    results.append({
        "threshold": threshold,
        "targeted_customers": tp + fp,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

# Create Spark DataFrame

from pyspark.sql.types import (
    StructType,
    StructField,
    DoubleType,
    LongType
)

# Define explicit schema for threshold results
threshold_schema = StructType([
    StructField("threshold", DoubleType(), False),
    StructField("targeted_customers", LongType(), False),
    StructField("TP", LongType(), False),
    StructField("FP", LongType(), False),
    StructField("TN", LongType(), False),
    StructField("FN", LongType(), False),
    StructField("accuracy", DoubleType(), False),
    StructField("precision", DoubleType(), False),
    StructField("recall", DoubleType(), False),
    StructField("f1_score", DoubleType(), False)
])

# Convert values to their expected Python types
typed_results = [
    {
        "threshold": float(r["threshold"]),
        "targeted_customers": int(r["targeted_customers"]),
        "TP": int(r["TP"]),
        "FP": int(r["FP"]),
        "TN": int(r["TN"]),
        "FN": int(r["FN"]),
        "accuracy": float(r["accuracy"]),
        "precision": float(r["precision"]),
        "recall": float(r["recall"]),
        "f1_score": float(r["f1_score"])
    }
    for r in results
]

# Create Spark DataFrame with explicit schema
threshold_df = spark.createDataFrame(
    typed_results,
    schema=threshold_schema
)

# Display results
display(threshold_df.orderBy("threshold"))

print("Threshold evaluation completed.")
print(f"Thresholds evaluated: {len(thresholds)}")
print(f"Customers analyzed: {len(customer_scores)}")




# COMMAND ----------


# Step 11.7: Cost-sensitive threshold optimization

import pandas as pd

# Illustrative economic assumptions
CUSTOMER_VALUE = 500.0
INTERVENTION_COST = 50.0
RETENTION_EFFECTIVENESS = 0.20

economic_results = []

for threshold in thresholds:

    # Customers selected for retention outreach
    targeted = [
        (p, y)
        for p, y in customer_scores
        if p >= threshold
    ]

    targeted_count = len(targeted)

    # Expected prevented churn events based on
    # predicted probabilities and assumed effectiveness
    expected_prevented_churn = sum(
        p * RETENTION_EFFECTIVENESS
        for p, y in targeted
    )

    # Simulated economic value
    expected_benefit = (
        expected_prevented_churn * CUSTOMER_VALUE
    )

    total_cost = (
        targeted_count * INTERVENTION_COST
    )

    expected_net_benefit = (
        expected_benefit - total_cost
    )

    economic_results.append({
        "threshold": threshold,
        "targeted_customers": targeted_count,
        "expected_prevented_churn": expected_prevented_churn,
        "expected_benefit": expected_benefit,
        "intervention_cost": total_cost,
        "expected_net_benefit": expected_net_benefit
    })

# Create Pandas DataFrame for the small summary
economic_pd = pd.DataFrame(economic_results)

# Sort by threshold
economic_pd = economic_pd.sort_values(
    "threshold"
).reset_index(drop=True)

# Display financial comparison
display(economic_pd.round(2))

# Identify the best threshold among those tested
best_row = economic_pd.loc[
    economic_pd["expected_net_benefit"].idxmax()
]

print("=" * 60)
print("COST-SENSITIVE THRESHOLD OPTIMIZATION")
print("=" * 60)

print(
    f"Best tested threshold: "
    f"{best_row['threshold']:.2f}"
)

print(
    f"Customers targeted: "
    f"{int(best_row['targeted_customers'])}"
)

print(
    f"Expected benefit: "
    f"${best_row['expected_benefit']:,.2f}"
)

print(
    f"Intervention cost: "
    f"${best_row['intervention_cost']:,.2f}"
)

print(
    f"Expected net benefit: "
    f"${best_row['expected_net_benefit']:,.2f}"
)


# COMMAND ----------


# Step 11.8: Compare classification and economic performance

# Classification results from Step 11.6
classification_pd = threshold_df.toPandas()

# Economic results from Step 11.7
combined_pd = classification_pd.merge(
    economic_pd[
        [
            "threshold",
            "expected_benefit",
            "intervention_cost",
            "expected_net_benefit"
        ]
    ],
    on="threshold",
    how="inner",
    validate="one_to_one"
)

# Select important comparison thresholds
comparison_thresholds = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60]

comparison_pd = combined_pd[
    combined_pd["threshold"].round(2).isin(comparison_thresholds)
].copy()

# Display key classification and financial metrics
display(
    comparison_pd[
        [
            "threshold",
            "targeted_customers",
            "precision",
            "recall",
            "f1_score",
            "expected_benefit",
            "intervention_cost",
            "expected_net_benefit"
        ]
    ].round(4).sort_values("threshold")
)

# Identify the best threshold for each objective
best_f1 = combined_pd.loc[
    combined_pd["f1_score"].idxmax()
]

best_economic = combined_pd.loc[
    combined_pd["expected_net_benefit"].idxmax()
]

print("=" * 60)
print("CLASSIFICATION VS ECONOMIC OPTIMIZATION")
print("=" * 60)

print(
    f"Best F1 threshold: "
    f"{best_f1['threshold']:.2f}"
)

print(
    f"Best F1 score: "
    f"{best_f1['f1_score']:.4f}"
)

print(
    f"Best economic threshold: "
    f"{best_economic['threshold']:.2f}"
)

print(
    f"Highest simulated net benefit: "
    f"${best_economic['expected_net_benefit']:,.2f}"
)


# COMMAND ----------


# Step 11.9: Economic sensitivity analysis

import pandas as pd

scenarios = [
    {
        "scenario": "Baseline",
        "customer_value": 500.0,
        "intervention_cost": 50.0,
        "effectiveness": 0.20
    },
    {
        "scenario": "Higher customer value",
        "customer_value": 800.0,
        "intervention_cost": 50.0,
        "effectiveness": 0.20
    },
    {
        "scenario": "Higher intervention cost",
        "customer_value": 500.0,
        "intervention_cost": 80.0,
        "effectiveness": 0.20
    },
    {
        "scenario": "More effective intervention",
        "customer_value": 500.0,
        "intervention_cost": 50.0,
        "effectiveness": 0.30
    },
    {
        "scenario": "Less effective intervention",
        "customer_value": 500.0,
        "intervention_cost": 50.0,
        "effectiveness": 0.10
    }
]

sensitivity_results = []

for scenario in scenarios:

    value = scenario["customer_value"]
    cost = scenario["intervention_cost"]
    effectiveness = scenario["effectiveness"]

    break_even_probability = cost / (value * effectiveness)

    for threshold in thresholds:

        targeted_probabilities = [
            p for p, y in customer_scores
            if p >= threshold
        ]

        targeted_count = len(targeted_probabilities)

        expected_benefit = sum(
            p * effectiveness * value
            for p in targeted_probabilities
        )

        total_cost = targeted_count * cost

        net_benefit = expected_benefit - total_cost

        sensitivity_results.append({
            "scenario": scenario["scenario"],
            "threshold": threshold,
            "break_even_probability": break_even_probability,
            "targeted_customers": targeted_count,
            "expected_benefit": expected_benefit,
            "intervention_cost": total_cost,
            "expected_net_benefit": net_benefit
        })

sensitivity_pd = pd.DataFrame(sensitivity_results)

# Select highest-net-benefit tested threshold per scenario
best_scenarios_pd = (
    sensitivity_pd
    .sort_values(
        ["scenario", "expected_net_benefit"],
        ascending=[True, False]
    )
    .groupby("scenario", as_index=False)
    .first()
)

display(
    best_scenarios_pd[
        [
            "scenario",
            "break_even_probability",
            "threshold",
            "targeted_customers",
            "expected_net_benefit"
        ]
    ].round(4)
)

print("Sensitivity analysis completed.")
print(f"Scenarios evaluated: {len(scenarios)}")
print(f"Thresholds per scenario: {len(thresholds)}")
print(f"Total evaluations: {len(sensitivity_results)}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 11.10 — Cost-Sensitive Threshold Optimization: Business Findings
# MAGIC
# MAGIC ### Executive Summary
# MAGIC
# MAGIC This analysis evaluated 17 churn probability thresholds, ranging from 0.10 to 0.90, using 1,409 historical customer records with model-generated churn probabilities and observed churn labels.
# MAGIC
# MAGIC The objective was to compare traditional classification performance with simulated economic value and assess how business assumptions influence retention targeting decisions.
# MAGIC
# MAGIC ### Classification Performance
# MAGIC
# MAGIC The highest F1 score was achieved at a threshold of **0.30**, with:
# MAGIC
# MAGIC - Precision: 56.67%
# MAGIC - Recall: 76.50%
# MAGIC - F1 score: 65.11%
# MAGIC - Customers flagged: 517
# MAGIC
# MAGIC The default threshold of 0.50 produced:
# MAGIC
# MAGIC - Precision: 65.90%
# MAGIC - Recall: 52.48%
# MAGIC - F1 score: 58.43%
# MAGIC - Customers flagged: 305
# MAGIC
# MAGIC These results demonstrate the trade-off between identifying more potential churners and reducing unnecessary customer outreach.
# MAGIC
# MAGIC ### Economic Optimization
# MAGIC
# MAGIC Using illustrative assumptions of $500 customer value, $50 intervention cost, and 20% intervention effectiveness, the highest simulated expected net benefit was achieved at a threshold of **0.50**.
# MAGIC
# MAGIC At this threshold:
# MAGIC
# MAGIC - Customers targeted: 305
# MAGIC - Expected benefit: $19,745.16
# MAGIC - Intervention cost: $15,250.00
# MAGIC - Expected net benefit: **$4,495.16**
# MAGIC
# MAGIC The F1-optimal threshold of 0.30 produced a lower simulated expected net benefit of $2,288.27.
# MAGIC
# MAGIC This demonstrates that optimizing predictive performance alone does not necessarily maximize economic value.
# MAGIC
# MAGIC ### Sensitivity Analysis
# MAGIC
# MAGIC Five illustrative business scenarios were evaluated.
# MAGIC
# MAGIC | Scenario | Best Tested Threshold | Expected Net Benefit |
# MAGIC |---|---:|---:|
# MAGIC | Baseline | 0.50 | $4,495.16 |
# MAGIC | Higher customer value | 0.30 | $19,171.23 |
# MAGIC | Higher intervention cost | 0.80 | $13.25 |
# MAGIC | Less effective intervention | 0.85 (no outreach) | $0.00 |
# MAGIC | More effective intervention | 0.35 | $16,437.11 |
# MAGIC
# MAGIC The results show that optimal targeting depends on customer economics, intervention costs, and expected retention effectiveness.
# MAGIC
# MAGIC ### Business Recommendations
# MAGIC
# MAGIC 1. Use cost-sensitive decision thresholds rather than relying exclusively on accuracy or F1 score.
# MAGIC 2. Calibrate predicted churn probabilities before using them to estimate expected financial value.
# MAGIC 3. Estimate intervention effectiveness using controlled experiments rather than assuming that contacting customers prevents churn.
# MAGIC 4. Evaluate different customer segments using their respective customer values and intervention costs.
# MAGIC 5. Validate threshold-selection decisions on independent data before production deployment.
# MAGIC 6. Monitor model performance, calibration, and retention economics as business conditions change.
# MAGIC
# MAGIC ### Limitations
# MAGIC
# MAGIC The financial results are simulations, not realized revenue or measured campaign outcomes.
# MAGIC
# MAGIC The analysis assumes uniform customer value, intervention cost, and retention effectiveness. It also assumes that model-generated probabilities are suitable for expected-value calculations; probability calibration has not yet been independently established.
# MAGIC
# MAGIC The same historical test dataset was used to explore classification and economic thresholds. Consequently, the findings should be treated as exploratory rather than independently validated production recommendations.
# MAGIC
# MAGIC The earlier model-evaluation notebooks also showed minor differences from the saved batch predictions, which should be reconciled for reproducibility.
# MAGIC
# MAGIC ### Conclusion
# MAGIC
# MAGIC The analysis demonstrates that machine learning decision thresholds should be selected according to the business objective.
# MAGIC
# MAGIC The threshold of 0.30 maximized F1 among the tested values, while 0.50 maximized simulated expected net benefit under the baseline assumptions.
# MAGIC
# MAGIC Sensitivity analysis further showed that changes in customer value, intervention costs, and retention effectiveness can materially change the economically preferred targeting strategy.
# MAGIC
# MAGIC This provides a foundation for more advanced customer ranking, lift analysis, and retention campaign design.
