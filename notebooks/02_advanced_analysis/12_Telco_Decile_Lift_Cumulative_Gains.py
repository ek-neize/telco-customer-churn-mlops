# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 12. Telco Customer Churn — Decile Lift and Cumulative Gains
# MAGIC
# MAGIC ## Business Objective
# MAGIC
# MAGIC Evaluate how effectively the trained churn prediction model
# MAGIC ranks customers by their likelihood of churn.
# MAGIC
# MAGIC The analysis focuses on identifying high-risk customers
# MAGIC when retention outreach resources are limited.
# MAGIC
# MAGIC ## Key Questions
# MAGIC
# MAGIC 1. What percentage of actual churners is captured
# MAGIC    within the highest-risk 10% of customers?
# MAGIC
# MAGIC 2. How many churners are captured by targeting the
# MAGIC    highest-risk 20% or 30%?
# MAGIC
# MAGIC 3. How does model-based targeting compare with
# MAGIC    random customer selection?
# MAGIC
# MAGIC 4. Which customer risk deciles have the highest
# MAGIC    observed churn rates?
# MAGIC
# MAGIC ## Planned Metrics
# MAGIC
# MAGIC - Decile churn rate
# MAGIC - Overall churn rate
# MAGIC - Decile lift
# MAGIC - Cumulative churn capture
# MAGIC - Cumulative lift
# MAGIC
# MAGIC ## Data Sources
# MAGIC
# MAGIC Predictions:
# MAGIC `workspace.default.telco_10_prediction_history`
# MAGIC
# MAGIC Actual churn outcomes:
# MAGIC `workspace.default.telco_test_raw`
# MAGIC
# MAGIC Validated batch:
# MAGIC `telco_788040999362398`
# MAGIC
# MAGIC ## Methodology
# MAGIC
# MAGIC Customers will be ranked by predicted churn probability,
# MAGIC from highest to lowest.
# MAGIC
# MAGIC They will then be divided into 10 approximately equal
# MAGIC groups (deciles).
# MAGIC
# MAGIC Decile 1 represents the highest-risk customers.
# MAGIC
# MAGIC Performance will be compared with a random-selection
# MAGIC baseline using the historical test labels.
# MAGIC
# MAGIC ## Important Limitations
# MAGIC
# MAGIC - Results reflect historical test-set performance.
# MAGIC - Actual churn labels are used for retrospective evaluation.
# MAGIC - No customer retention campaign was conducted.
# MAGIC - Lift and cumulative gains do not establish intervention
# MAGIC   effectiveness or realized financial savings.
# MAGIC - Results should be validated on independent data
# MAGIC   before operational deployment.
# MAGIC

# COMMAND ----------


# Step 12.3: Load and validate churn prediction data

from pyspark.sql import functions as F

# Select the previously validated prediction batch
BATCH_ID = "telco_788040999362398"

# Load saved predictions
predictions_df = (
    spark.table("workspace.default.telco_10_prediction_history")
    .filter(F.col("batch_id") == BATCH_ID)
    .select(
        "customerID",
        "churn_probability"
    )
)

# Load actual churn labels
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

# Join predictions with actual outcomes
decile_df = predictions_df.join(
    actual_df,
    on="customerID",
    how="inner"
)

# Validate the dataset
total_predictions = predictions_df.count()
total_actual = actual_df.count()
total_joined = decile_df.count()

unique_customers = (
    decile_df.select("customerID")
    .distinct()
    .count()
)

missing_probabilities = decile_df.filter(
    F.col("churn_probability").isNull()
).count()

missing_labels = decile_df.filter(
    F.col("actual_churn").isNull()
).count()

invalid_probabilities = decile_df.filter(
    (F.col("churn_probability") < 0) |
    (F.col("churn_probability") > 1)
).count()

# Print validation results
print("=" * 55)
print("NOTEBOOK 12 — DATA VALIDATION")
print("=" * 55)

print(f"Prediction records: {total_predictions}")
print(f"Actual records: {total_actual}")
print(f"Joined records: {total_joined}")
print(f"Unique customers: {unique_customers}")
print(f"Missing probabilities: {missing_probabilities}")
print(f"Missing labels: {missing_labels}")
print(f"Invalid probabilities: {invalid_probabilities}")

# Validation checks
assert total_predictions == 1409
assert total_actual == 1409
assert total_joined == 1409
assert unique_customers == 1409
assert missing_probabilities == 0
assert missing_labels == 0
assert invalid_probabilities == 0

print("\nDATA VALIDATION: PASS")

# Display sample records
display(
    decile_df.orderBy("customerID").limit(10)
)


# COMMAND ----------


# Step 12.4: Rank customers and assign risk deciles

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Rank customers from highest to lowest churn probability
# customerID provides deterministic tie-breaking
risk_window = Window.orderBy(
    F.col("churn_probability").desc(),
    F.col("customerID").asc()
)

# Assign customers to 10 approximately equal groups
ranked_df = (
    decile_df
    .withColumn(
        "risk_decile",
        F.ntile(10).over(risk_window)
    )
)

# Summarize each decile
decile_counts_df = (
    ranked_df
    .groupBy("risk_decile")
    .agg(
        F.count("*").alias("customers"),
        F.min("churn_probability").alias("min_probability"),
        F.max("churn_probability").alias("max_probability"),
        F.sum("actual_churn").alias("actual_churners")
    )
    .orderBy("risk_decile")
)

# Display results
display(decile_counts_df)

# Validate decile assignments
total_ranked = ranked_df.count()
total_deciles = ranked_df.select("risk_decile").distinct().count()

decile_sizes = [
    row["customers"]
    for row in decile_counts_df.select("customers").collect()
]

assert total_ranked == 1409
assert total_deciles == 10
assert sum(decile_sizes) == 1409
assert max(decile_sizes) - min(decile_sizes) <= 1

print("=" * 55)
print("DECILE ASSIGNMENT VALIDATION")
print("=" * 55)
print(f"Total ranked customers: {total_ranked}")
print(f"Number of deciles: {total_deciles}")
print(f"Smallest decile: {min(decile_sizes)}")
print(f"Largest decile: {max(decile_sizes)}")
print("\nDECILE ASSIGNMENT: PASS")


# COMMAND ----------


# Step 12.5: Calculate decile churn rates and lift

from pyspark.sql import functions as F

# Overall customer and churn counts
overall_stats = ranked_df.agg(
    F.count("*").alias("total_customers"),
    F.sum("actual_churn").alias("total_churners"),
    F.avg("actual_churn").alias("overall_churn_rate")
).first()

total_customers = overall_stats["total_customers"]
total_churners = overall_stats["total_churners"]
overall_churn_rate = overall_stats["overall_churn_rate"]

print("=" * 55)
print("OVERALL CHURN STATISTICS")
print("=" * 55)

print(f"Total customers: {total_customers}")
print(f"Actual churners: {total_churners}")
print(f"Overall churn rate: {overall_churn_rate:.2%}")

# Calculate decile-level churn rates and lift
decile_lift_df = (
    ranked_df
    .groupBy("risk_decile")
    .agg(
        F.count("*").alias("customers"),
        F.sum("actual_churn").alias("actual_churners"),
        F.avg("actual_churn").alias("decile_churn_rate"),
        F.avg("churn_probability").alias("avg_churn_probability")
    )
    .withColumn(
        "decile_lift",
        F.col("decile_churn_rate") / F.lit(overall_churn_rate)
    )
    .orderBy("risk_decile")
)

# Display decile lift results
display(decile_lift_df)

# Validate totals
validation = decile_lift_df.agg(
    F.sum("customers").alias("customers"),
    F.sum("actual_churners").alias("churners")
).first()

assert overall_churn_rate > 0
assert validation["customers"] == total_customers
assert validation["churners"] == total_churners
assert decile_lift_df.count() == 10

print("\nDECILE LIFT VALIDATION: PASS")


# COMMAND ----------


# Step 12.6: Cumulative gains and cumulative lift

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Define cumulative window from highest to lowest risk
cumulative_window = (
    Window
    .orderBy("risk_decile")
    .rowsBetween(
        Window.unboundedPreceding,
        Window.currentRow
    )
)

# Calculate cumulative metrics
cumulative_gains_df = (
    decile_lift_df
    .withColumn(
        "cumulative_customers",
        F.sum("customers").over(cumulative_window)
    )
    .withColumn(
        "cumulative_churners",
        F.sum("actual_churners").over(cumulative_window)
    )
    .withColumn(
        "customer_targeting_pct",
        F.col("cumulative_customers") / F.lit(total_customers)
    )
    .withColumn(
        "cumulative_gains",
        F.col("cumulative_churners") / F.lit(total_churners)
    )
    .withColumn(
        "cumulative_lift",
        F.col("cumulative_gains") /
        F.col("customer_targeting_pct")
    )
    .orderBy("risk_decile")
)

# Display results
display(
    cumulative_gains_df.select(
        "risk_decile",
        "cumulative_customers",
        "cumulative_churners",
        "customer_targeting_pct",
        "cumulative_gains",
        "cumulative_lift"
    )
)

# Validate final cumulative values
final_row = (
    cumulative_gains_df
    .filter(F.col("risk_decile") == 10)
    .first()
)

assert final_row["cumulative_customers"] == total_customers
assert final_row["cumulative_churners"] == total_churners
assert abs(final_row["cumulative_gains"] - 1.0) < 1e-9
assert abs(final_row["cumulative_lift"] - 1.0) < 1e-9

print("=" * 55)
print("CUMULATIVE GAINS VALIDATION")
print("=" * 55)
print(f"Total customers: {total_customers}")
print(f"Total churners: {total_churners}")
print(f"Final cumulative gains: {final_row['cumulative_gains']:.2%}")
print(f"Final cumulative lift: {final_row['cumulative_lift']:.2f}x")
print("\nCUMULATIVE GAINS VALIDATION: PASS")


# COMMAND ----------


# Step 12.7: Visualize cumulative gains and lift

import matplotlib.pyplot as plt
import pandas as pd

# Convert the 10-row Spark summary to Pandas
plot_pd = (
    cumulative_gains_df
    .select(
        "risk_decile",
        "customer_targeting_pct",
        "cumulative_gains",
        "cumulative_lift"
    )
    .orderBy("risk_decile")
    .toPandas()
)

# Add origin for cumulative gains chart
gains_pd = pd.concat(
    [
        pd.DataFrame({
            "customer_targeting_pct": [0.0],
            "cumulative_gains": [0.0]
        }),
        plot_pd[
            ["customer_targeting_pct", "cumulative_gains"]
        ]
    ],
    ignore_index=True
)

# Chart 1: Cumulative Gains
plt.figure(figsize=(10, 6))

plt.plot(
    gains_pd["customer_targeting_pct"] * 100,
    gains_pd["cumulative_gains"] * 100,
    marker="o",
    linewidth=2,
    label="Churn prediction model"
)

plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--",
    linewidth=2,
    label="Random targeting baseline"
)

plt.title("Telco Churn — Cumulative Gains")
plt.xlabel("Customers Targeted (%)")
plt.ylabel("Actual Churners Captured (%)")
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Chart 2: Cumulative Lift
plt.figure(figsize=(10, 6))

plt.plot(
    plot_pd["customer_targeting_pct"] * 100,
    plot_pd["cumulative_lift"],
    marker="o",
    linewidth=2,
    label="Model cumulative lift"
)

plt.axhline(
    y=1.0,
    linestyle="--",
    linewidth=2,
    label="Random targeting baseline"
)

plt.title("Telco Churn — Cumulative Lift")
plt.xlabel("Customers Targeted (%)")
plt.ylabel("Cumulative Lift")
plt.xlim(0, 100)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("Cumulative gains and lift charts generated successfully.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 12.8 Business Interpretation and Retention Targeting Recommendations
# MAGIC
# MAGIC ### Executive Summary
# MAGIC
# MAGIC The Telco Customer Churn model demonstrates strong customer risk-ranking performance on the historical test dataset.
# MAGIC
# MAGIC Using 1,409 customer records containing 383 actual churners, decile lift and cumulative gains analysis show that the model concentrates a substantial proportion of churners within the highest-risk customer groups.
# MAGIC
# MAGIC The highest-risk 10% of customers contains approximately 27.68% of all observed churners, achieving a cumulative lift of 2.77 compared with random targeting.
# MAGIC
# MAGIC This suggests that model-based customer prioritization could improve the efficiency of retention outreach when resources are limited.
# MAGIC
# MAGIC ### 1. Decile Lift Findings
# MAGIC
# MAGIC The model successfully ranks customers according to their predicted probability of churn.
# MAGIC
# MAGIC Key findings:
# MAGIC
# MAGIC - Decile 1 contains 141 customers, of whom 106 actually churned.
# MAGIC - The observed churn rate in Decile 1 is 75.18%.
# MAGIC - The overall observed churn rate is 27.18%.
# MAGIC - Decile 1 achieves a lift of 2.77.
# MAGIC - Decile 2 achieves a lift of 2.17.
# MAGIC - Decile 3 achieves a lift of 1.67.
# MAGIC
# MAGIC The declining lift across deciles indicates that the model effectively distinguishes higher-risk customers from lower-risk customers.
# MAGIC
# MAGIC ### 2. Cumulative Gains Findings
# MAGIC
# MAGIC | Customers Targeted | Actual Churners Identified | Churn Capture Rate | Cumulative Lift |
# MAGIC |---|---:|---:|---:|
# MAGIC | Top 10% | 106 | 27.68% | 2.77 |
# MAGIC | Top 20% | 189 | 49.35% | 2.47 |
# MAGIC | Top 30% | 253 | 66.06% | 2.20 |
# MAGIC | Top 40% | 304 | 79.37% | 1.98 |
# MAGIC | Top 50% | 336 | 87.73% | 1.75 |
# MAGIC
# MAGIC These findings demonstrate that targeting customers by predicted churn risk is more effective at identifying historical churners than random customer selection.
# MAGIC
# MAGIC ### 3. Business Recommendations
# MAGIC
# MAGIC **Recommendation 1: Use risk-based customer prioritization**
# MAGIC
# MAGIC Rank customers by predicted churn probability rather than selecting customers randomly.
# MAGIC
# MAGIC This can help retention teams focus their limited resources on customers with the highest estimated churn risk.
# MAGIC
# MAGIC **Recommendation 2: Align targeting with outreach capacity**
# MAGIC
# MAGIC When outreach capacity is limited, prioritize customers in the highest-risk deciles.
# MAGIC
# MAGIC For example:
# MAGIC
# MAGIC - At approximately 10% outreach capacity, prioritize Decile 1.
# MAGIC - At approximately 20% capacity, prioritize Deciles 1–2.
# MAGIC - At approximately 30% capacity, prioritize Deciles 1–3.
# MAGIC
# MAGIC These are capacity-based targeting examples, not proof that these outreach levels maximize financial returns.
# MAGIC
# MAGIC **Recommendation 3: Combine lift analysis with cost-sensitive optimization**
# MAGIC
# MAGIC Notebook 11 demonstrated that the threshold maximizing classification performance may differ from the threshold maximizing simulated economic value.
# MAGIC
# MAGIC Under the illustrative baseline assumptions used in Notebook 11, the economically preferred probability threshold was 0.50.
# MAGIC
# MAGIC Therefore, decile targeting should be considered alongside intervention costs, customer value, retention effectiveness, and available outreach capacity.
# MAGIC
# MAGIC **Recommendation 4: Validate intervention effectiveness**
# MAGIC
# MAGIC High churn risk does not necessarily mean a customer is likely to respond positively to a retention offer.
# MAGIC
# MAGIC A controlled experiment should be used to estimate whether interventions reduce churn and generate incremental business value.
# MAGIC
# MAGIC ### 4. Model Monitoring Recommendations
# MAGIC
# MAGIC The following metrics should be monitored as additional labeled customer outcomes become available:
# MAGIC
# MAGIC - Decile-level observed churn rates.
# MAGIC - Lift in the highest-risk deciles.
# MAGIC - Cumulative gains and cumulative lift.
# MAGIC - Changes in churn probability distributions.
# MAGIC - Changes in model calibration.
# MAGIC - Differences in performance across customer segments.
# MAGIC
# MAGIC Monitoring these metrics can help identify deterioration in model ranking performance.
# MAGIC
# MAGIC ### 5. Limitations
# MAGIC
# MAGIC - The analysis uses 1,409 historical test records.
# MAGIC - All 383 churn outcomes are historical observations.
# MAGIC - No retention intervention was performed.
# MAGIC - Lift measures risk-ranking effectiveness, not causal intervention impact.
# MAGIC - Results may change when applied to new customer populations.
# MAGIC - Decile boundaries depend on the scored customer population.
# MAGIC - Independent or future-period validation is recommended before deployment.
# MAGIC
# MAGIC ### 6. Conclusion
# MAGIC
# MAGIC The Telco Customer Churn model demonstrates useful risk-ranking capability.
# MAGIC
# MAGIC Targeting approximately 20% of customers identifies 49.35% of historical churners, while targeting approximately 30% identifies 66.06%.
# MAGIC
# MAGIC These findings support using churn probabilities to prioritize retention outreach, subject to business constraints and experimental validation.
# MAGIC
# MAGIC Combined with the cost-sensitive threshold optimization developed in Notebook 11, the decile lift and cumulative gains analysis provides a foundation for evaluating customer retention targeting strategies.
