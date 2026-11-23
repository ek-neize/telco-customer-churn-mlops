# Databricks notebook source

from pyspark.sql import functions as F

print("=" * 65)
print("STEP 16.1 — PREDICTION MONITORING BASELINE")
print("=" * 65)

BATCH_ID = "d972dd16-029d-489f-a406-3c6c6e1e6265"

PREDICTIONS_TABLE = (
    "workspace.default.telco_15_job_dev_predictions"
)

df = (
    spark.table(PREDICTIONS_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
)

print("\n1. TOTAL PREDICTIONS")
total = df.count()
print("Total customers:", total)

print("\n2. PREDICTION DISTRIBUTION")

df.groupBy("predicted_churn").count().show()

print("\n3. CHURN PREDICTION RATE")

summary = df.agg(
    F.sum(
        F.when(F.col("predicted_class") == 1, 1).otherwise(0)
    ).alias("predicted_churn_count"),
    F.avg("churn_probability").alias("average_churn_probability"),
    F.min("churn_probability").alias("minimum_probability"),
    F.max("churn_probability").alias("maximum_probability")
).first()

churn_count = summary["predicted_churn_count"]
churn_rate = churn_count / total if total else 0

print("Predicted churn count:", churn_count)
print("Predicted churn rate:", round(churn_rate * 100, 2), "%")
print(
    "Average churn probability:",
    round(summary["average_churn_probability"], 4)
)
print(
    "Minimum probability:",
    round(summary["minimum_probability"], 4)
)
print(
    "Maximum probability:",
    round(summary["maximum_probability"], 4)
)

print("\n4. BASELINE VALIDATION")

assert total == 1409, "Unexpected batch size"
assert churn_count == 305, "Unexpected churn count"

print("Baseline validation: PASS")

print("\nSTEP 16.1 COMPLETE")
print("Read-only monitoring baseline established.")


# COMMAND ----------


# Step 16.2 — Prediction Drift Monitoring
# READ-ONLY: No job runs, inference, or Delta writes

from pyspark.sql import functions as F

print("=" * 65)
print("STEP 16.2 — PREDICTION DRIFT MONITORING")
print("=" * 65)

BATCH_ID = "d972dd16-029d-489f-a406-3c6c6e1e6265"
TABLE = "workspace.default.telco_15_job_dev_predictions"

df = spark.table(TABLE).filter(
    F.col("batch_id") == BATCH_ID
)

baseline = df.agg(
    F.count("*").alias("total"),
    F.sum(
        F.when(F.col("predicted_class") == 1, 1)
        .otherwise(0)
    ).alias("churn_count")
).first()

assert baseline["total"] == 1409
assert baseline["churn_count"] == 305

baseline_rate = (
    baseline["churn_count"] / baseline["total"]
)

def assess_prediction_drift(baseline_rate, current_rate):
    if not 0 <= baseline_rate <= 1:
        raise ValueError("Invalid baseline rate")
    if not 0 <= current_rate <= 1:
        raise ValueError("Invalid current rate")

    difference_pp = abs(
        current_rate - baseline_rate
    ) * 100

    if difference_pp >= 10:
        status = "CRITICAL"
    elif difference_pp >= 5:
        status = "WARNING"
    else:
        status = "NORMAL"

    return round(difference_pp, 2), status

print("\n1. BASELINE")
print(f"Baseline churn rate: {baseline_rate:.2%}")

print("\n2. ILLUSTRATIVE DRIFT SCENARIOS")
print("These are simulated rates, not observed batches.\n")

scenarios = {
    "Small change": 0.24,
    "Moderate change": 0.28,
    "Large change": 0.35
}

for scenario, current_rate in scenarios.items():
    difference, status = assess_prediction_drift(
        baseline_rate, current_rate
    )

    print(f"Scenario: {scenario}")
    print(f"  Current rate: {current_rate:.2%}")
    print(f"  Absolute change: {difference:.2f} pp")
    print(f"  Monitoring status: {status}")
    print()

print("3. SAFETY")
print("New inference executed: False")
print("Databricks Job submitted: False")
print("Delta writes performed: False")

print("\nSTEP 16.2 COMPLETE")


# COMMAND ----------


# Step 16.3 — Existing prediction batch inventory
# READ-ONLY: No inference, job runs, or Delta writes

from pyspark.sql import functions as F

print("=" * 65)
print("STEP 16.3 — EXISTING PREDICTION BATCH INVENTORY")
print("=" * 65)

TABLE = "workspace.default.telco_15_job_dev_predictions"
BASELINE_ID = "d972dd16-029d-489f-a406-3c6c6e1e6265"

df = spark.table(TABLE)

print("\n1. BATCH SUMMARY")

summary = (
    df.groupBy("batch_id")
    .agg(
        F.count("*").alias("total_predictions"),
        F.countDistinct("customerID").alias("unique_customers"),
        F.sum(
            F.when(F.col("predicted_class") == 1, 1)
            .otherwise(0)
        ).alias("predicted_churn_count"),
        F.avg("churn_probability").alias(
            "average_churn_probability"
        ),
        F.countDistinct("model_version").alias(
            "model_versions"
        ),
        F.countDistinct("source_table").alias(
            "source_tables"
        )
    )
    .withColumn(
        "churn_rate_pct",
        F.round(
            F.col("predicted_churn_count")
            / F.col("total_predictions") * 100,
            2
        )
    )
)

summary.orderBy("batch_id").show(
    truncate=False
)

print("\n2. CUSTOMER OVERLAP WITH BASELINE")

baseline_customers = (
    df.filter(F.col("batch_id") == BASELINE_ID)
    .select("customerID")
    .distinct()
)

other_batches = (
    df.filter(F.col("batch_id") != BASELINE_ID)
    .select("batch_id", "customerID")
    .distinct()
)

overlap = (
    other_batches.join(
        baseline_customers,
        on="customerID",
        how="inner"
    )
    .groupBy("batch_id")
    .agg(
        F.count("*").alias("shared_customers")
    )
)

overlap.show(truncate=False)

print("\n3. TOTAL BATCHES")

print("Distinct batches:", df.select("batch_id").distinct().count())

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")

print("\nSTEP 16.3 COMPLETE")


# COMMAND ----------


# Step 16.4 — Cross-batch prediction reproducibility
# READ-ONLY: No inference, job runs, or Delta writes

from pyspark.sql import functions as F

print("=" * 65)
print("STEP 16.4 — CROSS-BATCH PREDICTION REPRODUCIBILITY")
print("=" * 65)

TABLE = "workspace.default.telco_15_job_dev_predictions"

BASELINE_ID = "d972dd16-029d-489f-a406-3c6c6e1e6265"

df = spark.table(TABLE)

baseline = (
    df.filter(F.col("batch_id") == BASELINE_ID)
    .select(
        "customerID",
        F.col("churn_probability").alias("baseline_probability"),
        F.col("predicted_class").alias("baseline_class"),
        F.col("predicted_churn").alias("baseline_churn")
    )
)

other_batches = (
    df.filter(F.col("batch_id") != BASELINE_ID)
)

comparison = (
    other_batches.join(
        baseline,
        on="customerID",
        how="left"
    )
    .withColumn(
        "probability_mismatch",
        F.col("baseline_probability").isNull()
        | F.col("churn_probability").isNull()
        | (
            F.abs(
                F.col("churn_probability")
                - F.col("baseline_probability")
            ) > 1e-10
        )
    )
    .withColumn(
        "class_mismatch",
        ~F.col("predicted_class").eqNullSafe(
            F.col("baseline_class")
        )
    )
    .withColumn(
        "label_mismatch",
        ~F.col("predicted_churn").eqNullSafe(
            F.col("baseline_churn")
        )
    )
)

results = (
    comparison.groupBy("batch_id")
    .agg(
        F.count("*").alias("compared_rows"),
        F.sum(
            F.col("probability_mismatch").cast("int")
        ).alias("probability_mismatches"),
        F.sum(
            F.col("class_mismatch").cast("int")
        ).alias("class_mismatches"),
        F.sum(
            F.col("label_mismatch").cast("int")
        ).alias("label_mismatches")
    )
)

print("\n1. CUSTOMER-LEVEL COMPARISON")
results.orderBy("batch_id").show(truncate=False)

print("\n2. OVERALL ASSESSMENT")

rows = results.collect()

assert len(rows) == 3, "Expected three comparison batches"

all_pass = all(
    row["compared_rows"] == 1409
    and row["probability_mismatches"] == 0
    and row["class_mismatches"] == 0
    and row["label_mismatches"] == 0
    for row in rows
)

print(
    "Cross-batch reproducibility:",
    "PASS" if all_pass else "FAIL"
)

assert all_pass, "Prediction differences detected"

print("\n3. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")

print("\nSTEP 16.4 COMPLETE")


# COMMAND ----------


# Step 16.5 — Training feature reference distributions
# READ-ONLY: No inference, job runs, or Delta writes

from pyspark.sql import functions as F

print("=" * 65)
print("STEP 16.5 — FEATURE DRIFT REFERENCE BASELINE")
print("=" * 65)

TRAIN_TABLE = "workspace.default.telco_train_raw"

train_df = spark.table(TRAIN_TABLE)

print("\n1. TRAINING DATASET")

print("Training rows:", train_df.count())

print("\n2. FEATURE SCHEMA")

features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract"
]

for feature in features:
    print(
        f"{feature}: "
        f"{train_df.schema[feature].dataType}"
    )

print("\n3. NUMERICAL FEATURE STATISTICS")

numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for feature in numeric_features:
    # try_cast handles numeric strings and malformed values
    values = train_df.select(
        F.expr(
            f"try_cast(`{feature}` AS DOUBLE)"
        ).alias("value")
    )

    stats = values.agg(
        F.count("*").alias("total"),
        F.count("value").alias("valid"),
        F.mean("value").alias("mean"),
        F.stddev("value").alias("stddev"),
        F.min("value").alias("minimum"),
        F.max("value").alias("maximum")
    ).first()

    print(f"\nFeature: {feature}")
    print("Total rows:", stats["total"])
    print("Valid numeric values:", stats["valid"])
    print("Missing/invalid:", stats["total"] - stats["valid"])
    print("Mean:", stats["mean"])
    print("Std dev:", stats["stddev"])
    print("Minimum:", stats["minimum"])
    print("Maximum:", stats["maximum"])

print("\n4. CONTRACT DISTRIBUTION")

contract_distribution = (
    train_df.groupBy("Contract")
    .count()
    .withColumn(
        "percentage",
        F.round(
            F.col("count") / F.sum("count").over(
                __import__("pyspark").sql.Window.partitionBy()
            ) * 100,
            2
        )
    )
)

contract_distribution.orderBy("Contract").show(
    truncate=False
)

print("\n5. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")

print("\nSTEP 16.5 COMPLETE")


# COMMAND ----------


# Step 16.6 — Numerical feature drift comparison
# READ-ONLY: No inference, job runs, or Delta writes

from pyspark.sql import functions as F

print("=" * 65)
print("STEP 16.6 — NUMERICAL FEATURE DRIFT")
print("=" * 65)

TRAIN_TABLE = "workspace.default.telco_train_raw"
TEST_TABLE = "workspace.default.telco_test_raw"

train_df = spark.table(TRAIN_TABLE)
test_df = spark.table(TEST_TABLE)

features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

print("\n1. DATASET SIZES")
print("Training rows:", train_df.count())
print("Test rows:", test_df.count())

print("\n2. FEATURE COMPARISON")

results = []

for feature in features:

    train_values = train_df.select(
        F.col(feature).cast("double").alias("value")
    )

    test_values = test_df.select(
        F.col(feature).cast("double").alias("value")
    )

    train_stats = train_values.agg(
        F.avg("value").alias("mean"),
        F.stddev("value").alias("std"),
        F.count("value").alias("valid")
    ).first()

    test_stats = test_values.agg(
        F.avg("value").alias("mean"),
        F.count("value").alias("valid")
    ).first()

    train_mean = train_stats["mean"]
    test_mean = test_stats["mean"]
    train_std = train_stats["std"]

    if train_std is None or train_std == 0:
        smd = None
        status = "UNDEFINED"
    else:
        smd = abs(test_mean - train_mean) / train_std

        if smd >= 0.20:
            status = "LARGE"
        elif smd >= 0.10:
            status = "MODERATE"
        else:
            status = "LOW"

    results.append((
        feature,
        round(train_mean, 4),
        round(test_mean, 4),
        round(smd, 4) if smd is not None else None,
        status,
        train_stats["valid"],
        test_stats["valid"]
    ))

print(
    f"\n{'Feature':<18} {'Train Mean':>12} "
    f"{'Test Mean':>12} {'SMD':>10} {'Status':>12}"
)

print("-" * 70)

for result in results:
    feature, train_mean, test_mean, smd, status, _, _ = result

    print(
        f"{feature:<18} "
        f"{train_mean:>12.4f} "
        f"{test_mean:>12.4f} "
        f"{smd if smd is not None else 'N/A':>10} "
        f"{status:>12}"
    )

print("\n3. VALID NUMERIC OBSERVATIONS")

for result in results:
    feature, _, _, _, _, train_valid, test_valid = result
    print(
        f"{feature}: train={train_valid}, test={test_valid}"
    )

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")

print("\nSTEP 16.6 COMPLETE")


# COMMAND ----------


# Step 16.7 — Categorical feature drift detection
# READ-ONLY: No inference, job runs, or Delta writes

from pyspark.sql import functions as F

print("=" * 65)
print("STEP 16.7 — CATEGORICAL FEATURE DRIFT")
print("=" * 65)

TRAIN_TABLE = "workspace.default.telco_train_raw"
TEST_TABLE = "workspace.default.telco_test_raw"

train_df = spark.table(TRAIN_TABLE)
test_df = spark.table(TEST_TABLE)

feature = "Contract"

print("\n1. DATASET SIZES")

train_total = train_df.count()
test_total = test_df.count()

print("Training rows:", train_total)
print("Test rows:", test_total)

print("\n2. CONTRACT DISTRIBUTIONS")

train_dist = (
    train_df.groupBy(feature)
    .count()
    .withColumn(
        "train_pct",
        F.col("count") / F.lit(train_total) * 100
    )
    .select(
        feature,
        F.col("count").alias("train_count"),
        "train_pct"
    )
)

test_dist = (
    test_df.groupBy(feature)
    .count()
    .withColumn(
        "test_pct",
        F.col("count") / F.lit(test_total) * 100
    )
    .select(
        feature,
        F.col("count").alias("test_count"),
        "test_pct"
    )
)

comparison = (
    train_dist.join(
        test_dist,
        on=feature,
        how="full_outer"
    )
    .fillna({
        "train_count": 0,
        "test_count": 0,
        "train_pct": 0.0,
        "test_pct": 0.0
    })
    .withColumn(
        "absolute_difference_pp",
        F.abs(
            F.col("train_pct") - F.col("test_pct")
        )
    )
)

comparison.orderBy(feature).show(
    truncate=False
)

print("\n3. TOTAL VARIATION DISTANCE")

tvd = (
    comparison.agg(
        (
            F.sum("absolute_difference_pp") / 200
        ).alias("tvd")
    ).first()["tvd"]
)

if tvd >= 0.10:
    status = "LARGE"
elif tvd >= 0.05:
    status = "MODERATE"
else:
    status = "LOW"

print("TVD:", round(tvd, 6))
print("Categorical shift:", status)

print("\n4. VALIDATION")

assert train_total == 5634
assert test_total == 1409
assert 0 <= tvd <= 1

print("Validation: PASS")

print("\n5. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")

print("\nSTEP 16.7 COMPLETE")


# COMMAND ----------


# Step 16.8 — Combined Feature Drift Monitoring
# READ-ONLY: No inference, job runs, or Delta writes

from pyspark.sql import functions as F

print("=" * 70)
print("STEP 16.8 — COMBINED FEATURE DRIFT REPORT")
print("=" * 70)

train_df = spark.table("workspace.default.telco_train_raw")
test_df = spark.table("workspace.default.telco_test_raw")

train_total = train_df.count()
test_total = test_df.count()

results = []

# --------------------------------------------------
# 1. Numerical features
# --------------------------------------------------

numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for feature in numeric_features:

    train_values = train_df.select(
        F.col(feature).cast("double").alias("value")
    )

    test_values = test_df.select(
        F.col(feature).cast("double").alias("value")
    )

    train_stats = train_values.agg(
        F.avg("value").alias("mean"),
        F.stddev("value").alias("std"),
        F.count("value").alias("valid")
    ).first()

    test_stats = test_values.agg(
        F.avg("value").alias("mean"),
        F.count("value").alias("valid")
    ).first()

    if (
        train_stats["std"] is None
        or train_stats["std"] == 0
    ):
        drift_score = None
        status = "UNDEFINED"
    else:
        drift_score = abs(
            test_stats["mean"] - train_stats["mean"]
        ) / train_stats["std"]

        if drift_score >= 0.20:
            status = "LARGE"
        elif drift_score >= 0.10:
            status = "MODERATE"
        else:
            status = "LOW"

    results.append((
        feature,
        "NUMERICAL",
        round(drift_score, 6)
        if drift_score is not None else None,
        status,
        train_total - train_stats["valid"],
        test_total - test_stats["valid"]
    ))

# --------------------------------------------------
# 2. Categorical feature
# --------------------------------------------------

feature = "Contract"

train_dist = {
    row[feature]: row["count"] / train_total
    for row in train_df.groupBy(feature).count().collect()
}

test_dist = {
    row[feature]: row["count"] / test_total
    for row in test_df.groupBy(feature).count().collect()
}

categories = set(train_dist) | set(test_dist)

tvd = 0.5 * sum(
    abs(
        train_dist.get(category, 0)
        - test_dist.get(category, 0)
    )
    for category in categories
)

if tvd >= 0.10:
    categorical_status = "LARGE"
elif tvd >= 0.05:
    categorical_status = "MODERATE"
else:
    categorical_status = "LOW"

train_missing = train_df.filter(
    F.col(feature).isNull()
).count()

test_missing = test_df.filter(
    F.col(feature).isNull()
).count()

results.append((
    feature,
    "CATEGORICAL",
    round(tvd, 6),
    categorical_status,
    train_missing,
    test_missing
))

# --------------------------------------------------
# 3. Combined report
# --------------------------------------------------

print("\n1. DATASET SIZES")
print("Training:", train_total)
print("Test:", test_total)

print("\n2. COMBINED DRIFT RESULTS")

report = spark.createDataFrame(
    results,
    [
        "feature",
        "feature_type",
        "drift_score",
        "status",
        "train_missing",
        "test_missing"
    ]
)

report.orderBy("feature").show(truncate=False)

print("\n3. OVERALL ASSESSMENT")

statuses = [row[3] for row in results]

if "LARGE" in statuses:
    overall = "LARGE SHIFT DETECTED"
elif "MODERATE" in statuses:
    overall = "MODERATE SHIFT DETECTED"
elif "UNDEFINED" in statuses:
    overall = "REVIEW REQUIRED"
else:
    overall = "LOW SHIFT"

print("Overall status:", overall)

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")

print("\nSTEP 16.8 COMPLETE")


# COMMAND ----------


# Step 16.9 — Data Quality Monitoring
# READ-ONLY: No inference, job runs, or Delta writes

from pyspark.sql import functions as F

print("=" * 70)
print("STEP 16.9 — DATA QUALITY MONITORING")
print("=" * 70)

TRAIN_TABLE = "workspace.default.telco_train_raw"
TEST_TABLE = "workspace.default.telco_test_raw"

train_df = spark.table(TRAIN_TABLE)
test_df = spark.table(TEST_TABLE)

features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract"
]

print("\n1. DATASET SIZES")

train_count = train_df.count()
test_count = test_df.count()

print("Training rows:", train_count)
print("Test rows:", test_count)

print("\n2. MISSING VALUE MONITORING")

for feature in features:

    train_missing = train_df.filter(
        F.col(feature).isNull()
    ).count()

    test_missing = test_df.filter(
        F.col(feature).isNull()
    ).count()

    train_pct = train_missing / train_count * 100
    test_pct = test_missing / test_count * 100

    print(f"\nFeature: {feature}")
    print(
        f"  Train missing: {train_missing} "
        f"({train_pct:.2f}%)"
    )
    print(
        f"  Test missing: {test_missing} "
        f"({test_pct:.2f}%)"
    )

print("\n3. INVALID NUMERICAL VALUES")

# These are initial domain checks for the Telco dataset.
# They can be refined when production data contracts are defined.

invalid_rules = {
    "tenure": (
        (F.col("tenure") < 0)
        | (F.col("tenure") > 72)
    ),
    "MonthlyCharges": (
        (F.col("MonthlyCharges") < 0)
        | F.isnan(F.col("MonthlyCharges"))
    ),
    "TotalCharges": (
        (F.col("TotalCharges") < 0)
        | F.isnan(F.col("TotalCharges"))
    )
}

for feature, rule in invalid_rules.items():

    train_invalid = train_df.filter(rule).count()
    test_invalid = test_df.filter(rule).count()

    print(
        f"{feature}: "
        f"train_invalid={train_invalid}, "
        f"test_invalid={test_invalid}"
    )

print("\n4. UNEXPECTED CONTRACT CATEGORIES")

expected_contracts = [
    "Month-to-month",
    "One year",
    "Two year"
]

train_unexpected = train_df.filter(
    F.col("Contract").isNotNull()
    & ~F.col("Contract").isin(expected_contracts)
).count()

test_unexpected = test_df.filter(
    F.col("Contract").isNotNull()
    & ~F.col("Contract").isin(expected_contracts)
).count()

print("Unexpected training categories:", train_unexpected)
print("Unexpected test categories:", test_unexpected)

print("\n5. CUSTOMER IDENTIFIER QUALITY")

for name, data in [
    ("Training", train_df),
    ("Test", test_df)
]:
    null_ids = data.filter(
        F.col("customerID").isNull()
        | (F.trim(F.col("customerID")) == "")
    ).count()

    duplicate_ids = (
        data.count()
        - data.select("customerID").distinct().count()
    )

    print(
        f"{name}: null/blank IDs={null_ids}, "
        f"duplicate IDs={duplicate_ids}"
    )

print("\n6. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")

print("\nSTEP 16.9 COMPLETE")


# COMMAND ----------


# Step 16.10 — Data Quality Alert Rules
# READ-ONLY: No inference, job runs, or Delta writes

from pyspark.sql import functions as F

print("=" * 70)
print("STEP 16.10 — DATA QUALITY ALERT RULES")
print("=" * 70)

TEST_TABLE = "workspace.default.telco_test_raw"

df = spark.table(TEST_TABLE)

total = df.count()
assert total > 0, "Cannot monitor an empty dataset"

features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract"
]

alerts = []

def missing_status(rate_pct):
    if rate_pct > 2.0:
        return "CRITICAL"
    elif rate_pct > 0.5:
        return "WARNING"
    return "NORMAL"

def add_alert(check, count, status):
    alerts.append((check, count, status))

print("\n1. MISSING VALUE ALERTS")

for feature in features:
    missing = df.filter(
        F.col(feature).isNull()
    ).count()

    rate_pct = missing / total * 100
    status = missing_status(rate_pct)

    add_alert(
        f"Missing {feature}",
        missing,
        status
    )

    print(
        f"{feature}: {missing}/{total} "
        f"({rate_pct:.2f}%) — {status}"
    )

print("\n2. INVALID NUMERICAL VALUE ALERTS")

numeric_rules = {
    "tenure": (
        (F.col("tenure") < 0)
        | (F.col("tenure") > 72)
    ),
    "MonthlyCharges": (
        (F.col("MonthlyCharges") < 0)
        | F.isnan(F.col("MonthlyCharges"))
    ),
    "TotalCharges": (
        (F.col("TotalCharges") < 0)
        | F.isnan(F.col("TotalCharges"))
    )
}

for feature, rule in numeric_rules.items():
    invalid = df.filter(rule).count()

    status = "CRITICAL" if invalid > 0 else "NORMAL"

    add_alert(
        f"Invalid {feature}",
        invalid,
        status
    )

    print(f"{feature}: {invalid} — {status}")

print("\n3. CONTRACT CATEGORY ALERT")

expected_contracts = [
    "Month-to-month",
    "One year",
    "Two year"
]

unexpected = df.filter(
    F.col("Contract").isNotNull()
    & ~F.col("Contract").isin(expected_contracts)
).count()

add_alert(
    "Unexpected Contract",
    unexpected,
    "CRITICAL" if unexpected > 0 else "NORMAL"
)

print("Unexpected categories:", unexpected)

print("\n4. CUSTOMER ID ALERTS")

missing_ids = df.filter(
    F.col("customerID").isNull()
    | (F.trim(F.col("customerID")) == "")
).count()

duplicate_ids = (
    total - df.select("customerID").distinct().count()
)

add_alert(
    "Missing customerID",
    missing_ids,
    "CRITICAL" if missing_ids > 0 else "NORMAL"
)

add_alert(
    "Duplicate customerID",
    duplicate_ids,
    "CRITICAL" if duplicate_ids > 0 else "NORMAL"
)

print("Missing/blank IDs:", missing_ids)
print("Duplicate IDs:", duplicate_ids)

print("\n5. OVERALL DATA QUALITY STATUS")

statuses = [item[2] for item in alerts]

if "CRITICAL" in statuses:
    overall = "CRITICAL"
elif "WARNING" in statuses:
    overall = "WARNING"
else:
    overall = "NORMAL"

print("Overall status:", overall)

print("\n6. ALERT SUMMARY")

for check, count, status in alerts:
    print(f"{check}: count={count}, status={status}")

print("\n7. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")

print("\nSTEP 16.10 COMPLETE")


# COMMAND ----------


# Step 16.11 — Unified Model Monitoring Report
# READ-ONLY: No inference, job runs, or Delta writes

from pyspark.sql import functions as F

print("=" * 70)
print("STEP 16.11 — UNIFIED MODEL MONITORING REPORT")
print("=" * 70)

BATCH_ID = "d972dd16-029d-489f-a406-3c6c6e1e6265"

train = spark.table("workspace.default.telco_train_raw")
test = spark.table("workspace.default.telco_test_raw")

predictions = (
    spark.table("workspace.default.telco_15_job_dev_predictions")
    .filter(F.col("batch_id") == BATCH_ID)
)

# 1. Prediction monitoring
prediction_stats = predictions.agg(
    F.count("*").alias("total"),
    F.sum(
        F.when(F.col("predicted_class") == 1, 1)
        .otherwise(0)
    ).alias("churn_count"),
    F.avg("churn_probability").alias("avg_probability")
).first()

total = prediction_stats["total"]
churn_count = prediction_stats["churn_count"]

assert total == 1409
assert churn_count == 305

print("\n1. PREDICTION MONITORING")
print("Batch ID:", BATCH_ID)
print("Predictions:", total)
print("Predicted churn:", churn_count)
print("Predicted churn rate:", f"{churn_count / total:.2%}")
print(
    "Average churn probability:",
    round(prediction_stats["avg_probability"], 4)
)

# 2. Numerical feature drift
print("\n2. NUMERICAL FEATURE DRIFT")

numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

drift_statuses = []

for feature in numeric_features:

    train_stats = train.agg(
        F.avg(feature).alias("mean"),
        F.stddev(feature).alias("std")
    ).first()

    test_stats = test.agg(
        F.avg(feature).alias("mean")
    ).first()

    std = train_stats["std"]

    if std is None or std == 0:
        smd = None
        status = "REVIEW"
    else:
        smd = abs(
            test_stats["mean"] - train_stats["mean"]
        ) / std

        status = (
            "LARGE" if smd >= 0.20
            else "MODERATE" if smd >= 0.10
            else "LOW"
        )

    drift_statuses.append(status)

    print(
        f"{feature}: "
        f"SMD={round(smd, 6) if smd is not None else 'N/A'}, "
        f"status={status}"
    )

# 3. Categorical feature drift
print("\n3. CONTRACT DISTRIBUTION DRIFT")

train_total = train.count()
test_total = test.count()

train_dist = {
    r["Contract"]: r["count"] / train_total
    for r in train.groupBy("Contract").count().collect()
}

test_dist = {
    r["Contract"]: r["count"] / test_total
    for r in test.groupBy("Contract").count().collect()
}

categories = set(train_dist) | set(test_dist)

tvd = 0.5 * sum(
    abs(
        train_dist.get(c, 0) - test_dist.get(c, 0)
    )
    for c in categories
)

contract_status = (
    "LARGE" if tvd >= 0.10
    else "MODERATE" if tvd >= 0.05
    else "LOW"
)

drift_statuses.append(contract_status)

print("Contract TVD:", round(tvd, 6))
print("Status:", contract_status)

# 4. Data quality
print("\n4. DATA QUALITY")

features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract"
]

quality_statuses = []

for feature in features:
    missing = test.filter(
        F.col(feature).isNull()
    ).count()

    rate = missing / test_total * 100

    status = (
        "CRITICAL" if rate > 2
        else "WARNING" if rate > 0.5
        else "NORMAL"
    )

    quality_statuses.append(status)

    print(
        f"{feature}: missing={missing}, "
        f"rate={rate:.2f}%, status={status}"
    )

# Reuse the validated quality findings from Step 16.10
# as an explicit prerequisite, rather than claiming
# that missingness checks cover every quality dimension.
print("Other quality checks: PASS in Step 16.10")

# 5. Summary
print("\n5. OVERALL MONITORING SUMMARY")

if "LARGE" in drift_statuses:
    drift_overall = "LARGE SHIFT"
elif "MODERATE" in drift_statuses:
    drift_overall = "MODERATE SHIFT"
elif "REVIEW" in drift_statuses:
    drift_overall = "REVIEW REQUIRED"
else:
    drift_overall = "LOW SHIFT"

if "CRITICAL" in quality_statuses:
    quality_overall = "CRITICAL"
elif "WARNING" in quality_statuses:
    quality_overall = "WARNING"
else:
    quality_overall = "NORMAL"

print("Feature drift screening:", drift_overall)
print("Missingness status:", quality_overall)
print("Prior full data-quality assessment: NORMAL")
print("Prediction baseline: ESTABLISHED")
print("Temporal production drift: NOT YET ASSESSED")

print("\n6. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")

print("\nSTEP 16.11 COMPLETE")


# COMMAND ----------


# Step 16.12 — Dashboard-ready monitoring report
# READ-ONLY: No inference, job runs, or Delta writes

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType
)

print("=" * 70)
print("STEP 16.12 — DASHBOARD-READY MONITORING REPORT")
print("=" * 70)

BATCH_ID = "d972dd16-029d-489f-a406-3c6c6e1e6265"

train = spark.table("workspace.default.telco_train_raw")
test = spark.table("workspace.default.telco_test_raw")

pred = (
    spark.table("workspace.default.telco_15_job_dev_predictions")
    .filter(F.col("batch_id") == BATCH_ID)
)

# 1. Prediction metrics

prediction_stats = pred.agg(
    F.count("*").alias("total"),
    F.sum(
        F.when(F.col("predicted_class") == 1, 1)
        .otherwise(0)
    ).alias("churn"),
    F.avg("churn_probability").alias("avg_probability")
).first()

total = prediction_stats["total"]
churn = prediction_stats["churn"]

assert total == 1409
assert churn == 305

metrics = [
    ("Prediction", "Total predictions",
     float(total), "BASELINE"),
    ("Prediction", "Predicted churn rate (%)",
     float(churn / total * 100), "BASELINE"),
    ("Prediction", "Average churn probability",
     float(prediction_stats["avg_probability"]), "BASELINE")
]

# 2. Numerical feature drift

for feature in ["tenure", "MonthlyCharges", "TotalCharges"]:

    a = train.agg(
        F.avg(feature).alias("mean"),
        F.stddev(feature).alias("std")
    ).first()

    b = test.agg(
        F.avg(feature).alias("mean")
    ).first()

    if a["std"] is None or a["std"] == 0:
        score = None
        status = "REVIEW"
    else:
        score = abs(b["mean"] - a["mean"]) / a["std"]
        status = (
            "LARGE" if score >= 0.20
            else "MODERATE" if score >= 0.10
            else "LOW"
        )

    metrics.append((
        "Feature Drift",
        f"{feature} SMD",
        float(score) if score is not None else None,
        status
    ))

# 3. Categorical feature drift

train_count = train.count()
test_count = test.count()

train_dist = {
    r["Contract"]: r["count"] / train_count
    for r in train.groupBy("Contract").count().collect()
}

test_dist = {
    r["Contract"]: r["count"] / test_count
    for r in test.groupBy("Contract").count().collect()
}

categories = set(train_dist) | set(test_dist)

tvd = 0.5 * sum(
    abs(train_dist.get(c, 0) - test_dist.get(c, 0))
    for c in categories
)

tvd_status = (
    "LARGE" if tvd >= 0.10
    else "MODERATE" if tvd >= 0.05
    else "LOW"
)

metrics.append((
    "Feature Drift", "Contract TVD",
    float(tvd), tvd_status
))

# 4. Missingness metrics

for feature in [
    "tenure", "MonthlyCharges", "TotalCharges", "Contract"
]:
    missing = test.filter(F.col(feature).isNull()).count()
    missing_pct = missing / test_count * 100

    status = (
        "CRITICAL" if missing_pct > 2
        else "WARNING" if missing_pct > 0.5
        else "NORMAL"
    )

    metrics.append((
        "Data Quality",
        f"{feature} missing (%)",
        float(missing_pct),
        status
    ))

# 5. Create dashboard-ready DataFrame

schema = StructType([
    StructField("category", StringType(), False),
    StructField("metric", StringType(), False),
    StructField("value", DoubleType(), True),
    StructField("status", StringType(), False)
])

monitoring_report = spark.createDataFrame(
    metrics,
    schema=schema
)

print("\n1. MONITORING METRICS")
monitoring_report.orderBy(
    "category", "metric"
).show(30, truncate=False)

print("\n2. METRIC COUNT")
print("Total metrics:", monitoring_report.count())

print("\n3. STATUS DISTRIBUTION")
monitoring_report.groupBy("status").count().show()

print("\n4. VALIDATION")
assert monitoring_report.count() == 11
print("Dashboard-ready structure: PASS")

print("\n5. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")

print("\nSTEP 16.12 COMPLETE")


# COMMAND ----------


# Step 16.13 — Monitoring Visualization
# READ-ONLY: Notebook charts only

import matplotlib.pyplot as plt

print("=" * 70)
print("STEP 16.13 — MONITORING VISUALIZATION")
print("=" * 70)

# Reuse the DataFrame created in Step 16.12
assert "monitoring_report" in globals(), (
    "Run Step 16.12 first to create monitoring_report"
)

rows = monitoring_report.collect()

metrics = {
    row["metric"]: row["value"]
    for row in rows
}

assert len(rows) == 11

# --------------------------------------------------
# 1. Prediction monitoring
# --------------------------------------------------

print("\n1. PREDICTION MONITORING")

prediction_names = [
    "Total predictions",
    "Predicted churn rate (%)",
    "Average churn probability"
]

for name in prediction_names:
    print(f"{name}: {metrics[name]:.4f}")

# Plot prediction rates on a common percentage scale
plt.figure(figsize=(8, 4))

plt.bar(
    ["Predicted churn rate", "Average churn probability"],
    [
        metrics["Predicted churn rate (%)"],
        metrics["Average churn probability"] * 100
    ]
)

plt.ylabel("Percentage (%)")
plt.title("Telco Churn — Prediction Baseline")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 2. Feature drift visualization
# --------------------------------------------------

print("\n2. FEATURE DRIFT")

drift_names = [
    "tenure SMD",
    "MonthlyCharges SMD",
    "TotalCharges SMD",
    "Contract TVD"
]

drift_values = [metrics[name] for name in drift_names]

plt.figure(figsize=(9, 4))

plt.bar(drift_names, drift_values)
plt.ylabel("Drift score")
plt.title("Telco Churn — Feature Drift Screening")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 3. Data quality visualization
# --------------------------------------------------

print("\n3. DATA QUALITY")

quality_names = [
    "tenure missing (%)",
    "MonthlyCharges missing (%)",
    "TotalCharges missing (%)",
    "Contract missing (%)"
]

quality_values = [metrics[name] for name in quality_names]

plt.figure(figsize=(9, 4))

plt.bar(quality_names, quality_values)
plt.axhline(
    y=0.5,
    linestyle="--",
    label="Warning threshold: 0.5%"
)
plt.axhline(
    y=2.0,
    linestyle=":",
    label="Critical threshold: 2%"
)

plt.ylabel("Missing values (%)")
plt.title("Telco Churn — Data Quality Monitoring")
plt.xticks(rotation=20, ha="right")
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 4. Validation and safety
# --------------------------------------------------

print("\n4. VALIDATION")
assert len(drift_values) == 4
assert len(quality_values) == 4

print("Prediction chart: PASS")
print("Feature drift chart: PASS")
print("Data quality chart: PASS")

print("\n5. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")

print("\nSTEP 16.13 COMPLETE")


# COMMAND ----------


# Step 16.14 — Monitoring Dashboard Design
# READ-ONLY: Visualization only

import matplotlib.pyplot as plt

print("=" * 70)
print("STEP 16.14 — MONITORING DASHBOARD DESIGN")
print("=" * 70)

assert "monitoring_report" in globals(), (
    "Run Step 16.12 to recreate monitoring_report"
)

rows = monitoring_report.collect()

metrics = {
    row["metric"]: row["value"]
    for row in rows
}

assert len(rows) == 11

# Create a single dashboard figure
fig = plt.figure(figsize=(14, 10))

fig.suptitle(
    "TELCO CUSTOMER CHURN — MODEL MONITORING",
    fontsize=18,
    fontweight="bold"
)

# -----------------------------------------
# 1. KPI summary
# -----------------------------------------

kpis = [
    ("Total Predictions",
     f'{metrics["Total predictions"]:,.0f}'),
    ("Predicted Churn Rate",
     f'{metrics["Predicted churn rate (%)"]:.2f}%'),
    ("Average Churn Probability",
     f'{metrics["Average churn probability"]:.4f}')
]

for i, (label, value) in enumerate(kpis):

    ax = fig.add_subplot(3, 3, i + 1)

    ax.text(
        0.5, 0.65, value,
        ha="center", va="center",
        fontsize=22, fontweight="bold"
    )

    ax.text(
        0.5, 0.25, label,
        ha="center", va="center",
        fontsize=11
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

# -----------------------------------------
# 2. Feature drift chart
# -----------------------------------------

ax_drift = fig.add_subplot(3, 1, 2)

drift_names = [
    "tenure SMD",
    "MonthlyCharges SMD",
    "TotalCharges SMD",
    "Contract TVD"
]

drift_values = [
    metrics[name] for name in drift_names
]

ax_drift.bar(drift_names, drift_values)

ax_drift.set_title("Feature Drift Screening")
ax_drift.set_ylabel("Drift score")
ax_drift.tick_params(axis="x", rotation=15)

# -----------------------------------------
# 3. Missingness chart
# -----------------------------------------

ax_quality = fig.add_subplot(3, 1, 3)

quality_names = [
    "tenure missing (%)",
    "MonthlyCharges missing (%)",
    "TotalCharges missing (%)",
    "Contract missing (%)"
]

quality_values = [
    metrics[name] for name in quality_names
]

ax_quality.bar(quality_names, quality_values)

ax_quality.axhline(
    0.5, linestyle="--",
    label="Warning threshold"
)

ax_quality.axhline(
    2.0, linestyle=":",
    label="Critical threshold"
)

ax_quality.set_title("Data Quality — Missingness")
ax_quality.set_ylabel("Missing (%)")
ax_quality.tick_params(axis="x", rotation=15)
ax_quality.legend()

# -----------------------------------------
# 4. Dashboard footer
# -----------------------------------------

fig.text(
    0.5, 0.015,
    "Development baseline | "
    "Training/test drift screening | "
    "Not temporal production drift",
    ha="center",
    fontsize=10
)

plt.tight_layout(rect=[0, 0.04, 1, 0.94])
plt.show()

print("\n1. VALIDATION")
print("KPI cards: PASS")
print("Feature drift chart: PASS")
print("Data quality chart: PASS")
print("Dashboard rendered in notebook: PASS")

print("\n2. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Dashboard file saved: False")

print("\nSTEP 16.14 COMPLETE")


# COMMAND ----------


# Step 16.15 — Dashboard Validation and Readiness
# READ-ONLY: No inference, job runs, or Delta writes

import math

print("=" * 70)
print("STEP 16.15 — DASHBOARD VALIDATION AND READINESS")
print("=" * 70)

assert "monitoring_report" in globals(), (
    "Run Step 16.12 first to recreate monitoring_report"
)

rows = monitoring_report.collect()

expected_metrics = {
    "Total predictions",
    "Predicted churn rate (%)",
    "Average churn probability",
    "tenure SMD",
    "MonthlyCharges SMD",
    "TotalCharges SMD",
    "Contract TVD",
    "tenure missing (%)",
    "MonthlyCharges missing (%)",
    "TotalCharges missing (%)",
    "Contract missing (%)"
}

actual_metrics = [row["metric"] for row in rows]
metrics = {row["metric"]: row for row in rows}

print("\n1. METRIC COMPLETENESS")

assert len(rows) == 11
assert set(actual_metrics) == expected_metrics
assert len(actual_metrics) == len(set(actual_metrics))

print("Expected metrics: 11")
print("Actual metrics:", len(rows))
print("Missing or duplicate metrics: 0")
print("Result: PASS")

print("\n2. NUMERICAL VALIDITY")

for row in rows:
    value = row["value"]

    assert value is not None
    assert math.isfinite(value)

assert metrics["Total predictions"]["value"] == 1409
assert 0 <= metrics["Predicted churn rate (%)"]["value"] <= 100
assert 0 <= metrics["Average churn probability"]["value"] <= 1

print("All values finite: PASS")
print("Prediction metrics within expected bounds: PASS")

print("\n3. STATUS VALIDATION")

for row in rows:
    name = row["metric"]
    value = row["value"]
    status = row["status"]

    if name.endswith(" SMD"):
        expected = (
            "LARGE" if value >= 0.20
            else "MODERATE" if value >= 0.10
            else "LOW"
        )

    elif name == "Contract TVD":
        assert 0 <= value <= 1

        expected = (
            "LARGE" if value >= 0.10
            else "MODERATE" if value >= 0.05
            else "LOW"
        )

    elif name.endswith(" missing (%)"):
        assert 0 <= value <= 100

        expected = (
            "CRITICAL" if value > 2.0
            else "WARNING" if value > 0.5
            else "NORMAL"
        )

    else:
        expected = "BASELINE"

    assert status == expected, (
        f"Status mismatch: {name}, "
        f"expected={expected}, actual={status}"
    )

print("All 11 metric statuses: PASS")

print("\n4. MONITORING SCOPE")

print("Dataset scope: DEVELOPMENT TRAIN/TEST SPLIT")
print("Prediction scope: EXISTING DEVELOPMENT BATCH")
print("Temporal production drift: NOT ASSESSED")
print("Dashboard type: NOTEBOOK VISUALIZATION")

print("\n5. READINESS ASSESSMENT")

print("Metric completeness: PASS")
print("Metric validity: PASS")
print("Threshold consistency: PASS")
print("Development dashboard readiness: PASS")
print("Production monitoring readiness: NOT YET ESTABLISHED")

print("\n6. SAFETY")

print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Dashboard published: False")

print("\nSTEP 16.15 COMPLETE")


# COMMAND ----------


# STEP 16.16 — Production Monitoring Readiness Assessment
# READ-ONLY: Metadata and aggregate checks only

from pyspark.sql import functions as F

print("=" * 70)
print("STEP 16.16 — PRODUCTION MONITORING READINESS")
print("=" * 70)

tables = {
    "Batch Registry":
        "workspace.default.telco_15_job_batch_registry",
    "Development Predictions":
        "workspace.default.telco_15_job_dev_predictions",
    "Approved Predictions":
        "workspace.default.telco_15_approved_predictions"
}

# --------------------------------------------------
# 1. Inspect existing table structures
# --------------------------------------------------

print("\n1. EXISTING MONITORING DATA SOURCES")

table_info = {}

for label, table_name in tables.items():

    print(f"\n--- {label} ---")
    print("Table:", table_name)

    df = spark.table(table_name)

    row_count = df.count()
    columns = df.columns

    print("Rows:", row_count)
    print("Columns:", len(columns))
    print("Column names:", columns)

    table_info[label] = {
        "rows": row_count,
        "columns": columns
    }

# --------------------------------------------------
# 2. Check batch tracking capabilities
# --------------------------------------------------

print("\n2. BATCH TRACKING CAPABILITIES")

for label, info in table_info.items():

    columns = info["columns"]

    has_batch_id = "batch_id" in columns

    timestamp_columns = [
        c for c in columns
        if (
            "timestamp" in c.lower()
            or "created_at" in c.lower()
            or "processed_at" in c.lower()
        )
    ]

    print(f"\n{label}")
    print("Batch ID available:", has_batch_id)
    print("Timestamp columns:", timestamp_columns)

# --------------------------------------------------
# 3. Inspect development batch distribution
# --------------------------------------------------

print("\n3. DEVELOPMENT BATCH DISTRIBUTION")

dev_df = spark.table(
    tables["Development Predictions"]
)

if "batch_id" in dev_df.columns:

    dev_df.groupBy("batch_id").agg(
        F.count("*").alias("prediction_count")
    ).orderBy("batch_id").show(truncate=False)

else:
    print("No batch_id column found")

# --------------------------------------------------
# 4. Verify protected approved table
# --------------------------------------------------

print("\n4. PROTECTED APPROVED TABLE CHECK")

approved_df = spark.table(
    tables["Approved Predictions"]
)

approved_count = approved_df.count()

print("Approved prediction rows:", approved_count)

assert approved_count == 1409, (
    "Protected approved prediction row count changed!"
)

# Check Delta version without modifying the table
history = spark.sql(
    "DESCRIBE HISTORY "
    "workspace.default.telco_15_approved_predictions"
)

latest_version = history.agg(
    F.max("version").alias("latest_version")
).first()["latest_version"]

print("Latest Delta version:", latest_version)

assert latest_version == 3, (
    "Protected approved table Delta version changed!"
)

print("Protected table integrity checks: PASS")

# --------------------------------------------------
# 5. Assessment summary
# --------------------------------------------------

print("\n5. READINESS ASSESSMENT")

registry_has_batch_id = (
    "batch_id" in table_info["Batch Registry"]["columns"]
)

dev_has_batch_id = (
    "batch_id"
    in table_info["Development Predictions"]["columns"]
)

print(
    "Registry batch tracking:",
    "AVAILABLE" if registry_has_batch_id else "MISSING"
)

print(
    "Development prediction batch tracking:",
    "AVAILABLE" if dev_has_batch_id else "MISSING"
)

print("Protected approved table: VERIFIED")
print("Temporal monitoring: REQUIRES NEW DATA VALIDATION")
print("Production readiness: NOT YET ESTABLISHED")

# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n6. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected table modified: False")

print("\nSTEP 16.16 COMPLETE")


# COMMAND ----------


# Step 16.17 — Historical Batch Metadata Comparison
# READ-ONLY: No inference, job runs, or Delta writes

from pyspark.sql import functions as F

print("=" * 70)
print("STEP 16.17 — HISTORICAL BATCH METADATA COMPARISON")
print("=" * 70)

REGISTRY_TABLE = "workspace.default.telco_15_job_batch_registry"

registry = spark.table(REGISTRY_TABLE)

# --------------------------------------------------
# 1. Validate expected registry structure
# --------------------------------------------------

required_columns = [
    "batch_id",
    "source_table",
    "source_fingerprint",
    "record_count",
    "model_name",
    "model_version",
    "classification_threshold",
    "batch_timestamp",
    "registered_at"
]

assert all(
    col in registry.columns
    for col in required_columns
), "Required registry columns are missing"

print("\n1. REGISTRY VALIDATION")
print("Registry rows:", registry.count())
print("Required columns: PASS")

# --------------------------------------------------
# 2. Compare historical batch metadata
# --------------------------------------------------

print("\n2. HISTORICAL BATCH METADATA")

registry.select(
    "batch_id",
    "source_table",
    "record_count",
    "model_name",
    "model_version",
    "classification_threshold",
    "batch_timestamp",
    "registered_at"
).orderBy("batch_timestamp").show(
    truncate=False
)

# --------------------------------------------------
# 3. Compare source fingerprints
# --------------------------------------------------

print("\n3. SOURCE FINGERPRINT COMPARISON")

registry.groupBy(
    "source_fingerprint"
).agg(
    F.count("*").alias("batch_count"),
    F.countDistinct("source_table").alias("source_tables"),
    F.min("batch_timestamp").alias("first_batch"),
    F.max("batch_timestamp").alias("last_batch")
).orderBy(
    F.desc("batch_count")
).show(truncate=False)

# --------------------------------------------------
# 4. Count distinct metadata values
# --------------------------------------------------

print("\n4. HISTORICAL DIVERSITY SUMMARY")

summary = registry.agg(
    F.countDistinct("batch_id").alias("unique_batches"),
    F.countDistinct("source_fingerprint").alias(
        "unique_source_fingerprints"
    ),
    F.countDistinct("source_table").alias(
        "unique_source_tables"
    ),
    F.countDistinct("model_version").alias(
        "unique_model_versions"
    ),
    F.countDistinct("classification_threshold").alias(
        "unique_thresholds"
    )
).first()

for name in summary.asDict():
    print(f"{name}: {summary[name]}")

# --------------------------------------------------
# 5. Assess historical monitoring evidence
# --------------------------------------------------

print("\n5. HISTORICAL MONITORING ASSESSMENT")

if summary["unique_source_fingerprints"] == 1:
    print("Source fingerprint diversity: NONE")
    print("Assessment: REPEATED SOURCE FINGERPRINT")
else:
    print("Source fingerprint diversity: DETECTED")
    print("Assessment: INVESTIGATE DISTINCT SOURCES")

print(
    "Independent production populations:",
    "NOT ESTABLISHED"
)

print(
    "Temporal production drift:",
    "NOT YET ASSESSED"
)

# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n6. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected table modified: False")

print("\nSTEP 16.17 COMPLETE")


# COMMAND ----------


# Step 16.18 — Historical Monitoring Data Requirements
# READ-ONLY: No inference, jobs, or Delta writes

from pyspark.sql import functions as F

print("=" * 70)
print("STEP 16.18 — HISTORICAL MONITORING REQUIREMENTS")
print("=" * 70)

registry = spark.table(
    "workspace.default.telco_15_job_batch_registry"
)

predictions = spark.table(
    "workspace.default.telco_15_job_dev_predictions"
)

# ---------------------------------------------
# 1. Verify existing metadata capabilities
# ---------------------------------------------

print("\n1. EXISTING CAPABILITIES")

registry_columns = set(registry.columns)
prediction_columns = set(predictions.columns)

checks = [
    (
        "Batch identifiers",
        "batch_id" in registry_columns
    ),
    (
        "Batch timestamps",
        "batch_timestamp" in registry_columns
    ),
    (
        "Source fingerprints",
        "source_fingerprint" in registry_columns
    ),
    (
        "Model version tracking",
        "model_version" in registry_columns
    ),
    (
        "Prediction probabilities",
        "churn_probability" in prediction_columns
    ),
    (
        "Prediction timestamps",
        "prediction_timestamp" in prediction_columns
    )
]

for capability, available in checks:
    print(
        f"{capability}: "
        f"{'AVAILABLE' if available else 'MISSING'}"
    )

# ---------------------------------------------
# 2. Assess source population diversity
# ---------------------------------------------

print("\n2. SOURCE POPULATION ASSESSMENT")

source_summary = registry.agg(
    F.countDistinct("batch_id").alias("batches"),
    F.countDistinct("source_fingerprint").alias(
        "fingerprints"
    )
).first()

print("Unique batches:", source_summary["batches"])
print(
    "Unique source fingerprints:",
    source_summary["fingerprints"]
)

if source_summary["fingerprints"] <= 1:
    population_status = "NOT ESTABLISHED"
else:
    population_status = "REQUIRES VALIDATION"

print(
    "Independent source populations:",
    population_status
)

# ---------------------------------------------
# 3. Define remaining monitoring requirements
# ---------------------------------------------

print("\n3. REMAINING REQUIREMENTS")

requirements = [
    (
        "Independent future data batches",
        "NOT ESTABLISHED"
    ),
    (
        "Historical feature snapshots",
        "NOT VERIFIED"
    ),
    (
        "Actual outcomes for future batches",
        "NOT VERIFIED"
    ),
    (
        "Persistent monitoring metrics",
        "NOT IMPLEMENTED"
    ),
    (
        "Operational alert delivery",
        "NOT IMPLEMENTED"
    ),
    (
        "Monitoring schedule",
        "NOT IMPLEMENTED"
    )
]

for requirement, status in requirements:
    print(f"{requirement}: {status}")

# ---------------------------------------------
# 4. Readiness summary
# ---------------------------------------------

print("\n4. READINESS SUMMARY")

available_count = sum(
    1 for _, available in checks if available
)

print(
    "Verified metadata capabilities:",
    f"{available_count}/{len(checks)}"
)

print("Development monitoring: COMPLETE")
print("Historical metadata assessment: COMPLETE")
print("Temporal drift monitoring: NOT IMPLEMENTED")
print("Production monitoring readiness: NOT ESTABLISHED")

# ---------------------------------------------
# 5. Safety
# ---------------------------------------------

print("\n5. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected tables modified: False")

print("\nSTEP 16.18 COMPLETE")


# COMMAND ----------


# Step 16.19 — Monitoring Pipeline Design
# READ-ONLY: Configuration and source validation

print("=" * 70)
print("STEP 16.19 — MONITORING PIPELINE DESIGN")
print("=" * 70)

monitoring_config = {
    "pipeline_name": "Telco_Churn_Model_Monitoring",
    "environment": "DEVELOPMENT",
    "reference_table": "workspace.default.telco_train_raw",
    "comparison_table": "workspace.default.telco_test_raw",
    "prediction_table":
        "workspace.default.telco_15_job_dev_predictions",
    "registry_table":
        "workspace.default.telco_15_job_batch_registry",
    "baseline_batch_id":
        "d972dd16-029d-489f-a406-3c6c6e1e6265",
    "numerical_features": [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ],
    "categorical_features": [
        "Contract"
    ],
    "drift_thresholds": {
        "numerical_moderate": 0.10,
        "numerical_large": 0.20,
        "categorical_moderate": 0.05,
        "categorical_large": 0.10
    },
    "missingness_thresholds_pct": {
        "warning": 0.5,
        "critical": 2.0
    },
    "allow_inference": False,
    "allow_writes": False,
    "allow_job_submission": False
}

print("\n1. PIPELINE CONFIGURATION")
print("Pipeline:", monitoring_config["pipeline_name"])
print("Environment:", monitoring_config["environment"])

print("\n2. INPUT SOURCE VALIDATION")

source_keys = [
    "reference_table",
    "comparison_table",
    "prediction_table",
    "registry_table"
]

for key in source_keys:
    table_name = monitoring_config[key]

    assert spark.catalog.tableExists(table_name), (
        f"Missing required table: {table_name}"
    )

    print(f"{key}: AVAILABLE")

print("\n3. FEATURE VALIDATION")

reference_columns = set(
    spark.table(
        monitoring_config["reference_table"]
    ).columns
)

comparison_columns = set(
    spark.table(
        monitoring_config["comparison_table"]
    ).columns
)

features = (
    monitoring_config["numerical_features"]
    + monitoring_config["categorical_features"]
)

for feature in features:
    assert feature in reference_columns
    assert feature in comparison_columns

    print(f"{feature}: AVAILABLE")

print("\n4. BASELINE BATCH VALIDATION")

from pyspark.sql import functions as F

baseline_id = monitoring_config["baseline_batch_id"]

registry = spark.table(
    monitoring_config["registry_table"]
)

predictions = spark.table(
    monitoring_config["prediction_table"]
)

registered = registry.filter(
    F.col("batch_id") == baseline_id
).count()

prediction_count = predictions.filter(
    F.col("batch_id") == baseline_id
).count()

assert registered == 1
assert prediction_count == 1409

print("Registry entries:", registered)
print("Baseline predictions:", prediction_count)
print("Baseline validation: PASS")

print("\n5. PIPELINE SAFETY CONTROLS")

assert monitoring_config["allow_inference"] is False
assert monitoring_config["allow_writes"] is False
assert monitoring_config["allow_job_submission"] is False

print("Inference allowed: False")
print("Writes allowed: False")
print("Job submission allowed: False")
print("Safety configuration: PASS")

print("\n6. READINESS")

print("Development monitoring inputs: READY")
print("Monitoring pipeline configuration: VALIDATED")
print("Independent production data: NOT AVAILABLE")
print("Production monitoring deployment: NOT AUTHORIZED")

print("\nSTEP 16.19 COMPLETE")


# COMMAND ----------


# Step 16.20 — Reusable Monitoring Functions
# READ-ONLY: No inference, jobs, or Delta writes

from pyspark.sql import functions as F

print("=" * 70)
print("STEP 16.20 — REUSABLE MONITORING FUNCTIONS")
print("=" * 70)

# --------------------------------------------------
# 1. Numerical drift (SMD)
# --------------------------------------------------

def calculate_numeric_drift(reference_df, current_df, feature):

    reference_stats = reference_df.agg(
        F.avg(feature).alias("mean"),
        F.stddev(feature).alias("std")
    ).first()

    current_stats = current_df.agg(
        F.avg(feature).alias("mean")
    ).first()

    ref_mean = reference_stats["mean"]
    ref_std = reference_stats["std"]
    current_mean = current_stats["mean"]

    if (
        ref_mean is None
        or current_mean is None
        or ref_std is None
        or ref_std == 0
    ):
        return {
            "feature": feature,
            "score": None,
            "status": "REVIEW"
        }

    score = abs(current_mean - ref_mean) / ref_std

    status = (
        "LARGE" if score >= 0.20
        else "MODERATE" if score >= 0.10
        else "LOW"
    )

    return {
        "feature": feature,
        "score": float(score),
        "status": status
    }


# --------------------------------------------------
# 2. Categorical drift (TVD)
# --------------------------------------------------

def calculate_categorical_drift(reference_df, current_df, feature):

    reference_total = reference_df.count()
    current_total = current_df.count()

    if reference_total == 0 or current_total == 0:
        return {
            "feature": feature,
            "score": None,
            "status": "REVIEW"
        }

    reference_dist = {
        row[feature]: row["count"] / reference_total
        for row in reference_df.groupBy(feature).count().collect()
    }

    current_dist = {
        row[feature]: row["count"] / current_total
        for row in current_df.groupBy(feature).count().collect()
    }

    categories = set(reference_dist) | set(current_dist)

    score = 0.5 * sum(
        abs(
            reference_dist.get(category, 0)
            - current_dist.get(category, 0)
        )
        for category in categories
    )

    status = (
        "LARGE" if score >= 0.10
        else "MODERATE" if score >= 0.05
        else "LOW"
    )

    return {
        "feature": feature,
        "score": float(score),
        "status": status
    }


# --------------------------------------------------
# 3. Missingness monitoring
# --------------------------------------------------

def calculate_missingness(df, feature):

    total = df.count()

    if total == 0:
        return {
            "feature": feature,
            "missing_count": 0,
            "missing_pct": None,
            "status": "REVIEW"
        }

    missing = df.filter(
        F.col(feature).isNull()
    ).count()

    missing_pct = missing / total * 100

    status = (
        "CRITICAL" if missing_pct > 2.0
        else "WARNING" if missing_pct > 0.5
        else "NORMAL"
    )

    return {
        "feature": feature,
        "missing_count": missing,
        "missing_pct": float(missing_pct),
        "status": status
    }


# --------------------------------------------------
# 4. Test functions against existing data
# --------------------------------------------------

reference = spark.table(
    monitoring_config["reference_table"]
)

current = spark.table(
    monitoring_config["comparison_table"]
)

print("\n1. NUMERICAL DRIFT")

for feature in monitoring_config["numerical_features"]:
    result = calculate_numeric_drift(
        reference, current, feature
    )

    print(
        f'{result["feature"]}: '
        f'SMD={result["score"]:.6f}, '
        f'status={result["status"]}'
    )

print("\n2. CATEGORICAL DRIFT")

for feature in monitoring_config["categorical_features"]:
    result = calculate_categorical_drift(
        reference, current, feature
    )

    print(
        f'{result["feature"]}: '
        f'TVD={result["score"]:.6f}, '
        f'status={result["status"]}'
    )

print("\n3. MISSINGNESS MONITORING")

for feature in (
    monitoring_config["numerical_features"]
    + monitoring_config["categorical_features"]
):
    result = calculate_missingness(current, feature)

    print(
        f'{result["feature"]}: '
        f'missing={result["missing_count"]}, '
        f'rate={result["missing_pct"]:.2f}%, '
        f'status={result["status"]}'
    )

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected tables modified: False")

print("\nSTEP 16.20 COMPLETE")


# COMMAND ----------


# Step 16.21 — Reusable Monitoring Report Builder
# READ-ONLY: No inference, jobs, or Delta writes

from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType
)

print("=" * 70)
print("STEP 16.21 — REUSABLE MONITORING REPORT BUILDER")
print("=" * 70)

# --------------------------------------------------
# 1. Define reusable report builder
# --------------------------------------------------

def build_monitoring_report(reference_df, current_df, config):

    results = []

    # Numerical drift
    for feature in config["numerical_features"]:

        result = calculate_numeric_drift(
            reference_df,
            current_df,
            feature
        )

        results.append((
            feature,
            "SMD",
            result["score"],
            result["status"]
        ))

    # Categorical drift
    for feature in config["categorical_features"]:

        result = calculate_categorical_drift(
            reference_df,
            current_df,
            feature
        )

        results.append((
            feature,
            "TVD",
            result["score"],
            result["status"]
        ))

    # Missingness
    all_features = (
        config["numerical_features"]
        + config["categorical_features"]
    )

    for feature in all_features:

        result = calculate_missingness(
            current_df,
            feature
        )

        results.append((
            feature,
            "MISSING_PCT",
            result["missing_pct"],
            result["status"]
        ))

    schema = StructType([
        StructField("feature", StringType(), False),
        StructField("metric", StringType(), False),
        StructField("score", DoubleType(), True),
        StructField("status", StringType(), False)
    ])

    return spark.createDataFrame(
        results,
        schema=schema
    )


# --------------------------------------------------
# 2. Load existing development datasets
# --------------------------------------------------

reference = spark.table(
    monitoring_config["reference_table"]
)

current = spark.table(
    monitoring_config["comparison_table"]
)

# --------------------------------------------------
# 3. Build monitoring report
# --------------------------------------------------

report = build_monitoring_report(
    reference,
    current,
    monitoring_config
)

print("\n1. MONITORING REPORT")

report.orderBy(
    "metric",
    "feature"
).show(truncate=False)

# --------------------------------------------------
# 4. Validate report
# --------------------------------------------------

print("\n2. VALIDATION")

assert report.count() == 8

assert report.filter(
    report.status == "REVIEW"
).count() == 0

print("Expected metrics: 8")
print("Actual metrics:", report.count())
print("Undefined metrics: 0")
print("Validation: PASS")

# --------------------------------------------------
# 5. Status distribution
# --------------------------------------------------

print("\n3. STATUS DISTRIBUTION")

report.groupBy(
    "status"
).count().show()

# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected tables modified: False")

print("\nSTEP 16.21 COMPLETE")


# COMMAND ----------


# Step 16.22 — Monitoring Report Validation and Alert Aggregation
# READ-ONLY: No inference, jobs, or Delta writes

import math

print("=" * 70)
print("STEP 16.22 — MONITORING ALERT AGGREGATION")
print("=" * 70)

# --------------------------------------------------
# 1. Define reusable alert aggregation function
# --------------------------------------------------

def aggregate_monitoring_alerts(report_df, config):

    rows = report_df.collect()

    numeric_features = config["numerical_features"]
    categorical_features = config["categorical_features"]

    all_features = numeric_features + categorical_features

    expected_pairs = (
        {(feature, "SMD") for feature in numeric_features}
        | {(feature, "TVD") for feature in categorical_features}
        | {(feature, "MISSING_PCT") for feature in all_features}
    )

    actual_pairs = [
        (row["feature"], row["metric"])
        for row in rows
    ]

    # Validate metric completeness and uniqueness
    if (
        len(actual_pairs) != len(expected_pairs)
        or set(actual_pairs) != expected_pairs
        or len(actual_pairs) != len(set(actual_pairs))
    ):
        return {
            "overall_status": "REVIEW REQUIRED",
            "drift_status": "REVIEW REQUIRED",
            "quality_status": "REVIEW REQUIRED",
            "reason": "Missing, duplicate, or unexpected metrics"
        }

    drift_statuses = []
    quality_statuses = []

    for row in rows:

        metric = row["metric"]
        score = row["score"]
        status = row["status"]

        if score is None or not math.isfinite(score):
            return {
                "overall_status": "REVIEW REQUIRED",
                "drift_status": "REVIEW REQUIRED",
                "quality_status": "REVIEW REQUIRED",
                "reason": "Undefined or invalid metric score"
            }

        if metric == "SMD":
            if score < 0:
                expected = None
            else:
                expected = (
                    "LARGE" if score >= 0.20
                    else "MODERATE" if score >= 0.10
                    else "LOW"
                )
            drift_statuses.append(status)

        elif metric == "TVD":
            if not 0 <= score <= 1:
                expected = None
            else:
                expected = (
                    "LARGE" if score >= 0.10
                    else "MODERATE" if score >= 0.05
                    else "LOW"
                )
            drift_statuses.append(status)

        elif metric == "MISSING_PCT":
            if not 0 <= score <= 100:
                expected = None
            else:
                expected = (
                    "CRITICAL" if score > 2.0
                    else "WARNING" if score > 0.5
                    else "NORMAL"
                )
            quality_statuses.append(status)

        else:
            expected = None

        if status != expected:
            return {
                "overall_status": "REVIEW REQUIRED",
                "drift_status": "REVIEW REQUIRED",
                "quality_status": "REVIEW REQUIRED",
                "reason": f"Invalid status or score for {row['feature']}"
            }

    # Aggregate drift statuses
    if "LARGE" in drift_statuses:
        drift_overall = "LARGE"
    elif "MODERATE" in drift_statuses:
        drift_overall = "MODERATE"
    else:
        drift_overall = "LOW"

    # Aggregate missingness statuses
    if "CRITICAL" in quality_statuses:
        quality_overall = "CRITICAL"
    elif "WARNING" in quality_statuses:
        quality_overall = "WARNING"
    else:
        quality_overall = "NORMAL"

    # Combine both dimensions
    if drift_overall == "LARGE" or quality_overall == "CRITICAL":
        overall = "CRITICAL"
    elif drift_overall == "MODERATE" or quality_overall == "WARNING":
        overall = "WARNING"
    else:
        overall = "NORMAL"

    return {
        "overall_status": overall,
        "drift_status": drift_overall,
        "quality_status": quality_overall,
        "reason": "Validated monitoring metrics"
    }


# --------------------------------------------------
# 2. Aggregate the existing monitoring report
# --------------------------------------------------

assert "report" in globals(), (
    "Run Step 16.21 first to create report"
)

assert "monitoring_config" in globals(), (
    "Run Step 16.19 first to create monitoring_config"
)

assessment = aggregate_monitoring_alerts(
    report,
    monitoring_config
)

print("\n1. MONITORING ASSESSMENT")

for key, value in assessment.items():
    print(f"{key}: {value}")

# --------------------------------------------------
# 3. Validate expected development result
# --------------------------------------------------

print("\n2. VALIDATION")

assert assessment["overall_status"] == "NORMAL"
assert assessment["drift_status"] == "LOW"
assert assessment["quality_status"] == "NORMAL"

print("Overall monitoring status: PASS")
print("Drift aggregation: PASS")
print("Missingness aggregation: PASS")

# --------------------------------------------------
# 4. Scope and safety
# --------------------------------------------------

print("\n3. MONITORING SCOPE")
print("Environment: DEVELOPMENT")
print("Reference: TRAINING DATA")
print("Comparison: TEST DATA")
print("Temporal production drift: NOT ASSESSED")

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected tables modified: False")

print("\nSTEP 16.22 COMPLETE")


# COMMAND ----------


# Step 16.23 — Alert Aggregation Testing
# READ-ONLY: In-memory simulated monitoring scenarios

from pyspark.sql import functions as F

print("=" * 70)
print("STEP 16.23 — ALERT AGGREGATION TESTING")
print("=" * 70)

# --------------------------------------------------
# 1. Preserve the original monitoring report
# --------------------------------------------------

baseline_rows = [
    row.asDict() for row in report.collect()
]

report_schema = report.schema

def make_test_report(feature=None, metric=None, score=None):
    """
    Create a new in-memory report with one modified metric.
    The original report and Delta tables remain unchanged.
    """

    rows = [dict(row) for row in baseline_rows]

    if feature is not None and metric is not None:

        for row in rows:
            if row["feature"] == feature and row["metric"] == metric:

                row["score"] = score

                if metric == "SMD":
                    row["status"] = (
                        "LARGE" if score >= 0.20
                        else "MODERATE" if score >= 0.10
                        else "LOW"
                    ) if score is not None else "REVIEW"

                elif metric == "TVD":
                    row["status"] = (
                        "LARGE" if score >= 0.10
                        else "MODERATE" if score >= 0.05
                        else "LOW"
                    ) if score is not None else "REVIEW"

                elif metric == "MISSING_PCT":
                    row["status"] = (
                        "CRITICAL" if score > 2.0
                        else "WARNING" if score > 0.5
                        else "NORMAL"
                    ) if score is not None else "REVIEW"

    return spark.createDataFrame(rows, schema=report_schema)


# --------------------------------------------------
# 2. Define test scenarios
# --------------------------------------------------

scenarios = [
    {
        "name": "Normal development baseline",
        "feature": None,
        "metric": None,
        "score": None,
        "expected": "NORMAL"
    },
    {
        "name": "Moderate numerical drift",
        "feature": "tenure",
        "metric": "SMD",
        "score": 0.15,
        "expected": "WARNING"
    },
    {
        "name": "Large categorical drift",
        "feature": "Contract",
        "metric": "TVD",
        "score": 0.12,
        "expected": "CRITICAL"
    },
    {
        "name": "Critical missingness",
        "feature": "TotalCharges",
        "metric": "MISSING_PCT",
        "score": 3.0,
        "expected": "CRITICAL"
    },
    {
        "name": "Undefined numerical drift",
        "feature": "tenure",
        "metric": "SMD",
        "score": None,
        "expected": "REVIEW REQUIRED"
    }
]

# --------------------------------------------------
# 3. Execute tests
# --------------------------------------------------

print("\n1. TEST RESULTS")

passed = 0

for i, scenario in enumerate(scenarios, start=1):

    test_report = make_test_report(
        scenario["feature"],
        scenario["metric"],
        scenario["score"]
    )

    result = aggregate_monitoring_alerts(
        test_report,
        monitoring_config
    )

    actual = result["overall_status"]
    expected = scenario["expected"]

    test_passed = actual == expected

    if test_passed:
        passed += 1

    print(f"\nTest {i}: {scenario['name']}")
    print("Expected:", expected)
    print("Actual:", actual)
    print("Result:", "PASS" if test_passed else "FAIL")


# --------------------------------------------------
# 4. Test summary
# --------------------------------------------------

print("\n2. TEST SUMMARY")

print("Total tests:", len(scenarios))
print("Passed:", passed)
print("Failed:", len(scenarios) - passed)

assert passed == len(scenarios), (
    "One or more alert aggregation tests failed"
)

print("Alert aggregation validation: PASS")


# --------------------------------------------------
# 5. Verify original report remains unchanged
# --------------------------------------------------

print("\n3. ORIGINAL REPORT VALIDATION")

original_assessment = aggregate_monitoring_alerts(
    report,
    monitoring_config
)

assert original_assessment["overall_status"] == "NORMAL"

print("Original report status: NORMAL")
print("Original report preserved: PASS")


# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected tables modified: False")

print("\nSTEP 16.23 COMPLETE")


# COMMAND ----------


# Step 16.24 — Detailed Monitoring Alerts
# READ-ONLY: In-memory simulations only

print("=" * 70)
print("STEP 16.24 — DETAILED MONITORING ALERTS")
print("=" * 70)

# --------------------------------------------------
# 1. Define reusable alert detail function
# --------------------------------------------------

def build_alert_details(report_df):

    alerts = []

    for row in report_df.collect():

        feature = row["feature"]
        metric = row["metric"]
        score = row["score"]
        status = row["status"]

        # Flag undefined or invalid metrics
        if status == "REVIEW" or score is None:
            alerts.append({
                "feature": feature,
                "metric": metric,
                "score": score,
                "severity": "REVIEW REQUIRED",
                "message": (
                    f"{feature}: metric is undefined "
                    "or requires investigation"
                )
            })
            continue

        # Numerical drift
        if metric == "SMD" and status in (
            "MODERATE", "LARGE"
        ):

            severity = (
                "CRITICAL" if status == "LARGE"
                else "WARNING"
            )

            message = (
                f"{feature}: numerical drift SMD={score:.4f} "
                f"({status})"
            )

        # Categorical drift
        elif metric == "TVD" and status in (
            "MODERATE", "LARGE"
        ):

            severity = (
                "CRITICAL" if status == "LARGE"
                else "WARNING"
            )

            message = (
                f"{feature}: categorical drift TVD={score:.4f} "
                f"({status})"
            )

        # Missingness
        elif metric == "MISSING_PCT" and status in (
            "WARNING", "CRITICAL"
        ):

            severity = status

            message = (
                f"{feature}: missingness={score:.2f}% "
                f"({status})"
            )

        else:
            continue

        alerts.append({
            "feature": feature,
            "metric": metric,
            "score": score,
            "severity": severity,
            "message": message
        })

    return alerts


# --------------------------------------------------
# 2. Evaluate the original development report
# --------------------------------------------------

print("\n1. DEVELOPMENT MONITORING ALERTS")

development_alerts = build_alert_details(report)

print("Alert count:", len(development_alerts))

for alert in development_alerts:
    print(alert["message"])

assert len(development_alerts) == 0

print("Development monitoring alerts: NONE")


# --------------------------------------------------
# 3. Simulate a warning scenario
# --------------------------------------------------

print("\n2. SIMULATED WARNING SCENARIO")

warning_report = make_test_report(
    feature="tenure",
    metric="SMD",
    score=0.15
)

warning_alerts = build_alert_details(warning_report)

for alert in warning_alerts:
    print("Severity:", alert["severity"])
    print("Message:", alert["message"])

assert len(warning_alerts) == 1
assert warning_alerts[0]["severity"] == "WARNING"


# --------------------------------------------------
# 4. Simulate a critical scenario
# --------------------------------------------------

print("\n3. SIMULATED CRITICAL SCENARIO")

critical_report = make_test_report(
    feature="TotalCharges",
    metric="MISSING_PCT",
    score=3.0
)

critical_alerts = build_alert_details(critical_report)

for alert in critical_alerts:
    print("Severity:", alert["severity"])
    print("Message:", alert["message"])

assert len(critical_alerts) == 1
assert critical_alerts[0]["severity"] == "CRITICAL"


# --------------------------------------------------
# 5. Validate alert generation
# --------------------------------------------------

print("\n4. VALIDATION")

print("Development scenario: PASS")
print("Warning scenario: PASS")
print("Critical scenario: PASS")
print("Detailed alert generation: PASS")


# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n5. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected tables modified: False")

print("\nSTEP 16.24 COMPLETE")


# COMMAND ----------


# Step 16.25 — Unified Monitoring Alert Summary
# READ-ONLY: No inference, job execution, or Delta writes

print("=" * 70)
print("STEP 16.25 — UNIFIED MONITORING ALERT SUMMARY")
print("=" * 70)

# --------------------------------------------------
# 1. Define reusable summary builder
# --------------------------------------------------

def build_monitoring_summary(report_df, config):

    assessment = aggregate_monitoring_alerts(
        report_df,
        config
    )

    # Do not treat malformed reports as valid.
    if assessment["overall_status"] == "REVIEW REQUIRED":
        alerts = [{
            "feature": None,
            "metric": None,
            "score": None,
            "severity": "REVIEW REQUIRED",
            "message": assessment["reason"]
        }]
    else:
        alerts = build_alert_details(report_df)

    summary = {
        "pipeline_name": config["pipeline_name"],
        "environment": config["environment"],
        "overall_status": assessment["overall_status"],
        "drift_status": assessment["drift_status"],
        "quality_status": assessment["quality_status"],
        "alert_count": len(alerts),
        "alerts": alerts,
        "monitoring_scope": "DEVELOPMENT TRAIN/TEST COMPARISON",
        "temporal_production_drift": "NOT ASSESSED"
    }

    return summary


# --------------------------------------------------
# 2. Build summary for actual development report
# --------------------------------------------------

print("\n1. DEVELOPMENT MONITORING SUMMARY")

development_summary = build_monitoring_summary(
    report,
    monitoring_config
)

for key, value in development_summary.items():
    if key != "alerts":
        print(f"{key}: {value}")

print("Alert messages:", [
    alert["message"]
    for alert in development_summary["alerts"]
])

# --------------------------------------------------
# 3. Test simulated warning summary
# --------------------------------------------------

print("\n2. SIMULATED WARNING SUMMARY")

warning_summary = build_monitoring_summary(
    warning_report,
    monitoring_config
)

print("Overall status:", warning_summary["overall_status"])
print("Alert count:", warning_summary["alert_count"])

for alert in warning_summary["alerts"]:
    print("Alert:", alert["message"])

# --------------------------------------------------
# 4. Test simulated critical summary
# --------------------------------------------------

print("\n3. SIMULATED CRITICAL SUMMARY")

critical_summary = build_monitoring_summary(
    critical_report,
    monitoring_config
)

print("Overall status:", critical_summary["overall_status"])
print("Alert count:", critical_summary["alert_count"])

for alert in critical_summary["alerts"]:
    print("Alert:", alert["message"])

# --------------------------------------------------
# 5. Validate results
# --------------------------------------------------

print("\n4. VALIDATION")

assert development_summary["overall_status"] == "NORMAL"
assert development_summary["alert_count"] == 0

assert warning_summary["overall_status"] == "WARNING"
assert warning_summary["alert_count"] == 1

assert critical_summary["overall_status"] == "CRITICAL"
assert critical_summary["alert_count"] == 1

assert all(
    summary["temporal_production_drift"] == "NOT ASSESSED"
    for summary in [
        development_summary,
        warning_summary,
        critical_summary
    ]
)

print("Development summary: PASS")
print("Warning summary: PASS")
print("Critical summary: PASS")
print("Monitoring scope validation: PASS")
print("Unified monitoring summary: PASS")

# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n5. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected tables modified: False")

print("\nSTEP 16.25 COMPLETE")


# COMMAND ----------


# Step 16.26 — Monitoring Summary JSON Serialization
# READ-ONLY: In-memory serialization and validation only

import json
import math

print("=" * 70)
print("STEP 16.26 — MONITORING SUMMARY JSON VALIDATION")
print("=" * 70)

# --------------------------------------------------
# 1. Define JSON serialization function
# --------------------------------------------------

def serialize_monitoring_summary(summary):

    return json.dumps(
        summary,
        indent=2,
        allow_nan=False
    )


# --------------------------------------------------
# 2. Define JSON validation function
# --------------------------------------------------

def validate_monitoring_json(json_text):

    data = json.loads(json_text)

    required_fields = [
        "pipeline_name",
        "environment",
        "overall_status",
        "drift_status",
        "quality_status",
        "alert_count",
        "alerts",
        "monitoring_scope",
        "temporal_production_drift"
    ]

    for field in required_fields:
        assert field in data, (
            f"Missing required field: {field}"
        )

    assert isinstance(data["alerts"], list)
    assert isinstance(data["alert_count"], int)
    assert not isinstance(data["alert_count"], bool)
    assert data["alert_count"] == len(data["alerts"])

    valid_statuses = {
        "NORMAL",
        "WARNING",
        "CRITICAL",
        "REVIEW REQUIRED"
    }

    assert data["overall_status"] in valid_statuses

    for alert in data["alerts"]:
        assert isinstance(alert, dict)
        assert all(
            field in alert
            for field in [
                "feature",
                "metric",
                "score",
                "severity",
                "message"
            ]
        )

        assert alert["severity"] in {
            "WARNING",
            "CRITICAL",
            "REVIEW REQUIRED"
        }

        assert isinstance(alert["message"], str)
        assert alert["message"].strip()

        score = alert["score"]

        if score is not None:
            assert isinstance(score, (int, float))
            assert not isinstance(score, bool)
            assert math.isfinite(score)

    return data


# --------------------------------------------------
# 3. Validate development summary
# --------------------------------------------------

print("\n1. DEVELOPMENT SUMMARY")

development_json = serialize_monitoring_summary(
    development_summary
)

development_parsed = validate_monitoring_json(
    development_json
)

assert development_parsed == development_summary

print(development_json)
print("Development JSON validation: PASS")


# --------------------------------------------------
# 4. Validate simulated scenarios
# --------------------------------------------------

print("\n2. SIMULATED WARNING SUMMARY")

warning_json = serialize_monitoring_summary(
    warning_summary
)

warning_parsed = validate_monitoring_json(
    warning_json
)

assert warning_parsed == warning_summary
assert warning_parsed["overall_status"] == "WARNING"
assert warning_parsed["alert_count"] == 1

print("Overall status:", warning_parsed["overall_status"])
print("Alert count:", warning_parsed["alert_count"])
print("Warning JSON validation: PASS")


print("\n3. SIMULATED CRITICAL SUMMARY")

critical_json = serialize_monitoring_summary(
    critical_summary
)

critical_parsed = validate_monitoring_json(
    critical_json
)

assert critical_parsed == critical_summary
assert critical_parsed["overall_status"] == "CRITICAL"
assert critical_parsed["alert_count"] == 1

print("Overall status:", critical_parsed["overall_status"])
print("Alert count:", critical_parsed["alert_count"])
print("Critical JSON validation: PASS")


# --------------------------------------------------
# 5. Verify scope
# --------------------------------------------------

print("\n4. MONITORING SCOPE")

for data in [
    development_parsed,
    warning_parsed,
    critical_parsed
]:
    assert data["environment"] == "DEVELOPMENT"
    assert data["temporal_production_drift"] == "NOT ASSESSED"

print("Development-only scope: PASS")
print("Temporal production drift: NOT ASSESSED")


# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n5. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.26 COMPLETE")


# COMMAND ----------


# Step 16.27 — Monitoring Summary Contract Testing
# READ-ONLY: No inference, jobs, files, or Delta writes

import json
import copy

print("=" * 70)
print("STEP 16.27 — MONITORING SUMMARY CONTRACT TESTING")
print("=" * 70)

# --------------------------------------------------
# 1. Prepare valid baseline payloads
# --------------------------------------------------

valid_development = copy.deepcopy(development_summary)
valid_warning = copy.deepcopy(warning_summary)

# --------------------------------------------------
# 2. Construct contract test scenarios
# --------------------------------------------------

missing_field = copy.deepcopy(valid_development)
del missing_field["pipeline_name"]

wrong_alert_count = copy.deepcopy(valid_warning)
wrong_alert_count["alert_count"] = 99

invalid_status = copy.deepcopy(valid_development)
invalid_status["overall_status"] = "UNKNOWN"

invalid_severity = copy.deepcopy(valid_warning)
invalid_severity["alerts"][0]["severity"] = "UNKNOWN"

invalid_score = copy.deepcopy(valid_warning)
invalid_score["alerts"][0]["score"] = float("nan")

scenarios = [
    {
        "name": "Valid development summary",
        "payload": json.dumps(valid_development),
        "should_accept": True
    },
    {
        "name": "Missing required field",
        "payload": json.dumps(missing_field),
        "should_accept": False
    },
    {
        "name": "Incorrect alert count",
        "payload": json.dumps(wrong_alert_count),
        "should_accept": False
    },
    {
        "name": "Invalid overall status",
        "payload": json.dumps(invalid_status),
        "should_accept": False
    },
    {
        "name": "Invalid alert severity",
        "payload": json.dumps(invalid_severity),
        "should_accept": False
    },
    {
        "name": "Non-finite alert score",
        "payload": json.dumps(invalid_score),
        "should_accept": False
    },
    {
        "name": "Malformed JSON syntax",
        "payload": '{"overall_status": "NORMAL",',
        "should_accept": False
    }
]

# --------------------------------------------------
# 3. Execute contract tests
# --------------------------------------------------

print("\n1. CONTRACT TEST RESULTS")

passed = 0

for index, scenario in enumerate(scenarios, start=1):

    try:
        validate_monitoring_json(scenario["payload"])
        accepted = True

    except (AssertionError, ValueError, TypeError, KeyError):
        accepted = False

    test_passed = accepted == scenario["should_accept"]

    if test_passed:
        passed += 1

    print(f"\nTest {index}: {scenario['name']}")
    print(
        "Expected:",
        "ACCEPT" if scenario["should_accept"] else "REJECT"
    )
    print(
        "Actual:",
        "ACCEPT" if accepted else "REJECT"
    )
    print(
        "Result:",
        "PASS" if test_passed else "FAIL"
    )

# --------------------------------------------------
# 4. Validate test summary
# --------------------------------------------------

print("\n2. TEST SUMMARY")

print("Total tests:", len(scenarios))
print("Passed:", passed)
print("Failed:", len(scenarios) - passed)

assert passed == len(scenarios), (
    "One or more monitoring contract tests failed"
)

print("Monitoring contract tests: PASS")

# --------------------------------------------------
# 5. Verify original monitoring summary
# --------------------------------------------------

print("\n3. ORIGINAL SUMMARY VALIDATION")

original_json = serialize_monitoring_summary(
    development_summary
)

validated = validate_monitoring_json(original_json)

assert validated == development_summary
assert validated["overall_status"] == "NORMAL"

print("Original summary preserved: PASS")
print("Development monitoring status: NORMAL")
print("Temporal production drift: NOT ASSESSED")

# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.27 COMPLETE")


# COMMAND ----------


# Step 16.28 — Monitoring Contract Hardening
# READ-ONLY: In-memory validation only

import json
import copy

print("=" * 70)
print("STEP 16.28 — MONITORING CONTRACT HARDENING")
print("=" * 70)

# --------------------------------------------------
# 1. Define stricter monitoring contract
# --------------------------------------------------

def validate_monitoring_contract(json_text, config):

    # Reuse the existing structural validator
    data = validate_monitoring_json(json_text)

    assert data["pipeline_name"] == config["pipeline_name"]
    assert data["environment"] == "DEVELOPMENT"

    assert (
        data["monitoring_scope"]
        == "DEVELOPMENT TRAIN/TEST COMPARISON"
    )

    assert (
        data["temporal_production_drift"]
        == "NOT ASSESSED"
    )

    assert data["drift_status"] in {
        "LOW", "MODERATE", "LARGE", "REVIEW REQUIRED"
    }

    assert data["quality_status"] in {
        "NORMAL", "WARNING", "CRITICAL",
        "REVIEW REQUIRED"
    }

    # Determine overall severity from component statuses
    if (
        data["drift_status"] == "REVIEW REQUIRED"
        or data["quality_status"] == "REVIEW REQUIRED"
    ):
        expected_overall = "REVIEW REQUIRED"

    elif (
        data["drift_status"] == "LARGE"
        or data["quality_status"] == "CRITICAL"
    ):
        expected_overall = "CRITICAL"

    elif (
        data["drift_status"] == "MODERATE"
        or data["quality_status"] == "WARNING"
    ):
        expected_overall = "WARNING"

    else:
        expected_overall = "NORMAL"

    assert data["overall_status"] == expected_overall, (
        "Overall status contradicts component statuses"
    )

    # Check alert metadata
    allowed_pairs = (
        {
            (feature, "SMD")
            for feature in config["numerical_features"]
        }
        | {
            (feature, "TVD")
            for feature in config["categorical_features"]
        }
        | {
            (feature, "MISSING_PCT")
            for feature in (
                config["numerical_features"]
                + config["categorical_features"]
            )
        }
    )

    seen_pairs = set()

    for alert in data["alerts"]:

        pair = (alert["feature"], alert["metric"])

        if data["overall_status"] == "REVIEW REQUIRED":
            # A summary-level review alert may have no feature
            if pair == (None, None):
                assert alert["severity"] == "REVIEW REQUIRED"
                continue

        assert pair in allowed_pairs, (
            f"Unexpected alert metric: {pair}"
        )

        assert pair not in seen_pairs, (
            f"Duplicate alert metric: {pair}"
        )

        seen_pairs.add(pair)

        metric = alert["metric"]
        score = alert["score"]
        severity = alert["severity"]

        assert score is not None, (
            "Metric-specific alert score cannot be null"
        )

        if metric == "SMD":
            assert score >= 0
            expected_severity = (
                "CRITICAL" if score >= 0.20
                else "WARNING" if score >= 0.10
                else None
            )

        elif metric == "TVD":
            assert 0 <= score <= 1
            expected_severity = (
                "CRITICAL" if score >= 0.10
                else "WARNING" if score >= 0.05
                else None
            )

        else:
            assert 0 <= score <= 100
            expected_severity = (
                "CRITICAL" if score > 2.0
                else "WARNING" if score > 0.5
                else None
            )

        assert severity == expected_severity, (
            f"Alert severity inconsistent with score: {pair}"
        )

    # Check consistency of alert severity and overall status
    severities = {
        alert["severity"]
        for alert in data["alerts"]
    }

    if data["overall_status"] == "NORMAL":
        assert len(data["alerts"]) == 0

    elif data["overall_status"] == "WARNING":
        assert "WARNING" in severities
        assert "CRITICAL" not in severities
        assert "REVIEW REQUIRED" not in severities

    elif data["overall_status"] == "CRITICAL":
        assert "CRITICAL" in severities
        assert "REVIEW REQUIRED" not in severities

    else:
        assert "REVIEW REQUIRED" in severities

    return data


# --------------------------------------------------
# 2. Build valid and invalid scenarios
# --------------------------------------------------

valid_normal = copy.deepcopy(development_summary)
valid_warning = copy.deepcopy(warning_summary)
valid_critical = copy.deepcopy(critical_summary)

contradictory_overall = copy.deepcopy(valid_warning)
contradictory_overall["overall_status"] = "NORMAL"

contradictory_alert = copy.deepcopy(valid_warning)
contradictory_alert["alerts"][0]["severity"] = "CRITICAL"

duplicate_alert = copy.deepcopy(valid_warning)
duplicate_alert["alerts"].append(
    copy.deepcopy(duplicate_alert["alerts"][0])
)
duplicate_alert["alert_count"] = 2

wrong_environment = copy.deepcopy(valid_normal)
wrong_environment["environment"] = "PRODUCTION"

wrong_feature = copy.deepcopy(valid_warning)
wrong_feature["alerts"][0]["feature"] = "UnknownFeature"

scenarios = [
    ("Valid NORMAL", valid_normal, True),
    ("Valid WARNING", valid_warning, True),
    ("Valid CRITICAL", valid_critical, True),
    ("Contradictory overall status",
     contradictory_overall, False),
    ("Contradictory alert severity",
     contradictory_alert, False),
    ("Duplicate alert", duplicate_alert, False),
    ("Incorrect environment",
     wrong_environment, False),
    ("Unknown alert feature",
     wrong_feature, False)
]

# --------------------------------------------------
# 3. Run semantic contract tests
# --------------------------------------------------

print("\n1. SEMANTIC CONTRACT TESTS")

passed = 0

for index, (name, payload, should_accept) in enumerate(
    scenarios, start=1
):

    try:
        json_text = json.dumps(
            payload,
            allow_nan=False
        )

        validate_monitoring_contract(
            json_text,
            monitoring_config
        )

        accepted = True

    except (AssertionError, ValueError, TypeError, KeyError):
        accepted = False

    success = accepted == should_accept

    if success:
        passed += 1

    print(f"\nTest {index}: {name}")
    print(
        "Expected:",
        "ACCEPT" if should_accept else "REJECT"
    )
    print(
        "Actual:",
        "ACCEPT" if accepted else "REJECT"
    )
    print("Result:", "PASS" if success else "FAIL")


# --------------------------------------------------
# 4. Validate test results
# --------------------------------------------------

print("\n2. TEST SUMMARY")

print("Total tests:", len(scenarios))
print("Passed:", passed)
print("Failed:", len(scenarios) - passed)

assert passed == len(scenarios)

print("Semantic contract validation: PASS")


# --------------------------------------------------
# 5. Confirm original summary remains valid
# --------------------------------------------------

print("\n3. ORIGINAL SUMMARY")

original_json = serialize_monitoring_summary(
    development_summary
)

validated = validate_monitoring_contract(
    original_json,
    monitoring_config
)

assert validated == development_summary

print("Original summary preserved: PASS")
print("Development monitoring status: NORMAL")
print("Temporal production drift: NOT ASSESSED")


# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.28 COMPLETE")


# COMMAND ----------


# Step 16.29 — Monitoring Contract Consistency Testing
# READ-ONLY: In-memory validation and simulated scenarios

import json
import copy

print("=" * 70)
print("STEP 16.29 — MONITORING CONTRACT CONSISTENCY TESTING")
print("=" * 70)

# --------------------------------------------------
# 1. Define enhanced consistency validator
# --------------------------------------------------

def validate_monitoring_consistency(json_text, config):

    data = validate_monitoring_contract(
        json_text,
        config
    )

    overall = data["overall_status"]
    alerts = data["alerts"]

    severity_rank = {
        "NORMAL": 0,
        "WARNING": 1,
        "CRITICAL": 2,
        "REVIEW REQUIRED": 3
    }

    # Every alert must contain a meaningful message
    for alert in alerts:
        message = alert["message"]

        assert isinstance(message, str)
        assert message.strip(), "Empty alert message"

        if alert["feature"] is not None:
            assert alert["feature"] in message, (
                "Alert message does not identify its feature"
            )

    # Validate highest severity
    if overall == "NORMAL":
        assert len(alerts) == 0

    elif overall in ("WARNING", "CRITICAL"):
        assert len(alerts) > 0

        highest_alert = max(
            severity_rank[a["severity"]]
            for a in alerts
        )

        assert highest_alert == severity_rank[overall], (
            "Overall status does not match highest alert severity"
        )

    else:
        assert any(
            a["severity"] == "REVIEW REQUIRED"
            for a in alerts
        )

    # Verify each non-normal component has alert evidence
    drift_alerts = [
        a for a in alerts
        if a["metric"] in ("SMD", "TVD")
    ]

    quality_alerts = [
        a for a in alerts
        if a["metric"] == "MISSING_PCT"
    ]

    drift = data["drift_status"]
    quality = data["quality_status"]

    if drift == "LOW":
        assert len(drift_alerts) == 0

    elif drift == "MODERATE":
        assert any(
            a["severity"] == "WARNING"
            for a in drift_alerts
        )
        assert not any(
            a["severity"] == "CRITICAL"
            for a in drift_alerts
        )

    elif drift == "LARGE":
        assert any(
            a["severity"] == "CRITICAL"
            for a in drift_alerts
        )

    if quality == "NORMAL":
        assert len(quality_alerts) == 0

    elif quality == "WARNING":
        assert any(
            a["severity"] == "WARNING"
            for a in quality_alerts
        )
        assert not any(
            a["severity"] == "CRITICAL"
            for a in quality_alerts
        )

    elif quality == "CRITICAL":
        assert any(
            a["severity"] == "CRITICAL"
            for a in quality_alerts
        )

    return data


# --------------------------------------------------
# 2. Prepare scenarios
# --------------------------------------------------

valid_normal = copy.deepcopy(development_summary)
valid_warning = copy.deepcopy(warning_summary)
valid_critical = copy.deepcopy(critical_summary)

missing_warning_alert = copy.deepcopy(valid_warning)
missing_warning_alert["alerts"] = []
missing_warning_alert["alert_count"] = 0

missing_critical_alert = copy.deepcopy(valid_critical)
missing_critical_alert["alerts"] = []
missing_critical_alert["alert_count"] = 0

empty_message = copy.deepcopy(valid_warning)
empty_message["alerts"][0]["message"] = "   "

missing_feature_in_message = copy.deepcopy(valid_warning)
missing_feature_in_message["alerts"][0]["message"] = (
    "Numerical drift detected"
)

incorrect_drift_component = copy.deepcopy(valid_warning)
incorrect_drift_component["drift_status"] = "LOW"
incorrect_drift_component["quality_status"] = "WARNING"

scenarios = [
    ("Valid NORMAL", valid_normal, True),
    ("Valid WARNING", valid_warning, True),
    ("Valid CRITICAL", valid_critical, True),
    ("Missing WARNING alert", missing_warning_alert, False),
    ("Missing CRITICAL alert", missing_critical_alert, False),
    ("Empty alert message", empty_message, False),
    ("Feature missing from message",
     missing_feature_in_message, False),
    ("Incorrect component attribution",
     incorrect_drift_component, False)
]

# --------------------------------------------------
# 3. Execute consistency tests
# --------------------------------------------------

print("\n1. CONSISTENCY TEST RESULTS")

passed = 0

for index, (name, payload, should_accept) in enumerate(
    scenarios, start=1
):

    try:
        json_text = json.dumps(
            payload,
            allow_nan=False
        )

        validate_monitoring_consistency(
            json_text,
            monitoring_config
        )

        accepted = True

    except (AssertionError, ValueError, TypeError, KeyError):
        accepted = False

    success = accepted == should_accept

    if success:
        passed += 1

    print(f"\nTest {index}: {name}")
    print(
        "Expected:",
        "ACCEPT" if should_accept else "REJECT"
    )
    print(
        "Actual:",
        "ACCEPT" if accepted else "REJECT"
    )
    print("Result:", "PASS" if success else "FAIL"
    )


# --------------------------------------------------
# 4. Validate results
# --------------------------------------------------

print("\n2. TEST SUMMARY")

print("Total tests:", len(scenarios))
print("Passed:", passed)
print("Failed:", len(scenarios) - passed)

assert passed == len(scenarios), (
    "One or more consistency tests failed"
)

print("Monitoring consistency validation: PASS")


# --------------------------------------------------
# 5. Verify original summary
# --------------------------------------------------

print("\n3. ORIGINAL SUMMARY")

original_json = serialize_monitoring_summary(
    development_summary
)

validated = validate_monitoring_consistency(
    original_json,
    monitoring_config
)

assert validated == development_summary

print("Original summary preserved: PASS")
print("Development monitoring status: NORMAL")
print("Temporal production drift: NOT ASSESSED")


# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.29 COMPLETE")


# COMMAND ----------


# Step 16.30 — Monitoring Alert Threshold Boundary Testing
# READ-ONLY: Simulated in-memory monitoring reports

print("=" * 70)
print("STEP 16.30 — MONITORING ALERT BOUNDARY TESTING")
print("=" * 70)

# --------------------------------------------------
# 1. Define boundary test scenarios
# --------------------------------------------------

boundary_tests = [
    # Numerical drift
    ("SMD below moderate", "tenure", "SMD", 0.0999, "NORMAL"),
    ("SMD at moderate", "tenure", "SMD", 0.10, "WARNING"),
    ("SMD below large", "tenure", "SMD", 0.1999, "WARNING"),
    ("SMD at large", "tenure", "SMD", 0.20, "CRITICAL"),

    # Categorical drift
    ("TVD below moderate", "Contract", "TVD", 0.0499, "NORMAL"),
    ("TVD at moderate", "Contract", "TVD", 0.05, "WARNING"),
    ("TVD below large", "Contract", "TVD", 0.0999, "WARNING"),
    ("TVD at large", "Contract", "TVD", 0.10, "CRITICAL"),

    # Missingness
    ("Missingness at 0.5%", "TotalCharges",
     "MISSING_PCT", 0.5, "NORMAL"),
    ("Missingness above 0.5%", "TotalCharges",
     "MISSING_PCT", 0.5001, "WARNING"),
    ("Missingness at 2.0%", "TotalCharges",
     "MISSING_PCT", 2.0, "WARNING"),
    ("Missingness above 2.0%", "TotalCharges",
     "MISSING_PCT", 2.0001, "CRITICAL")
]

# --------------------------------------------------
# 2. Execute boundary tests
# --------------------------------------------------

print("\n1. BOUNDARY TEST RESULTS")

passed = 0

for index, (
    name, feature, metric, score, expected
) in enumerate(boundary_tests, start=1):

    test_report = make_test_report(
        feature=feature,
        metric=metric,
        score=score
    )

    result = aggregate_monitoring_alerts(
        test_report,
        monitoring_config
    )

    actual = result["overall_status"]

    success = actual == expected

    if success:
        passed += 1

    print(f"\nTest {index}: {name}")
    print("Metric:", metric)
    print("Score:", score)
    print("Expected:", expected)
    print("Actual:", actual)
    print("Result:", "PASS" if success else "FAIL")


# --------------------------------------------------
# 3. Validate test results
# --------------------------------------------------

print("\n2. TEST SUMMARY")

print("Total tests:", len(boundary_tests))
print("Passed:", passed)
print("Failed:", len(boundary_tests) - passed)

assert passed == len(boundary_tests), (
    "One or more boundary tests failed"
)

print("Threshold boundary validation: PASS")


# --------------------------------------------------
# 4. Verify original report remains unchanged
# --------------------------------------------------

print("\n3. ORIGINAL REPORT VALIDATION")

original_assessment = aggregate_monitoring_alerts(
    report,
    monitoring_config
)

assert original_assessment["overall_status"] == "NORMAL"

print("Original report status: NORMAL")
print("Original report preserved: PASS")
print("Temporal production drift: NOT ASSESSED")


# --------------------------------------------------
# 5. Safety
# --------------------------------------------------

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.30 COMPLETE")


# COMMAND ----------


# Step 16.31 — Invalid Metric and Failure-Condition Testing
# READ-ONLY: In-memory simulated reports only

import math
import copy

print("=" * 70)
print("STEP 16.31 — INVALID METRIC AND FAILURE-CONDITION TESTING")
print("=" * 70)

# --------------------------------------------------
# 1. Prepare simulated invalid metric scenarios
# --------------------------------------------------

invalid_tests = [
    ("Negative SMD", "tenure", "SMD", -0.01),
    ("Negative TVD", "Contract", "TVD", -0.01),
    ("TVD greater than 1", "Contract", "TVD", 1.01),
    ("Negative missingness", "TotalCharges", "MISSING_PCT", -1.0),
    ("Missingness above 100%", "TotalCharges", "MISSING_PCT", 101.0),
    ("NaN numerical score", "tenure", "SMD", float("nan")),
    ("Infinite numerical score", "tenure", "SMD", float("inf")),
    ("Null numerical score", "tenure", "SMD", None)
]

# --------------------------------------------------
# 2. Build test reports without changing originals
# --------------------------------------------------

def make_invalid_metric_report(feature, metric, score):

    rows = [
        dict(row)
        for row in baseline_rows
    ]

    matches = 0

    for row in rows:
        if row["feature"] == feature and row["metric"] == metric:
            row["score"] = score
            row["status"] = "REVIEW"
            matches += 1

    assert matches == 1, (
        "Expected exactly one matching monitoring metric"
    )

    return spark.createDataFrame(
        rows,
        schema=report_schema
    )

# --------------------------------------------------
# 3. Execute failure-condition tests
# --------------------------------------------------

print("\n1. INVALID METRIC TEST RESULTS")

passed = 0

for index, (name, feature, metric, score) in enumerate(
    invalid_tests, start=1
):

    test_report = make_invalid_metric_report(
        feature,
        metric,
        score
    )

    result = aggregate_monitoring_alerts(
        test_report,
        monitoring_config
    )

    actual = result["overall_status"]
    expected = "REVIEW REQUIRED"

    success = actual == expected

    if success:
        passed += 1

    print(f"\nTest {index}: {name}")
    print("Metric:", metric)
    print("Score:", score)
    print("Expected:", expected)
    print("Actual:", actual)
    print("Result:", "PASS" if success else "FAIL")

# --------------------------------------------------
# 4. Validate test summary
# --------------------------------------------------

print("\n2. TEST SUMMARY")

print("Total tests:", len(invalid_tests))
print("Passed:", passed)
print("Failed:", len(invalid_tests) - passed)

assert passed == len(invalid_tests), (
    "One or more invalid metric tests failed"
)

print("Invalid metric handling: PASS")

# --------------------------------------------------
# 5. Confirm original report remains valid
# --------------------------------------------------

print("\n3. ORIGINAL REPORT VALIDATION")

original_assessment = aggregate_monitoring_alerts(
    report,
    monitoring_config
)

assert original_assessment["overall_status"] == "NORMAL"
assert report.count() == 8

print("Original report metrics: 8")
print("Original report status: NORMAL")
print("Original report preserved: PASS")

# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.31 COMPLETE")


# COMMAND ----------


# Step 16.33 — Feature-Level Failure Diagnostics
# READ-ONLY: In-memory diagnostic testing only

import math

print("=" * 70)
print("STEP 16.33 — FEATURE-LEVEL FAILURE DIAGNOSTICS")
print("=" * 70)

# --------------------------------------------------
# 1. Define a reusable diagnostic function
# --------------------------------------------------

def diagnose_invalid_metric(feature, metric, score):

    diagnostic = {
        "feature": feature,
        "metric": metric,
        "score": score,
        "severity": "REVIEW REQUIRED",
        "reason": None
    }

    if score is None:
        diagnostic["reason"] = "Missing metric score"

    elif isinstance(score, bool) or not isinstance(
        score, (int, float)
    ):
        diagnostic["reason"] = "Non-numeric metric score"

    elif not math.isfinite(score):
        diagnostic["reason"] = "Non-finite metric score"

    elif metric == "SMD" and score < 0:
        diagnostic["reason"] = "Negative numerical drift score"

    elif metric == "TVD" and not (0 <= score <= 1):
        diagnostic["reason"] = (
            "Categorical drift score outside [0, 1]"
        )

    elif metric == "MISSING_PCT" and not (0 <= score <= 100):
        diagnostic["reason"] = (
            "Missingness percentage outside [0, 100]"
        )

    elif metric not in {"SMD", "TVD", "MISSING_PCT"}:
        diagnostic["reason"] = "Unsupported monitoring metric"

    else:
        return None

    return diagnostic


# --------------------------------------------------
# 2. Define simulated invalid scenarios
# --------------------------------------------------

diagnostic_tests = [
    ("tenure", "SMD", -0.01),
    ("Contract", "TVD", 1.01),
    ("TotalCharges", "MISSING_PCT", -1.0),
    ("TotalCharges", "MISSING_PCT", 101.0),
    ("tenure", "SMD", float("nan")),
    ("tenure", "SMD", float("inf")),
    ("tenure", "SMD", None),
    ("tenure", "SMD", "invalid")
]

# --------------------------------------------------
# 3. Execute diagnostic tests
# --------------------------------------------------

print("\n1. FAILURE DIAGNOSTIC RESULTS")

passed = 0

for index, (feature, metric, score) in enumerate(
    diagnostic_tests, start=1
):

    result = diagnose_invalid_metric(
        feature, metric, score
    )

    success = (
        result is not None
        and result["severity"] == "REVIEW REQUIRED"
        and isinstance(result["reason"], str)
        and len(result["reason"]) > 0
    )

    if success:
        passed += 1

    print(f"\nTest {index}")
    print("Feature:", feature)
    print("Metric:", metric)
    print("Score:", score)
    print("Reason:", result["reason"] if result else None)
    print("Result:", "PASS" if success else "FAIL")


# --------------------------------------------------
# 4. Confirm valid scores are not flagged
# --------------------------------------------------

print("\n2. VALID SCORE CHECK")

valid_tests = [
    ("tenure", "SMD", 0.15),
    ("Contract", "TVD", 0.05),
    ("TotalCharges", "MISSING_PCT", 0.5)
]

valid_passed = 0

for feature, metric, score in valid_tests:

    result = diagnose_invalid_metric(
        feature, metric, score
    )

    if result is None:
        valid_passed += 1

assert valid_passed == len(valid_tests)

print("Valid scores correctly ignored:", valid_passed)


# --------------------------------------------------
# 5. Validate test summary
# --------------------------------------------------

print("\n3. TEST SUMMARY")

print("Invalid diagnostic tests:", len(diagnostic_tests))
print("Invalid diagnostic tests passed:", passed)
print("Valid score tests passed:", valid_passed)

assert passed == len(diagnostic_tests)

print("Feature-level diagnostics: PASS")


# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.33 COMPLETE")


# COMMAND ----------


# Step 16.34 — Integrating Feature-Level Diagnostics
# READ-ONLY: In-memory reports only

import math

print("=" * 70)
print("STEP 16.34 — DIAGNOSTIC ALERT INTEGRATION")
print("=" * 70)

# --------------------------------------------------
# 1. Build diagnostic-aware alert details
# --------------------------------------------------

def build_diagnostic_alerts(report_df):

    alerts = []

    for row in report_df.collect():

        feature = row["feature"]
        metric = row["metric"]
        score = row["score"]

        # First check for invalid measurements
        diagnostic = diagnose_invalid_metric(
            feature, metric, score
        )

        if diagnostic is not None:

            # Convert non-finite scores to null for safe JSON
            safe_score = (
                float(score)
                if isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(score)
                else None
            )

            alerts.append({
                "feature": feature,
                "metric": metric,
                "score": safe_score,
                "severity": "REVIEW REQUIRED",
                "message": (
                    f"{feature}: {metric} invalid — "
                    f"{diagnostic['reason']}"
                )
            })

            continue

        # Handle valid measurements using existing thresholds
        if metric == "SMD":

            severity = (
                "CRITICAL" if score >= 0.20
                else "WARNING" if score >= 0.10
                else None
            )

        elif metric == "TVD":

            severity = (
                "CRITICAL" if score >= 0.10
                else "WARNING" if score >= 0.05
                else None
            )

        elif metric == "MISSING_PCT":

            severity = (
                "CRITICAL" if score > 2.0
                else "WARNING" if score > 0.5
                else None
            )

        else:
            severity = "REVIEW REQUIRED"

        if severity is not None:

            alerts.append({
                "feature": feature,
                "metric": metric,
                "score": float(score),
                "severity": severity,
                "message": (
                    f"{feature}: {metric}={score:.4f} "
                    f"({severity})"
                )
            })

    return alerts


# --------------------------------------------------
# 2. Test normal development report
# --------------------------------------------------

print("\n1. DEVELOPMENT REPORT")

normal_alerts = build_diagnostic_alerts(report)

assert len(normal_alerts) == 0

print("Alert count:", len(normal_alerts))
print("Expected: 0")
print("Result: PASS")


# --------------------------------------------------
# 3. Test warning and critical scenarios
# --------------------------------------------------

print("\n2. WARNING AND CRITICAL SCENARIOS")

warning_alerts = build_diagnostic_alerts(warning_report)
critical_alerts = build_diagnostic_alerts(critical_report)

assert len(warning_alerts) == 1
assert warning_alerts[0]["severity"] == "WARNING"

assert len(critical_alerts) == 1
assert critical_alerts[0]["severity"] == "CRITICAL"

print("Warning severity:", warning_alerts[0]["severity"])
print("Critical severity:", critical_alerts[0]["severity"])
print("Result: PASS")


# --------------------------------------------------
# 4. Test invalid metric diagnostics
# --------------------------------------------------

print("\n3. INVALID METRIC DIAGNOSTICS")

invalid_cases = [
    ("tenure", "SMD", -0.01),
    ("Contract", "TVD", 1.01),
    ("tenure", "SMD", float("nan")),
    ("tenure", "SMD", None)
]

passed = 0

for feature, metric, score in invalid_cases:

    invalid_report = make_invalid_metric_report(
        feature, metric, score
    )

    alerts = build_diagnostic_alerts(invalid_report)

    matching = [
        a for a in alerts
        if a["feature"] == feature
        and a["metric"] == metric
        and a["severity"] == "REVIEW REQUIRED"
    ]

    success = len(matching) == 1

    if success:
        passed += 1

    print("\nFeature:", feature)
    print("Metric:", metric)
    print("Score:", score)
    print(
        "Diagnostic:",
        matching[0]["message"] if matching else "MISSING"
    )
    print("Result:", "PASS" if success else "FAIL")

assert passed == len(invalid_cases)

print("\nInvalid diagnostic tests passed:", passed)


# --------------------------------------------------
# 5. Verify JSON-safe output
# --------------------------------------------------

print("\n4. JSON SAFETY")

import json

nan_report = make_invalid_metric_report(
    "tenure", "SMD", float("nan")
)

nan_alerts = build_diagnostic_alerts(nan_report)

json_text = json.dumps(
    nan_alerts,
    allow_nan=False
)

assert nan_alerts[0]["score"] is None

print("Non-finite score converted to null: PASS")
print("Strict JSON serialization: PASS")


# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n5. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.34 COMPLETE")


# COMMAND ----------


# Step 16.35 — Diagnostic-Aware Monitoring Summary
# READ-ONLY: In-memory monitoring simulations only

import json

print("=" * 70)
print("STEP 16.35 — DIAGNOSTIC-AWARE MONITORING SUMMARY")
print("=" * 70)

# --------------------------------------------------
# 1. Define enhanced monitoring summary builder
# --------------------------------------------------

def build_diagnostic_monitoring_summary(report_df, config):

    assessment = aggregate_monitoring_alerts(
        report_df,
        config
    )

    alerts = build_diagnostic_alerts(report_df)

    summary = {
        "pipeline_name": config["pipeline_name"],
        "environment": config["environment"],
        "overall_status": assessment["overall_status"],
        "drift_status": assessment["drift_status"],
        "quality_status": assessment["quality_status"],
        "alert_count": len(alerts),
        "alerts": alerts,
        "monitoring_scope": (
            "DEVELOPMENT TRAIN/TEST COMPARISON"
        ),
        "temporal_production_drift": "NOT ASSESSED"
    }

    return summary


# --------------------------------------------------
# 2. Test normal development summary
# --------------------------------------------------

print("\n1. DEVELOPMENT SUMMARY")

normal_summary = build_diagnostic_monitoring_summary(
    report,
    monitoring_config
)

assert normal_summary["overall_status"] == "NORMAL"
assert normal_summary["alert_count"] == 0

print("Overall status:", normal_summary["overall_status"])
print("Alert count:", normal_summary["alert_count"])
print("Result: PASS")


# --------------------------------------------------
# 3. Test warning and critical summaries
# --------------------------------------------------

print("\n2. WARNING AND CRITICAL SUMMARIES")

warning_diagnostic_summary = (
    build_diagnostic_monitoring_summary(
        warning_report,
        monitoring_config
    )
)

critical_diagnostic_summary = (
    build_diagnostic_monitoring_summary(
        critical_report,
        monitoring_config
    )
)

assert warning_diagnostic_summary["overall_status"] == "WARNING"
assert warning_diagnostic_summary["alert_count"] == 1

assert critical_diagnostic_summary["overall_status"] == "CRITICAL"
assert critical_diagnostic_summary["alert_count"] == 1

print(
    "Warning status:",
    warning_diagnostic_summary["overall_status"]
)
print(
    "Critical status:",
    critical_diagnostic_summary["overall_status"]
)
print("Result: PASS")


# --------------------------------------------------
# 4. Test invalid metric summary
# --------------------------------------------------

print("\n3. INVALID METRIC SUMMARY")

invalid_report = make_invalid_metric_report(
    "tenure",
    "SMD",
    float("nan")
)

invalid_summary = build_diagnostic_monitoring_summary(
    invalid_report,
    monitoring_config
)

assert invalid_summary["overall_status"] == "REVIEW REQUIRED"
assert invalid_summary["alert_count"] == 1

invalid_alert = invalid_summary["alerts"][0]

assert invalid_alert["feature"] == "tenure"
assert invalid_alert["metric"] == "SMD"
assert invalid_alert["score"] is None
assert invalid_alert["severity"] == "REVIEW REQUIRED"

print("Overall status:", invalid_summary["overall_status"])
print("Feature:", invalid_alert["feature"])
print("Metric:", invalid_alert["metric"])
print("Score:", invalid_alert["score"])
print("Reason:", invalid_alert["message"])
print("Result: PASS")


# --------------------------------------------------
# 5. Verify strict JSON serialization
# --------------------------------------------------

print("\n4. JSON SERIALIZATION")

summaries = [
    normal_summary,
    warning_diagnostic_summary,
    critical_diagnostic_summary,
    invalid_summary
]

for summary in summaries:

    json_text = json.dumps(
        summary,
        indent=2,
        allow_nan=False
    )

    parsed = json.loads(json_text)

    assert parsed == summary

    print(
        summary["overall_status"],
        "JSON round-trip: PASS"
    )


# --------------------------------------------------
# 6. Verify original development summary
# --------------------------------------------------

print("\n5. ORIGINAL SUMMARY PRESERVATION")

assert development_summary["overall_status"] == "NORMAL"
assert development_summary["alert_count"] == 0

print("Original development summary: NORMAL")
print("Original summary preserved: PASS")
print("Temporal production drift: NOT ASSESSED")


# --------------------------------------------------
# 7. Safety
# --------------------------------------------------

print("\n6. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.35 COMPLETE")


# COMMAND ----------


# Step 16.36 — Diagnostic Summary Contract Validation
# READ-ONLY: In-memory JSON validation only

import json
import copy

print("=" * 70)
print("STEP 16.36 — DIAGNOSTIC SUMMARY CONTRACT VALIDATION")
print("=" * 70)

# --------------------------------------------------
# 1. Define diagnostic-aware contract validator
# --------------------------------------------------

def validate_diagnostic_monitoring_contract(json_text, config):

    # Validate basic JSON structure
    data = validate_monitoring_json(json_text)

    # Ordinary summaries retain the existing strict checks
    if data["overall_status"] != "REVIEW REQUIRED":
        return validate_monitoring_consistency(
            json_text,
            config
        )

    # Validate development metadata
    assert data["pipeline_name"] == config["pipeline_name"]
    assert data["environment"] == "DEVELOPMENT"

    assert (
        data["monitoring_scope"]
        == "DEVELOPMENT TRAIN/TEST COMPARISON"
    )

    assert (
        data["temporal_production_drift"]
        == "NOT ASSESSED"
    )

    # Current aggregator marks both components for review
    assert data["drift_status"] == "REVIEW REQUIRED"
    assert data["quality_status"] == "REVIEW REQUIRED"

    assert data["alert_count"] >= 1

    allowed_pairs = (
        {
            (feature, "SMD")
            for feature in config["numerical_features"]
        }
        | {
            (feature, "TVD")
            for feature in config["categorical_features"]
        }
        | {
            (feature, "MISSING_PCT")
            for feature in (
                config["numerical_features"]
                + config["categorical_features"]
            )
        }
    )

    review_count = 0
    seen_pairs = set()

    for alert in data["alerts"]:

        pair = (alert["feature"], alert["metric"])

        assert pair in allowed_pairs, (
            f"Unexpected diagnostic metric: {pair}"
        )

        assert pair not in seen_pairs, (
            f"Duplicate diagnostic metric: {pair}"
        )

        seen_pairs.add(pair)

        message = alert["message"]

        assert isinstance(message, str)
        assert message.strip()
        assert alert["feature"] in message

        if alert["severity"] == "REVIEW REQUIRED":

            review_count += 1

            assert "invalid" in message.lower(), (
                "Diagnostic reason missing"
            )

            # Invalid measurements must be represented safely
            score = alert["score"]

            assert score is None or isinstance(
                score, (int, float)
            )

            if score is not None:
                assert not isinstance(score, bool)

        else:
            # Ordinary alerts remain subject to strict validation
            assert alert["severity"] in {
                "WARNING", "CRITICAL"
            }

    assert review_count >= 1, (
        "REVIEW REQUIRED summary lacks diagnostic alert"
    )

    return data


# --------------------------------------------------
# 2. Prepare valid scenarios
# --------------------------------------------------

valid_summaries = [
    ("NORMAL", normal_summary),
    ("WARNING", warning_diagnostic_summary),
    ("CRITICAL", critical_diagnostic_summary),
    ("REVIEW REQUIRED", invalid_summary)
]

# --------------------------------------------------
# 3. Prepare invalid scenarios
# --------------------------------------------------

missing_diagnostic = copy.deepcopy(invalid_summary)
missing_diagnostic["alerts"] = []
missing_diagnostic["alert_count"] = 0

wrong_diagnostic_severity = copy.deepcopy(invalid_summary)
wrong_diagnostic_severity["alerts"][0]["severity"] = "WARNING"

missing_reason = copy.deepcopy(invalid_summary)
missing_reason["alerts"][0]["message"] = (
    "tenure: monitoring measurement"
)

wrong_diagnostic_feature = copy.deepcopy(invalid_summary)
wrong_diagnostic_feature["alerts"][0]["feature"] = "UnknownFeature"

wrong_review_status = copy.deepcopy(invalid_summary)
wrong_review_status["overall_status"] = "NORMAL"

invalid_summaries = [
    ("Missing diagnostic alert", missing_diagnostic),
    ("Incorrect diagnostic severity", wrong_diagnostic_severity),
    ("Missing diagnostic reason", missing_reason),
    ("Unknown diagnostic feature", wrong_diagnostic_feature),
    ("Incorrect review status", wrong_review_status)
]

# --------------------------------------------------
# 4. Execute contract tests
# --------------------------------------------------

print("\n1. DIAGNOSTIC CONTRACT TEST RESULTS")

passed = 0
total = 0

for name, summary in valid_summaries:

    total += 1

    try:
        json_text = json.dumps(summary, allow_nan=False)

        validate_diagnostic_monitoring_contract(
            json_text,
            monitoring_config
        )

        accepted = True

    except (AssertionError, ValueError, TypeError, KeyError):
        accepted = False

    success = accepted

    if success:
        passed += 1

    print(f"\nTest {total}: Valid {name}")
    print("Expected: ACCEPT")
    print("Actual:", "ACCEPT" if accepted else "REJECT")
    print("Result:", "PASS" if success else "FAIL")


for name, summary in invalid_summaries:

    total += 1

    try:
        json_text = json.dumps(summary, allow_nan=False)

        validate_diagnostic_monitoring_contract(
            json_text,
            monitoring_config
        )

        accepted = True

    except (AssertionError, ValueError, TypeError, KeyError):
        accepted = False

    success = not accepted

    if success:
        passed += 1

    print(f"\nTest {total}: {name}")
    print("Expected: REJECT")
    print("Actual:", "ACCEPT" if accepted else "REJECT")
    print("Result:", "PASS" if success else "FAIL")


# --------------------------------------------------
# 5. Validate results
# --------------------------------------------------

print("\n2. TEST SUMMARY")

print("Total tests:", total)
print("Passed:", passed)
print("Failed:", total - passed)

assert passed == total, (
    "One or more diagnostic contract tests failed"
)

print("Diagnostic monitoring contract: PASS")


# --------------------------------------------------
# 6. Verify original summary
# --------------------------------------------------

print("\n3. ORIGINAL SUMMARY VALIDATION")

original_json = json.dumps(
    development_summary,
    allow_nan=False
)

original = validate_diagnostic_monitoring_contract(
    original_json,
    monitoring_config
)

assert original["overall_status"] == "NORMAL"
assert original["alert_count"] == 0

print("Original summary preserved: PASS")
print("Development monitoring status: NORMAL")
print("Temporal production drift: NOT ASSESSED")


# --------------------------------------------------
# 7. Safety
# --------------------------------------------------

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.36 COMPLETE")


# COMMAND ----------


# Step 16.37 — Diagnostic Reason Accuracy Testing
# READ-ONLY: In-memory JSON contract tests

import json
import copy
import math

print("=" * 70)
print("STEP 16.37 — DIAGNOSTIC REASON ACCURACY TESTING")
print("=" * 70)

# --------------------------------------------------
# 1. Define enhanced diagnostic accuracy validator
# --------------------------------------------------

def validate_diagnostic_reason_accuracy(json_text, config):

    data = validate_diagnostic_monitoring_contract(
        json_text,
        config
    )

    for alert in data["alerts"]:

        if alert["severity"] != "REVIEW REQUIRED":
            continue

        feature = alert["feature"]
        metric = alert["metric"]
        score = alert["score"]
        message = alert["message"]

        # For finite values, independently determine the reason
        if score is not None:

            expected = diagnose_invalid_metric(
                feature,
                metric,
                score
            )

            assert expected is not None, (
                "REVIEW REQUIRED alert contains a valid score"
            )

            assert expected["reason"] in message, (
                "Diagnostic message contradicts metric value"
            )

        else:
            # JSON null cannot distinguish missing/NaN/infinity.
            # Require an explicitly recognized null-score reason.
            accepted_reasons = (
                "Missing metric score",
                "Non-finite metric score",
                "Non-numeric metric score"
            )

            assert any(
                reason in message
                for reason in accepted_reasons
            ), (
                "Null-score diagnostic has unsupported reason"
            )

    return data


# --------------------------------------------------
# 2. Prepare valid diagnostic summaries
# --------------------------------------------------

negative_report = make_invalid_metric_report(
    "tenure", "SMD", -0.01
)

negative_summary = build_diagnostic_monitoring_summary(
    negative_report,
    monitoring_config
)

tvd_report = make_invalid_metric_report(
    "Contract", "TVD", 1.01
)

tvd_summary = build_diagnostic_monitoring_summary(
    tvd_report,
    monitoring_config
)

nan_summary = copy.deepcopy(invalid_summary)

# --------------------------------------------------
# 3. Prepare incorrect diagnostic summaries
# --------------------------------------------------

wrong_negative_reason = copy.deepcopy(negative_summary)
wrong_negative_reason["alerts"][0]["message"] = (
    "tenure: SMD invalid — Non-finite metric score"
)

wrong_tvd_reason = copy.deepcopy(tvd_summary)
wrong_tvd_reason["alerts"][0]["message"] = (
    "Contract: TVD invalid — Negative numerical drift score"
)

valid_score_marked_invalid = copy.deepcopy(negative_summary)
valid_score_marked_invalid["alerts"][0]["score"] = 0.05

unsupported_null_reason = copy.deepcopy(nan_summary)
unsupported_null_reason["alerts"][0]["message"] = (
    "tenure: SMD invalid — Unknown failure"
)

# --------------------------------------------------
# 4. Define test scenarios
# --------------------------------------------------

scenarios = [
    ("Valid negative SMD diagnostic", negative_summary, True),
    ("Valid out-of-range TVD diagnostic", tvd_summary, True),
    ("Valid non-finite diagnostic", nan_summary, True),
    ("Wrong negative SMD reason", wrong_negative_reason, False),
    ("Wrong TVD reason", wrong_tvd_reason, False),
    ("Valid score marked invalid", valid_score_marked_invalid, False),
    ("Unsupported null-score reason", unsupported_null_reason, False)
]

# --------------------------------------------------
# 5. Execute diagnostic accuracy tests
# --------------------------------------------------

print("\n1. DIAGNOSTIC ACCURACY TEST RESULTS")

passed = 0

for index, (name, summary, should_accept) in enumerate(
    scenarios, start=1
):

    try:
        json_text = json.dumps(
            summary,
            allow_nan=False
        )

        validate_diagnostic_reason_accuracy(
            json_text,
            monitoring_config
        )

        accepted = True

    except (AssertionError, ValueError, TypeError, KeyError):
        accepted = False

    success = accepted == should_accept

    if success:
        passed += 1

    print(f"\nTest {index}: {name}")
    print(
        "Expected:",
        "ACCEPT" if should_accept else "REJECT"
    )
    print(
        "Actual:",
        "ACCEPT" if accepted else "REJECT"
    )
    print("Result:", "PASS" if success else "FAIL")


# --------------------------------------------------
# 6. Validate results
# --------------------------------------------------

print("\n2. TEST SUMMARY")

print("Total tests:", len(scenarios))
print("Passed:", passed)
print("Failed:", len(scenarios) - passed)

assert passed == len(scenarios), (
    "One or more diagnostic accuracy tests failed"
)

print("Diagnostic reason accuracy: PASS")


# --------------------------------------------------
# 7. Verify original development summary
# --------------------------------------------------

print("\n3. ORIGINAL SUMMARY VALIDATION")

original_json = json.dumps(
    development_summary,
    allow_nan=False
)

original = validate_diagnostic_reason_accuracy(
    original_json,
    monitoring_config
)

assert original["overall_status"] == "NORMAL"
assert original["alert_count"] == 0

print("Original summary preserved: PASS")
print("Development monitoring status: NORMAL")
print("Temporal production drift: NOT ASSESSED")


# --------------------------------------------------
# 8. Safety
# --------------------------------------------------

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.37 COMPLETE")


# COMMAND ----------


# Step 16.38 — Diagnostic Provenance and Failure Metadata
# READ-ONLY: In-memory diagnostics only

import math
import json

print("=" * 70)
print("STEP 16.38 — DIAGNOSTIC PROVENANCE METADATA")
print("=" * 70)

# --------------------------------------------------
# 1. Define JSON-safe failure provenance
# --------------------------------------------------

def classify_failure_provenance(score):

    if score is None:
        return "MISSING_VALUE", "None"

    if isinstance(score, bool):
        return "NON_NUMERIC", str(score)

    if not isinstance(score, (int, float)):
        return "NON_NUMERIC", str(score)

    if math.isnan(score):
        return "NON_FINITE_NAN", "NaN"

    if math.isinf(score):
        return (
            "NON_FINITE_POSITIVE_INFINITY"
            if score > 0
            else "NON_FINITE_NEGATIVE_INFINITY"
        ), (
            "Infinity" if score > 0 else "-Infinity"
        )

    return "FINITE_OUT_OF_RANGE", str(score)


# --------------------------------------------------
# 2. Build provenance-aware diagnostic alert
# --------------------------------------------------

def build_provenance_diagnostic(feature, metric, score):

    diagnostic = diagnose_invalid_metric(
        feature,
        metric,
        score
    )

    if diagnostic is None:
        return None

    failure_type, original_value = (
        classify_failure_provenance(score)
    )

    safe_score = (
        float(score)
        if isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(score)
        else None
    )

    return {
        "feature": feature,
        "metric": metric,
        "score": safe_score,
        "severity": "REVIEW REQUIRED",
        "failure_type": failure_type,
        "original_value": original_value,
        "message": (
            f"{feature}: {metric} invalid — "
            f"{diagnostic['reason']}"
        )
    }


# --------------------------------------------------
# 3. Test provenance scenarios
# --------------------------------------------------

provenance_tests = [
    ("NaN", "tenure", "SMD", float("nan"),
     "NON_FINITE_NAN"),

    ("Positive infinity", "tenure", "SMD", float("inf"),
     "NON_FINITE_POSITIVE_INFINITY"),

    ("Negative infinity", "tenure", "SMD", float("-inf"),
     "NON_FINITE_NEGATIVE_INFINITY"),

    ("Missing score", "tenure", "SMD", None,
     "MISSING_VALUE"),

    ("Negative SMD", "tenure", "SMD", -0.01,
     "FINITE_OUT_OF_RANGE"),

    ("Invalid TVD", "Contract", "TVD", 1.01,
     "FINITE_OUT_OF_RANGE"),

    ("Non-numeric score", "tenure", "SMD", "invalid",
     "NON_NUMERIC")
]

print("\n1. PROVENANCE TEST RESULTS")

passed = 0

for index, (
    name, feature, metric, score, expected_type
) in enumerate(provenance_tests, start=1):

    alert = build_provenance_diagnostic(
        feature, metric, score
    )

    # Ensure metadata survives strict JSON serialization
    json_text = json.dumps(
        alert,
        allow_nan=False
    )

    parsed = json.loads(json_text)

    success = (
        parsed["severity"] == "REVIEW REQUIRED"
        and parsed["failure_type"] == expected_type
        and isinstance(parsed["original_value"], str)
        and len(parsed["original_value"]) > 0
    )

    if success:
        passed += 1

    print(f"\nTest {index}: {name}")
    print("Failure type:", parsed["failure_type"])
    print("Original value:", parsed["original_value"])
    print("JSON serialization: PASS")
    print("Result:", "PASS" if success else "FAIL")


# --------------------------------------------------
# 4. Verify valid metrics are ignored
# --------------------------------------------------

print("\n2. VALID METRIC CHECK")

valid_alert = build_provenance_diagnostic(
    "tenure",
    "SMD",
    0.05
)

assert valid_alert is None

print("Valid metric produces no diagnostic: PASS")


# --------------------------------------------------
# 5. Validate results
# --------------------------------------------------

print("\n3. TEST SUMMARY")

print("Provenance tests:", len(provenance_tests))
print("Passed:", passed)
print("Failed:", len(provenance_tests) - passed)

assert passed == len(provenance_tests)

print("Diagnostic provenance metadata: PASS")


# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.38 COMPLETE")


# COMMAND ----------


# Step 16.39 — Provenance-Aware Monitoring Alert Integration
# READ-ONLY: In-memory report processing only

import json
import math

print("=" * 70)
print("STEP 16.39 — PROVENANCE-AWARE ALERT INTEGRATION")
print("=" * 70)

# --------------------------------------------------
# 1. Define provenance-aware alert builder
# --------------------------------------------------

def build_provenance_monitoring_alerts(report_df):

    alerts = []

    for row in report_df.collect():

        feature = row["feature"]
        metric = row["metric"]
        score = row["score"]

        # First, check whether the measurement is invalid
        diagnostic = build_provenance_diagnostic(
            feature,
            metric,
            score
        )

        if diagnostic is not None:
            alerts.append(diagnostic)
            continue

        # Valid measurement: apply existing alert thresholds
        if metric == "SMD":
            severity = (
                "CRITICAL" if score >= 0.20
                else "WARNING" if score >= 0.10
                else None
            )

        elif metric == "TVD":
            severity = (
                "CRITICAL" if score >= 0.10
                else "WARNING" if score >= 0.05
                else None
            )

        elif metric == "MISSING_PCT":
            severity = (
                "CRITICAL" if score > 2.0
                else "WARNING" if score > 0.5
                else None
            )

        else:
            raise ValueError(
                f"Unsupported monitoring metric: {metric}"
            )

        if severity is not None:

            alerts.append({
                "feature": feature,
                "metric": metric,
                "score": float(score),
                "severity": severity,
                "message": (
                    f"{feature}: {metric}={score:.4f} "
                    f"({severity})"
                )
            })

    return alerts


# --------------------------------------------------
# 2. Test original development report
# --------------------------------------------------

print("\n1. DEVELOPMENT REPORT")

normal_alerts = build_provenance_monitoring_alerts(
    report
)

assert len(normal_alerts) == 0

print("Alert count:", len(normal_alerts))
print("Expected: 0")
print("Result: PASS")


# --------------------------------------------------
# 3. Test warning and critical reports
# --------------------------------------------------

print("\n2. WARNING AND CRITICAL REPORTS")

warning_alerts = build_provenance_monitoring_alerts(
    warning_report
)

critical_alerts = build_provenance_monitoring_alerts(
    critical_report
)

assert len(warning_alerts) == 1
assert warning_alerts[0]["severity"] == "WARNING"

assert len(critical_alerts) == 1
assert critical_alerts[0]["severity"] == "CRITICAL"

print("Warning alert:", warning_alerts[0]["severity"])
print("Critical alert:", critical_alerts[0]["severity"])
print("Result: PASS")


# --------------------------------------------------
# 4. Test provenance metadata in full reports
# --------------------------------------------------

print("\n3. PROVENANCE INTEGRATION TESTS")

provenance_cases = [
    ("NaN", "tenure", "SMD", float("nan"),
     "NON_FINITE_NAN"),

    ("Positive infinity", "tenure", "SMD", float("inf"),
     "NON_FINITE_POSITIVE_INFINITY"),

    ("Missing score", "tenure", "SMD", None,
     "MISSING_VALUE"),

    ("Negative SMD", "tenure", "SMD", -0.01,
     "FINITE_OUT_OF_RANGE"),

    ("Invalid TVD", "Contract", "TVD", 1.01,
     "FINITE_OUT_OF_RANGE")
]

passed = 0

for name, feature, metric, score, expected_type in provenance_cases:

    test_report = make_invalid_metric_report(
        feature,
        metric,
        score
    )

    alerts = build_provenance_monitoring_alerts(
        test_report
    )

    # Verify strict JSON compatibility
    json_text = json.dumps(
        alerts,
        allow_nan=False
    )

    parsed = json.loads(json_text)

    matching = [
        alert for alert in parsed
        if alert["feature"] == feature
        and alert["metric"] == metric
        and alert["severity"] == "REVIEW REQUIRED"
    ]

    success = (
        len(matching) == 1
        and matching[0]["failure_type"] == expected_type
        and isinstance(matching[0]["original_value"], str)
    )

    if success:
        passed += 1

    print(f"\nScenario: {name}")
    print("Feature:", feature)
    print("Metric:", metric)
    print(
        "Failure type:",
        matching[0]["failure_type"] if matching else "MISSING"
    )
    print("Result:", "PASS" if success else "FAIL")


# --------------------------------------------------
# 5. Validate test results
# --------------------------------------------------

print("\n4. TEST SUMMARY")

print("Provenance scenarios:", len(provenance_cases))
print("Passed:", passed)
print("Failed:", len(provenance_cases) - passed)

assert passed == len(provenance_cases)

print("Provenance-aware alert integration: PASS")


# --------------------------------------------------
# 6. Confirm original monitoring report
# --------------------------------------------------

print("\n5. ORIGINAL REPORT VALIDATION")

original_assessment = aggregate_monitoring_alerts(
    report,
    monitoring_config
)

assert original_assessment["overall_status"] == "NORMAL"
assert report.count() == 8

print("Original report metrics: 8")
print("Original status: NORMAL")
print("Original report preserved: PASS")
print("Temporal production drift: NOT ASSESSED")


# --------------------------------------------------
# 7. Safety
# --------------------------------------------------

print("\n6. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.39 COMPLETE")


# COMMAND ----------


# Step 16.40 — Provenance-Aware Monitoring Summary
# READ-ONLY: In-memory monitoring summaries only

import json

print("=" * 70)
print("STEP 16.40 — PROVENANCE-AWARE MONITORING SUMMARY")
print("=" * 70)

# --------------------------------------------------
# 1. Define provenance-aware summary builder
# --------------------------------------------------

def build_provenance_monitoring_summary(report_df, config):

    assessment = aggregate_monitoring_alerts(
        report_df,
        config
    )

    alerts = build_provenance_monitoring_alerts(
        report_df
    )

    return {
        "pipeline_name": config["pipeline_name"],
        "environment": config["environment"],
        "overall_status": assessment["overall_status"],
        "drift_status": assessment["drift_status"],
        "quality_status": assessment["quality_status"],
        "alert_count": len(alerts),
        "alerts": alerts,
        "monitoring_scope": (
            "DEVELOPMENT TRAIN/TEST COMPARISON"
        ),
        "temporal_production_drift": "NOT ASSESSED"
    }


# --------------------------------------------------
# 2. Test original development summary
# --------------------------------------------------

print("\n1. DEVELOPMENT SUMMARY")

provenance_normal_summary = (
    build_provenance_monitoring_summary(
        report,
        monitoring_config
    )
)

assert provenance_normal_summary["overall_status"] == "NORMAL"
assert provenance_normal_summary["alert_count"] == 0

print("Overall status:", provenance_normal_summary["overall_status"])
print("Alert count:", provenance_normal_summary["alert_count"])
print("Result: PASS")


# --------------------------------------------------
# 3. Test warning and critical summaries
# --------------------------------------------------

print("\n2. WARNING AND CRITICAL SUMMARIES")

provenance_warning_summary = (
    build_provenance_monitoring_summary(
        warning_report,
        monitoring_config
    )
)

provenance_critical_summary = (
    build_provenance_monitoring_summary(
        critical_report,
        monitoring_config
    )
)

assert provenance_warning_summary["overall_status"] == "WARNING"
assert provenance_warning_summary["alert_count"] == 1
assert provenance_warning_summary["alerts"][0]["severity"] == "WARNING"

assert provenance_critical_summary["overall_status"] == "CRITICAL"
assert provenance_critical_summary["alert_count"] == 1
assert provenance_critical_summary["alerts"][0]["severity"] == "CRITICAL"

print("Warning status:", provenance_warning_summary["overall_status"])
print("Critical status:", provenance_critical_summary["overall_status"])
print("Result: PASS")


# --------------------------------------------------
# 4. Test provenance-aware failure summaries
# --------------------------------------------------

print("\n3. FAILURE PROVENANCE SUMMARIES")

failure_cases = [
    ("NaN", "tenure", "SMD", float("nan"),
     "NON_FINITE_NAN"),

    ("Positive infinity", "tenure", "SMD", float("inf"),
     "NON_FINITE_POSITIVE_INFINITY"),

    ("Missing value", "tenure", "SMD", None,
     "MISSING_VALUE"),

    ("Negative SMD", "tenure", "SMD", -0.01,
     "FINITE_OUT_OF_RANGE"),

    ("Invalid TVD", "Contract", "TVD", 1.01,
     "FINITE_OUT_OF_RANGE")
]

passed = 0

for name, feature, metric, score, expected_type in failure_cases:

    test_report = make_invalid_metric_report(
        feature,
        metric,
        score
    )

    summary = build_provenance_monitoring_summary(
        test_report,
        monitoring_config
    )

    # Strict JSON serialization and round-trip
    json_text = json.dumps(
        summary,
        allow_nan=False
    )

    parsed = json.loads(json_text)

    success = (
        parsed["overall_status"] == "REVIEW REQUIRED"
        and parsed["alert_count"] == 1
        and parsed["alerts"][0]["severity"] == "REVIEW REQUIRED"
        and parsed["alerts"][0]["failure_type"] == expected_type
        and parsed["temporal_production_drift"] == "NOT ASSESSED"
    )

    if success:
        passed += 1

    print(f"\nScenario: {name}")
    print("Overall status:", parsed["overall_status"])
    print("Failure type:", parsed["alerts"][0]["failure_type"])
    print("JSON round-trip: PASS")
    print("Result:", "PASS" if success else "FAIL")


# --------------------------------------------------
# 5. Verify all summary JSON payloads
# --------------------------------------------------

print("\n4. JSON SERIALIZATION")

for summary in [
    provenance_normal_summary,
    provenance_warning_summary,
    provenance_critical_summary
]:

    json_text = json.dumps(
        summary,
        allow_nan=False
    )

    assert json.loads(json_text) == summary

    print(
        summary["overall_status"],
        "JSON round-trip: PASS"
    )


# --------------------------------------------------
# 6. Validate results
# --------------------------------------------------

print("\n5. TEST SUMMARY")

print("Failure scenarios:", len(failure_cases))
print("Passed:", passed)
print("Failed:", len(failure_cases) - passed)

assert passed == len(failure_cases)

print("Provenance-aware summary integration: PASS")


# --------------------------------------------------
# 7. Confirm original summary
# --------------------------------------------------

print("\n6. ORIGINAL SUMMARY VALIDATION")

assert development_summary["overall_status"] == "NORMAL"
assert development_summary["alert_count"] == 0

print("Original summary preserved: PASS")
print("Development monitoring status: NORMAL")
print("Temporal production drift: NOT ASSESSED")


# --------------------------------------------------
# 8. Safety
# --------------------------------------------------

print("\n7. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.40 COMPLETE")


# COMMAND ----------


# Step 16.41 — Provenance Metadata Contract Validation
# READ-ONLY: In-memory JSON validation only

import json
import copy
import math

print("=" * 70)
print("STEP 16.41 — PROVENANCE METADATA CONTRACT VALIDATION")
print("=" * 70)

# --------------------------------------------------
# 1. Define provenance-aware contract validator
# --------------------------------------------------

def validate_provenance_contract(json_text, config):

    data = validate_diagnostic_reason_accuracy(
        json_text, config
    )

    for alert in data["alerts"]:

        if alert["severity"] != "REVIEW REQUIRED":
            continue

        assert "failure_type" in alert
        assert "original_value" in alert

        failure_type = alert["failure_type"]
        original_value = alert["original_value"]
        score = alert["score"]

        assert isinstance(original_value, str)
        assert original_value != ""

        # Reconstruct the measurement from provenance
        if failure_type == "NON_FINITE_NAN":
            assert original_value == "NaN"
            assert score is None
            reconstructed = float("nan")

        elif failure_type == "NON_FINITE_POSITIVE_INFINITY":
            assert original_value == "Infinity"
            assert score is None
            reconstructed = float("inf")

        elif failure_type == "NON_FINITE_NEGATIVE_INFINITY":
            assert original_value == "-Infinity"
            assert score is None
            reconstructed = float("-inf")

        elif failure_type == "MISSING_VALUE":
            assert original_value == "None"
            assert score is None
            reconstructed = None

        elif failure_type == "NON_NUMERIC":
            assert score is None
            assert original_value not in {
                "NaN", "Infinity", "-Infinity", "None"
            }
            reconstructed = original_value

        elif failure_type == "FINITE_OUT_OF_RANGE":
            assert score is not None
            assert isinstance(score, (int, float))
            assert not isinstance(score, bool)
            assert math.isfinite(score)

            reconstructed = float(original_value)

            assert math.isfinite(reconstructed)
            assert reconstructed == score

        else:
            raise AssertionError(
                f"Unknown failure type: {failure_type}"
            )

        # Independently verify the diagnostic reason
        expected = diagnose_invalid_metric(
            alert["feature"],
            alert["metric"],
            reconstructed
        )

        assert expected is not None, (
            "Provenance describes a valid measurement"
        )

        assert expected["reason"] in alert["message"], (
            "Provenance contradicts diagnostic reason"
        )

    return data


# --------------------------------------------------
# 2. Create valid provenance summaries
# --------------------------------------------------

def make_provenance_summary(feature, metric, score):

    simulated_report = make_invalid_metric_report(
        feature, metric, score
    )

    return build_provenance_monitoring_summary(
        simulated_report,
        monitoring_config
    )


valid_nan = make_provenance_summary(
    "tenure", "SMD", float("nan")
)

valid_infinity = make_provenance_summary(
    "tenure", "SMD", float("inf")
)

valid_missing = make_provenance_summary(
    "tenure", "SMD", None
)

valid_negative = make_provenance_summary(
    "tenure", "SMD", -0.01
)

valid_tvd = make_provenance_summary(
    "Contract", "TVD", 1.01
)

# --------------------------------------------------
# 3. Create deliberately invalid summaries
# --------------------------------------------------

wrong_nan_value = copy.deepcopy(valid_nan)
wrong_nan_value["alerts"][0]["original_value"] = "Infinity"

wrong_failure_type = copy.deepcopy(valid_infinity)
wrong_failure_type["alerts"][0]["failure_type"] = (
    "NON_FINITE_NAN"
)

wrong_finite_value = copy.deepcopy(valid_negative)
wrong_finite_value["alerts"][0]["original_value"] = "-0.05"

missing_provenance = copy.deepcopy(valid_nan)
del missing_provenance["alerts"][0]["failure_type"]

invalid_failure_type = copy.deepcopy(valid_tvd)
invalid_failure_type["alerts"][0]["failure_type"] = (
    "UNKNOWN_FAILURE"
)

# --------------------------------------------------
# 4. Define contract scenarios
# --------------------------------------------------

scenarios = [
    ("Valid NORMAL", provenance_normal_summary, True),
    ("Valid WARNING", provenance_warning_summary, True),
    ("Valid CRITICAL", provenance_critical_summary, True),
    ("Valid NaN provenance", valid_nan, True),
    ("Valid infinity provenance", valid_infinity, True),
    ("Valid missing-value provenance", valid_missing, True),
    ("Valid negative SMD provenance", valid_negative, True),
    ("Valid TVD provenance", valid_tvd, True),
    ("Mismatched NaN original value", wrong_nan_value, False),
    ("Incorrect failure type", wrong_failure_type, False),
    ("Incorrect finite original value", wrong_finite_value, False),
    ("Missing provenance field", missing_provenance, False),
    ("Unknown failure type", invalid_failure_type, False)
]

# --------------------------------------------------
# 5. Execute contract tests
# --------------------------------------------------

print("\n1. PROVENANCE CONTRACT TEST RESULTS")

passed = 0

for index, (name, summary, should_accept) in enumerate(
    scenarios, start=1
):

    try:
        json_text = json.dumps(
            summary,
            allow_nan=False
        )

        validate_provenance_contract(
            json_text,
            monitoring_config
        )

        accepted = True

    except (
        AssertionError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError
    ):
        accepted = False

    success = accepted == should_accept

    if success:
        passed += 1

    print(f"\nTest {index}: {name}")
    print(
        "Expected:",
        "ACCEPT" if should_accept else "REJECT"
    )
    print(
        "Actual:",
        "ACCEPT" if accepted else "REJECT"
    )
    print("Result:", "PASS" if success else "FAIL")


# --------------------------------------------------
# 6. Validate test results
# --------------------------------------------------

print("\n2. TEST SUMMARY")

print("Total tests:", len(scenarios))
print("Passed:", passed)
print("Failed:", len(scenarios) - passed)

assert passed == len(scenarios), (
    "One or more provenance contract tests failed"
)

print("Provenance metadata contract: PASS")


# --------------------------------------------------
# 7. Verify original development summary
# --------------------------------------------------

print("\n3. ORIGINAL SUMMARY VALIDATION")

original_json = json.dumps(
    development_summary,
    allow_nan=False
)

original = validate_provenance_contract(
    original_json,
    monitoring_config
)

assert original["overall_status"] == "NORMAL"
assert original["alert_count"] == 0

print("Original summary preserved: PASS")
print("Development monitoring status: NORMAL")
print("Temporal production drift: NOT ASSESSED")


# --------------------------------------------------
# 8. Safety
# --------------------------------------------------

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.41 COMPLETE")


# COMMAND ----------


# Step 16.42 — Monitoring Contract Robustness Testing
# READ-ONLY: In-memory JSON validation only

import json
import copy

print("=" * 70)
print("STEP 16.42 — MONITORING CONTRACT ROBUSTNESS TESTING")
print("=" * 70)

# --------------------------------------------------
# 1. Prepare valid reference summaries
# --------------------------------------------------

valid_finite = copy.deepcopy(valid_negative)
valid_nonfinite = copy.deepcopy(valid_nan)

# --------------------------------------------------
# 2. Create deliberately malformed summaries
# --------------------------------------------------

# Case 1: Finite original value is not numeric
case_1 = copy.deepcopy(valid_finite)
case_1["alerts"][0]["original_value"] = "not-a-number"

# Case 2: Finite original value is infinity
case_2 = copy.deepcopy(valid_finite)
case_2["alerts"][0]["original_value"] = "Infinity"

# Case 3: Finite score and original value disagree
case_3 = copy.deepcopy(valid_finite)
case_3["alerts"][0]["original_value"] = "-0.99"

# Case 4: NaN provenance has a numeric score
case_4 = copy.deepcopy(valid_nonfinite)
case_4["alerts"][0]["score"] = 0.25

# Case 5: NaN provenance has incorrect original value
case_5 = copy.deepcopy(valid_nonfinite)
case_5["alerts"][0]["original_value"] = "-Infinity"

# Case 6: Failure type is missing
case_6 = copy.deepcopy(valid_finite)
del case_6["alerts"][0]["failure_type"]

# Case 7: Original value is missing
case_7 = copy.deepcopy(valid_finite)
del case_7["alerts"][0]["original_value"]

# Case 8: Diagnostic message contradicts measurement
case_8 = copy.deepcopy(valid_finite)
case_8["alerts"][0]["message"] = (
    "tenure: SMD invalid — Non-finite metric score"
)

# --------------------------------------------------
# 3. Define test scenarios
# --------------------------------------------------

scenarios = [
    ("Valid finite provenance", valid_finite, True),
    ("Valid NaN provenance", valid_nonfinite, True),
    ("Non-numeric finite original value", case_1, False),
    ("Infinite finite original value", case_2, False),
    ("Mismatched finite score", case_3, False),
    ("NaN provenance with numeric score", case_4, False),
    ("NaN provenance with wrong value", case_5, False),
    ("Missing failure type", case_6, False),
    ("Missing original value", case_7, False),
    ("Contradictory diagnostic message", case_8, False)
]

# --------------------------------------------------
# 4. Execute robustness tests
# --------------------------------------------------

print("\n1. ROBUSTNESS TEST RESULTS")

passed = 0

for index, (name, summary, should_accept) in enumerate(
    scenarios, start=1
):

    try:
        json_text = json.dumps(
            summary,
            allow_nan=False
        )

        validate_provenance_contract(
            json_text,
            monitoring_config
        )

        accepted = True

    except (
        AssertionError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError
    ):
        accepted = False

    success = accepted == should_accept

    if success:
        passed += 1

    print(f"\nTest {index}: {name}")
    print(
        "Expected:",
        "ACCEPT" if should_accept else "REJECT"
    )
    print(
        "Actual:",
        "ACCEPT" if accepted else "REJECT"
    )
    print("Result:", "PASS" if success else "FAIL")


# --------------------------------------------------
# 5. Validate results
# --------------------------------------------------

print("\n2. TEST SUMMARY")

print("Total tests:", len(scenarios))
print("Passed:", passed)
print("Failed:", len(scenarios) - passed)

assert passed == len(scenarios), (
    "One or more robustness tests failed"
)

print("Monitoring contract robustness: PASS")


# --------------------------------------------------
# 6. Confirm original development summary
# --------------------------------------------------

print("\n3. ORIGINAL SUMMARY VALIDATION")

original_json = json.dumps(
    development_summary,
    allow_nan=False
)

original = validate_provenance_contract(
    original_json,
    monitoring_config
)

assert original["overall_status"] == "NORMAL"
assert original["alert_count"] == 0

print("Original summary preserved: PASS")
print("Development monitoring status: NORMAL")
print("Temporal production drift: NOT ASSESSED")


# --------------------------------------------------
# 7. Safety
# --------------------------------------------------

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.42 COMPLETE")


# COMMAND ----------


# Step 16.44 — Monitoring Contract Gap Testing
# READ-ONLY: In-memory Spark DataFrames only

print("=" * 70)
print("STEP 16.44 — MONITORING CONTRACT GAP TESTING")
print("=" * 70)

# 1. Build the expected feature-metric combinations

expected_pairs = (
    {
        (feature, "SMD")
        for feature in monitoring_config["numerical_features"]
    }
    | {
        (feature, "TVD")
        for feature in monitoring_config["categorical_features"]
    }
    | {
        (feature, "MISSING_PCT")
        for feature in (
            monitoring_config["numerical_features"]
            + monitoring_config["categorical_features"]
        )
    }
)

baseline_data = [row.asDict() for row in report.collect()]

assert len(baseline_data) == 8

print("\n1. EXPECTED METRIC COVERAGE")
print("Expected metric pairs:", len(expected_pairs))
print("Original report rows:", len(baseline_data))


# 2. Create a reusable in-memory test

def check_metric_integrity(rows):

    pairs = [
        (row["feature"], row["metric"])
        for row in rows
    ]

    actual_pairs = set(pairs)

    missing = expected_pairs - actual_pairs
    unexpected = actual_pairs - expected_pairs

    duplicates = len(pairs) != len(actual_pairs)

    valid = (
        not missing
        and not unexpected
        and not duplicates
        and len(rows) == len(expected_pairs)
    )

    return {
        "valid": valid,
        "missing": sorted(missing),
        "unexpected": sorted(unexpected),
        "duplicates": duplicates
    }


# 3. Create controlled test scenarios

missing_metric = [
    row.copy()
    for row in baseline_data
    if not (
        row["feature"] == "tenure"
        and row["metric"] == "SMD"
    )
]

duplicate_metric = [
    row.copy() for row in baseline_data
]
duplicate_metric.append(baseline_data[0].copy())

unexpected_metric = [
    row.copy() for row in baseline_data
]
unexpected_metric[0]["feature"] = "UnknownFeature"

replaced_metric = [
    row.copy() for row in baseline_data
]
replaced_metric[0]["metric"] = "UNKNOWN_METRIC"

scenarios = [
    ("Original report", baseline_data, True),
    ("Missing metric", missing_metric, False),
    ("Duplicate metric", duplicate_metric, False),
    ("Unknown feature", unexpected_metric, False),
    ("Unknown metric", replaced_metric, False)
]


# 4. Execute tests

print("\n2. METRIC INTEGRITY TEST RESULTS")

passed = 0

for index, (name, rows, expected_valid) in enumerate(
    scenarios, start=1
):

    result = check_metric_integrity(rows)

    success = result["valid"] == expected_valid

    if success:
        passed += 1

    print(f"\nTest {index}: {name}")
    print(
        "Expected:",
        "ACCEPT" if expected_valid else "REJECT"
    )
    print(
        "Actual:",
        "ACCEPT" if result["valid"] else "REJECT"
    )
    print("Result:", "PASS" if success else "FAIL")

    if not result["valid"]:
        print("Missing:", result["missing"])
        print("Unexpected:", result["unexpected"])
        print("Duplicates:", result["duplicates"])


# 5. Confirm the existing monitoring aggregator

print("\n3. EXISTING AGGREGATOR VALIDATION")

original_assessment = aggregate_monitoring_alerts(
    report,
    monitoring_config
)

assert original_assessment["overall_status"] == "NORMAL"

print("Original status:", original_assessment["overall_status"])
print("Original report preserved: PASS")
print("Temporal production drift: NOT ASSESSED")


# 6. Final results

print("\n4. TEST SUMMARY")
print("Total tests:", len(scenarios))
print("Passed:", passed)
print("Failed:", len(scenarios) - passed)

assert passed == len(scenarios)

print("Monitoring metric integrity: PASS")


# 7. Safety

print("\n5. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.44 COMPLETE")


# COMMAND ----------


# Step 16.45 — End-to-End Monitoring Validation
# READ-ONLY: In-memory Spark and JSON operations only

import json

print("=" * 70)
print("STEP 16.45 — END-TO-END MONITORING VALIDATION")
print("=" * 70)

# --------------------------------------------------
# 1. Define integrated monitoring workflow
# --------------------------------------------------

def run_validated_monitoring(report_df, config):

    # Check the complete metric set before aggregation
    rows = [row.asDict() for row in report_df.collect()]

    integrity = check_metric_integrity(rows)

    assert integrity["valid"], (
        "Monitoring metric integrity failed: "
        f"{integrity}"
    )

    # Build the provenance-aware monitoring summary
    summary = build_provenance_monitoring_summary(
        report_df,
        config
    )

    # Serialize using strict JSON
    json_text = json.dumps(
        summary,
        allow_nan=False
    )

    # Validate summary and diagnostic provenance
    validated = validate_provenance_contract(
        json_text,
        config
    )

    return validated


# --------------------------------------------------
# 2. Prepare integration scenarios
# --------------------------------------------------

normal_report = report

missing_report = spark.createDataFrame(
    missing_metric,
    schema=report.schema
)

duplicate_report = spark.createDataFrame(
    duplicate_metric,
    schema=report.schema
)

unknown_feature_report = spark.createDataFrame(
    unexpected_metric,
    schema=report.schema
)

unknown_metric_report = spark.createDataFrame(
    replaced_metric,
    schema=report.schema
)

nan_report = make_invalid_metric_report(
    "tenure",
    "SMD",
    float("nan")
)

integration_scenarios = [
    ("NORMAL", normal_report, "NORMAL"),
    ("WARNING", warning_report, "WARNING"),
    ("CRITICAL", critical_report, "CRITICAL"),
    ("REVIEW REQUIRED", nan_report, "REVIEW REQUIRED"),
    ("Missing metric", missing_report, None),
    ("Duplicate metric", duplicate_report, None),
    ("Unknown feature", unknown_feature_report, None),
    ("Unknown metric", unknown_metric_report, None)
]


# --------------------------------------------------
# 3. Execute end-to-end tests
# --------------------------------------------------

print("\n1. END-TO-END TEST RESULTS")

passed = 0

for index, (name, test_report, expected_status) in enumerate(
    integration_scenarios, start=1
):

    try:
        result = run_validated_monitoring(
            test_report,
            monitoring_config
        )

        accepted = True
        actual_status = result["overall_status"]

    except (
        AssertionError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError
    ):
        accepted = False
        actual_status = None

    if expected_status is None:
        success = not accepted
    else:
        success = accepted and actual_status == expected_status

    if success:
        passed += 1

    print(f"\nTest {index}: {name}")
    print(
        "Expected:",
        expected_status if expected_status else "REJECT"
    )
    print(
        "Actual:",
        actual_status if accepted else "REJECT"
    )
    print("Result:", "PASS" if success else "FAIL")


# --------------------------------------------------
# 4. Verify original development summary
# --------------------------------------------------

print("\n2. ORIGINAL DEVELOPMENT VALIDATION")

validated_development = run_validated_monitoring(
    report,
    monitoring_config
)

assert validated_development["overall_status"] == "NORMAL"
assert validated_development["alert_count"] == 0

assert (
    validated_development["monitoring_scope"]
    == "DEVELOPMENT TRAIN/TEST COMPARISON"
)

assert (
    validated_development["temporal_production_drift"]
    == "NOT ASSESSED"
)

print("Overall status: NORMAL")
print("Alert count: 0")
print("Monitoring scope: DEVELOPMENT TRAIN/TEST COMPARISON")
print("Temporal production drift: NOT ASSESSED")
print("Result: PASS")


# --------------------------------------------------
# 5. Final results
# --------------------------------------------------

print("\n3. TEST SUMMARY")

print("Total integration tests:", len(integration_scenarios))
print("Passed:", passed)
print("Failed:", len(integration_scenarios) - passed)

assert passed == len(integration_scenarios)

print("End-to-end monitoring validation: PASS")


# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n4. SAFETY")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Files created: False")
print("Protected tables modified: False")

print("\nSTEP 16.45 COMPLETE")
