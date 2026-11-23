# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Telco Model Optimization
# MAGIC
# MAGIC **Goal:** Improve customer-churn recall by tuning the Logistic Regression classification threshold using a validation set, then compare the selected model with the original model.
# MAGIC
# MAGIC Run cells from top to bottom. Each numbered Markdown explanation is followed by a Python code cell.
# MAGIC
# MAGIC **Starting point:** Accuracy 80.06%, churn precision 67.00%, churn recall 52.48%, F1-score 58.86%, ROC-AUC 83.85%.
# MAGIC
# MAGIC **Important:** We have already inspected the original test-set results. This notebook is an exploratory optimization exercise. For a truly unbiased final performance estimate after repeated experimentation, use fresh, untouched data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Import libraries
# MAGIC
# MAGIC PySpark MLlib supplies the saved preprocessing pipeline, Logistic Regression classifier, and evaluators. `vector_to_array` converts probability vectors into arrays so we can extract the probability of churn (class 1).
# MAGIC
# MAGIC **Checkpoint:** Imports complete.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.ml import PipelineModel, Pipeline
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

print("Libraries imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load the saved training and testing splits
# MAGIC
# MAGIC Notebook 03 saved the original 80/20 split. The **training** data will be divided again into a smaller training portion and a **validation** portion. The original test split is reserved for the final comparison.
# MAGIC
# MAGIC **Checkpoint:** 5,634 training rows and 1,409 test rows.

# COMMAND ----------

train_raw = spark.table("workspace.default.telco_train_raw")
test_raw = spark.table("workspace.default.telco_test_raw")
print("Original training rows:", train_raw.count())
print("Original testing rows:", test_raw.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create a training/validation split
# MAGIC
# MAGIC A **validation set** helps us choose settings without repeatedly adjusting them against the test set. Here we split the original training data into 80% development training and 20% validation. Because the split is random, the exact counts may vary.
# MAGIC
# MAGIC We use a fixed `seed` to make the split reproducible.
# MAGIC
# MAGIC **Checkpoint:** The two new counts sum to 5,634.

# COMMAND ----------

development_raw, validation_raw = train_raw.randomSplit([0.8, 0.2], seed=123)
development_raw = development_raw.cache()
validation_raw = validation_raw.cache()
print("Development training rows:", development_raw.count())
print("Validation rows:", validation_raw.count())
print("Combined rows:", development_raw.count() + validation_raw.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Refit preprocessing using development training data only
# MAGIC
# MAGIC The preprocessing pipeline in Notebook 03 learned medians, category mappings, and scaling statistics from all 5,634 original training rows. To avoid leaking validation information into preprocessing, we create **fresh, unfitted copies of the same 20 pipeline stages** and fit them using only the development training portion.
# MAGIC
# MAGIC We load the saved fitted pipeline to recover its stage configurations, then use each stage's `copy()` method for a fresh estimator. For any transformer stages, copying preserves their settings. This is safe here because our original pipeline's stages were estimators, and `PipelineModel.stages` are fitted models; therefore, **we must reconstruct the unfitted pipeline explicitly** rather than refit the saved fitted stages.
# MAGIC
# MAGIC **Checkpoint:** The new pipeline has 20 stages and transforms both datasets into `features` vectors.

# COMMAND ----------

from pyspark.ml.feature import (
    Imputer, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)

categorical_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

# The source CSV stores blank TotalCharges values as text.
# Clean the column BEFORE passing it into the Imputer.
def clean_total_charges(df):
    return df.withColumn(
        "TotalCharges",
        F.when(F.trim(F.col("TotalCharges").cast("string")) == "", None)
         .otherwise(F.col("TotalCharges")).cast("double")
    )

development_clean = clean_total_charges(development_raw)
validation_clean = clean_total_charges(validation_raw)

imputer = Imputer(inputCols=["TotalCharges"],
                  outputCols=["TotalCharges_imputed"], strategy="median")
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_index",
                          handleInvalid="keep") for c in categorical_cols]
encoder = OneHotEncoder(
    inputCols=[f"{c}_index" for c in categorical_cols],
    outputCols=[f"{c}_encoded" for c in categorical_cols],
    handleInvalid="keep"
)
numeric_assembler = VectorAssembler(
    inputCols=["tenure", "MonthlyCharges", "TotalCharges_imputed", "SeniorCitizen"],
    outputCol="numerical_features"
)
scaler = StandardScaler(inputCol="numerical_features",
                        outputCol="scaled_numerical_features",
                        withMean=True, withStd=True)
final_assembler = VectorAssembler(
    inputCols=["scaled_numerical_features"] +
              [f"{c}_encoded" for c in categorical_cols],
    outputCol="features"
)

fresh_pipeline = Pipeline(stages=[
    imputer, *indexers, encoder, numeric_assembler, scaler, final_assembler
])
validation_pipeline_model = fresh_pipeline.fit(development_clean)
development_prepared = validation_pipeline_model.transform(development_clean)
validation_prepared = validation_pipeline_model.transform(validation_clean)
print("Pipeline stages:", len(validation_pipeline_model.stages))
print("Development prepared:", development_prepared.count())
print("Validation prepared:", validation_prepared.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create numerical churn labels
# MAGIC
# MAGIC Logistic Regression requires a numerical target column. We map `No` to `0.0` and `Yes` to `1.0`, preserving the original text column for interpretation.
# MAGIC
# MAGIC **Checkpoint:** No null labels in either dataset.

# COMMAND ----------

def add_churn_label(df):
    return df.withColumn(
        "label",
        F.when(F.col("Churn") == "Yes", 1.0)
         .when(F.col("Churn") == "No", 0.0)
         .otherwise(None).cast("double")
    )

development_labeled = add_churn_label(development_prepared)
validation_labeled = add_churn_label(validation_prepared)

for name, df in [("Development", development_labeled),
                 ("Validation", validation_labeled)]:
    missing = df.filter(F.col("label").isNull()).count()
    print(name, "invalid labels:", missing)
    assert missing == 0, f"Unexpected labels in {name}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Train a Logistic Regression model on development data
# MAGIC
# MAGIC The model learns a coefficient for each feature. We train on development data only; validation data remains unseen until we measure how well the model predicts it.
# MAGIC
# MAGIC We start with standard Logistic Regression settings, using `maxIter=100` to allow enough optimization iterations.
# MAGIC
# MAGIC **Checkpoint:** A trained classifier and validation predictions.

# COMMAND ----------

candidate_lr = LogisticRegression(
    featuresCol="features", labelCol="label",
    maxIter=100, regParam=0.0
)
candidate_model = candidate_lr.fit(development_labeled)
validation_predictions = candidate_model.transform(validation_labeled)
print("Coefficients:", len(candidate_model.coefficients))
print("Validation predictions:", validation_predictions.count())
display(validation_predictions.select("Churn", "label", "prediction", "probability").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Extract churn probabilities
# MAGIC
# MAGIC The `probability` column stores a two-element vector: position 0 is the estimated probability of staying, and position 1 is the estimated probability of churning.
# MAGIC
# MAGIC We extract position 1 into a new `churn_probability` column. A **threshold** converts this probability into a yes/no prediction. The default threshold is 0.50.
# MAGIC
# MAGIC **Checkpoint:** Values between 0 and 1.

# COMMAND ----------

validation_scores = validation_predictions.withColumn(
    "churn_probability", vector_to_array("probability")[1]
)
display(validation_scores.select(
    "customerID", "label", "churn_probability", "prediction"
).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Compare thresholds on validation data
# MAGIC
# MAGIC Reducing the threshold typically catches more actual churners (**higher recall**) but also flags more customers who will stay (**lower precision**). We compare thresholds from 0.25 to 0.60.
# MAGIC
# MAGIC For each threshold we count true positives (TP), false positives (FP), and false negatives (FN), then calculate churn precision, recall, and F1. These are measured **on validation data only**.
# MAGIC
# MAGIC **Checkpoint:** A table showing the precision–recall trade-off.

# COMMAND ----------

thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# Evaluate all thresholds in one aggregation to avoid repeatedly scanning the data.
expressions = []
for i, threshold in enumerate(thresholds):
    predicted_positive = F.col("churn_probability") >= F.lit(threshold)
    actual_positive = F.col("label") == 1.0
    expressions.extend([
        F.sum(F.when(predicted_positive & actual_positive, 1).otherwise(0)).alias(f"tp_{i}"),
        F.sum(F.when(predicted_positive & ~actual_positive, 1).otherwise(0)).alias(f"fp_{i}"),
        F.sum(F.when(~predicted_positive & actual_positive, 1).otherwise(0)).alias(f"fn_{i}")
    ])
counts = validation_scores.agg(*expressions).first()
results = []
for i, threshold in enumerate(thresholds):
    tp, fp, fn = counts[f"tp_{i}"], counts[f"fp_{i}"], counts[f"fn_{i}"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    results.append((float(threshold), float(precision), float(recall), float(f1), int(tp), int(fp), int(fn)))

threshold_metrics = spark.createDataFrame(
    results, ["Threshold", "Precision", "Recall", "F1", "TP", "FP", "FN"]
)
display(threshold_metrics.orderBy("Threshold"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Choose a threshold using a stated business rule
# MAGIC
# MAGIC A retention team may prioritize identifying customers who are likely to leave. For this exercise, our rule is: **choose the threshold with the highest validation F1-score**, balancing precision and recall. This is an example rule, not necessarily the best business objective.
# MAGIC
# MAGIC We make the decision using validation data only. We will not change the threshold after looking at test results.
# MAGIC
# MAGIC **Checkpoint:** One selected threshold and its validation scores.

# COMMAND ----------

best_row = threshold_metrics.orderBy(
    F.desc("F1"), F.desc("Recall"), F.desc("Threshold")
).first()
selected_threshold = float(best_row["Threshold"])
print("Selected threshold:", selected_threshold)
print(f"Validation precision: {best_row['Precision']:.4f}")
print(f"Validation recall:    {best_row['Recall']:.4f}")
print(f"Validation F1-score:  {best_row['F1']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Refit the final classifier using the original training split
# MAGIC
# MAGIC After choosing a threshold, we return to the **original model saved in Notebook 04**, which was trained on all 5,634 training records. We reuse its matching saved preprocessing pipeline and evaluate the selected threshold on the held-out test split.
# MAGIC
# MAGIC This keeps the same feature mapping and model coefficients as the baseline. The only intended change is the classification threshold.
# MAGIC
# MAGIC **Checkpoint:** 1,409 test predictions.

# COMMAND ----------

saved_pipeline_path = (
    "/Volumes/workspace/default/telco_data/telco_preprocessing_pipeline"
)
saved_model_path = (
    "/Volumes/workspace/default/telco_data/telco_logistic_regression_model"
)
original_pipeline = PipelineModel.load(saved_pipeline_path)
original_model = LogisticRegressionModel.load(saved_model_path)

test_clean = clean_total_charges(test_raw)
test_prepared = original_pipeline.transform(test_clean)
test_labeled = add_churn_label(test_prepared)
assert test_labeled.filter(F.col("label").isNull()).count() == 0

baseline_predictions = original_model.transform(test_labeled)
final_predictions = baseline_predictions.withColumn(
    "churn_probability", vector_to_array("probability")[1]
).withColumn(
    "optimized_prediction",
    F.when(F.col("churn_probability") >= selected_threshold, 1.0).otherwise(0.0)
)
print("Testing predictions:", final_predictions.count())
display(final_predictions.select(
    "customerID", "label", "prediction", "optimized_prediction", "churn_probability"
).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Compare original and selected thresholds on the test data
# MAGIC
# MAGIC We now compare the original 0.50 threshold with the threshold selected using validation data. The model's probabilities do not change, so ROC-AUC should be the same for both thresholds. Accuracy, precision, recall, and F1 can change.
# MAGIC
# MAGIC **Caution:** Because earlier notebooks already reported test performance, these results are useful for learning but should not be presented as a fully independent final estimate.
# MAGIC
# MAGIC **Checkpoint:** Two rows of metrics, one for each threshold.

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def evaluate_prediction_column(df, prediction_col):
    evaluator = lambda metric, **kw: MulticlassClassificationEvaluator(
        labelCol="label", predictionCol=prediction_col,
        metricName=metric, **kw
    ).evaluate(df)
    accuracy = evaluator("accuracy")
    precision = evaluator("precisionByLabel", metricLabel=1.0)
    recall = evaluator("recallByLabel", metricLabel=1.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (float(accuracy), float(precision), float(recall), float(f1))

baseline_metrics = evaluate_prediction_column(final_predictions, "prediction")
optimized_metrics = evaluate_prediction_column(final_predictions, "optimized_prediction")

comparison = spark.createDataFrame([
    ("Original (0.50)", 0.50, *baseline_metrics),
    ("Validation-selected", selected_threshold, *optimized_metrics)
], ["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1"])

display(comparison)

auc = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
).evaluate(final_predictions)
print(f"ROC-AUC (same model, both thresholds): {auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Save optimization results
# MAGIC
# MAGIC We save the comparison table as a Delta table, along with the selected threshold. This creates a reproducible record of the experiment.
# MAGIC
# MAGIC The original trained model is **not overwritten**. Its saved parameters and default threshold remain unchanged.
# MAGIC
# MAGIC **Checkpoint:** Two saved comparison rows.

# COMMAND ----------

optimization_table = "workspace.default.telco_optimization_results"
(
    comparison.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(optimization_table)
)
print("Saved results:", optimization_table)
print("Rows:", spark.table(optimization_table).count())
print("Selected validation threshold:", selected_threshold)
display(spark.table(optimization_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Completion checklist
# MAGIC
# MAGIC - [ ] Loaded saved training and testing data
# MAGIC - [ ] Created a validation split from training data
# MAGIC - [ ] Refit preprocessing on development data only
# MAGIC - [ ] Trained a validation-stage classifier
# MAGIC - [ ] Compared churn thresholds on validation data
# MAGIC - [ ] Selected a threshold without consulting the test set
# MAGIC - [ ] Compared the baseline and selected thresholds on test data
# MAGIC - [ ] Saved the comparison table
# MAGIC
# MAGIC **Next learning topic:** Compare Logistic Regression with a tree-based classifier, using the same development/validation discipline.
