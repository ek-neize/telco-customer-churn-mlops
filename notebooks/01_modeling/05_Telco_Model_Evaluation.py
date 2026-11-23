# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Telco Model Evaluation
# MAGIC
# MAGIC **Goal:** Evaluate how accurately our trained Logistic Regression model predicts customer churn using the 1,409 testing records that were not used for model training.
# MAGIC
# MAGIC Run cells from top to bottom. Each numbered explanation is followed by its Python code cell.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the testing dataset
# MAGIC
# MAGIC Notebook 03 saved the raw testing split as a Unity Catalog Delta table. The **testing dataset** contains customers held out during training. `spark.table()` reads it into a DataFrame.
# MAGIC
# MAGIC **Checkpoint:** 1,409 rows.

# COMMAND ----------

# Load the testing dataset saved in Notebook 03.
test_df = spark.table("workspace.default.telco_test_raw")
print("Testing records:", test_df.count())
display(test_df.select("customerID", "Churn", "tenure", "MonthlyCharges").limit(5))

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ## 2. Load the saved pipeline and model
# MAGIC
# MAGIC Notebook 03 saved a fitted **preprocessing pipeline** that converts raw customer fields into 60 numerical feature positions. Notebook 04 saved the trained **Logistic Regression model**. We need both: the pipeline prepares data, and the classifier predicts churn.
# MAGIC
# MAGIC **Checkpoint:** 20 pipeline stages and 60 coefficients.

# COMMAND ----------

from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql import functions as F

pipeline_path = "/Volumes/workspace/default/telco_data/telco_preprocessing_pipeline"
model_path = "/Volumes/workspace/default/telco_data/telco_logistic_regression_model"

loaded_pipeline = PipelineModel.load(pipeline_path)
loaded_model = LogisticRegressionModel.load(model_path)
print("Pipeline stages:", len(loaded_pipeline.stages))
print("Model coefficients:", len(loaded_model.coefficients))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Prepare the testing features
# MAGIC
# MAGIC The saved pipeline transforms raw customer information into the `features` vector expected by the classifier. We use **`transform()`**, not `fit()`, because fitting on test data would leak information from the held-out set into preprocessing.
# MAGIC
# MAGIC **Checkpoint:** 1,409 prepared records.

# COMMAND ----------

# Apply training-fitted transformations without refitting.
test_prepared = loaded_pipeline.transform(test_df)
print("Prepared testing records:", test_prepared.count())
display(test_prepared.select("customerID", "Churn", "features").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create the numerical target label
# MAGIC
# MAGIC The original `Churn` column contains `No` and `Yes`. A **label** is the known correct outcome used to judge a prediction. Use the same mapping as training: **No → 0.0**, **Yes → 1.0**. Unexpected values become null so they are visible rather than silently classified.
# MAGIC
# MAGIC **Checkpoint:** Zero invalid or missing labels.
# MAGIC

# COMMAND ----------

test_labeled = test_prepared.withColumn(
    "label",
    F.when(F.col("Churn") == "Yes", 1.0)
     .when(F.col("Churn") == "No", 0.0)
     .otherwise(None)
     .cast("double")
)
invalid_labels = test_labeled.filter(F.col("label").isNull()).count()
print("Invalid or missing labels:", invalid_labels)
display(test_labeled.select("customerID", "Churn", "label").limit(10))
assert invalid_labels == 0, "Unexpected Churn values found"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate churn predictions
# MAGIC
# MAGIC `loaded_model.transform()` adds predictions for every customer. **`prediction`** is the predicted class (0 or 1), **`probability`** contains estimated class probabilities, and **`rawPrediction`** contains model scores. A probability is an estimate, not a guarantee.
# MAGIC
# MAGIC **Checkpoint:** 1,409 predictions.

# COMMAND ----------

predictions = loaded_model.transform(test_labeled)
print("Predictions generated:", predictions.count())
display(predictions.select("customerID", "Churn", "label", "prediction", "probability").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Calculate accuracy, precision, recall and F1-score
# MAGIC
# MAGIC **Accuracy** is the fraction of all correct predictions. **Precision (churn)** is the fraction of predicted churners who really churned. **Recall (churn)** is the fraction of actual churners identified. **F1-score** balances precision and recall. We calculate precision and recall specifically for positive class **1.0 (churn)**.
# MAGIC
# MAGIC **Checkpoint:** Each score is between 0 and 1.

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

accuracy = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
).evaluate(predictions)
precision = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="precisionByLabel", metricLabel=1.0
).evaluate(predictions)
recall = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="recallByLabel", metricLabel=1.0
).evaluate(predictions)
f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Build the confusion matrix
# MAGIC
# MAGIC A **confusion matrix** compares actual outcomes with predicted outcomes. **TN:** correctly predicted stay. **FP:** predicted churn but stayed. **FN:** predicted stay but churned. **TP:** correctly predicted churn. False negatives represent customers the retention team may miss.
# MAGIC
# MAGIC **Checkpoint:** All counts add up to 1,409.

# COMMAND ----------

confusion_matrix = (
    predictions.groupBy("label", "prediction")
    .agg(F.count("*").alias("Customers"))
    .orderBy("label", "prediction")
)
display(confusion_matrix)
print("Total evaluated customers:", confusion_matrix.agg(F.sum("Customers")).first()[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Calculate ROC-AUC
# MAGIC
# MAGIC **ROC-AUC** measures how well the model ranks churners above non-churners across classification thresholds. **0.5** is roughly random discrimination; **1.0** indicates perfect discrimination. This metric evaluates ranking, not necessarily the accuracy of a particular probability threshold.
# MAGIC
# MAGIC **Checkpoint:** ROC-AUC between 0 and 1.

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

auc_evaluator = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)
roc_auc = auc_evaluator.evaluate(predictions)
print(f"ROC-AUC: {roc_auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Create an evaluation summary
# MAGIC
# MAGIC We combine the five evaluation metrics into one Spark DataFrame. This makes it easier to compare models, report results, and save the evaluation.
# MAGIC
# MAGIC **Checkpoint:** Five metric rows.

# COMMAND ----------

evaluation_results = [
    ("Accuracy", float(accuracy)),
    ("Precision (Churn)", float(precision)),
    ("Recall (Churn)", float(recall)),
    ("F1-score (Churn)", float(f1)),
    ("ROC-AUC", float(roc_auc))
]
metrics_df = spark.createDataFrame(evaluation_results, ["Metric", "Score"])
display(metrics_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Save the evaluation results
# MAGIC
# MAGIC Save the evaluation summary to a Unity Catalog **Delta table** so you can inspect it later. `overwrite` replaces this practice table if it already exists; it does not overwrite your training or testing datasets.
# MAGIC
# MAGIC **Checkpoint:** Five saved metrics.

# COMMAND ----------

evaluation_table = "workspace.default.telco_model_evaluation"
(
    metrics_df.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true").saveAsTable(evaluation_table)
)
saved_metrics = spark.table(evaluation_table)
print("Saved evaluation table:", evaluation_table)
print("Number of metrics:", saved_metrics.count())
display(saved_metrics)
