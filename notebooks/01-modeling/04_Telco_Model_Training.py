# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Telco Model Training
# MAGIC
# MAGIC **Goal:** Load the saved training split and fitted preprocessing pipeline, create a numerical churn label, train a Logistic Regression classifier, and save the trained model.
# MAGIC
# MAGIC Run cells from top to bottom. **Logistic Regression** predicts the probability of one of two outcomes: churn (`1`) or no churn (`0`). We do not use the test set to train the model.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read the saved training data
# MAGIC
# MAGIC Notebook 03 saved the **raw training split** as a Unity Catalog Delta table. `spark.table()` reads that table into a DataFrame. The training data contains 5,634 customers; it has not yet been converted into final ML features.
# MAGIC
# MAGIC **Checkpoint:** 5,634 rows.
# MAGIC

# COMMAND ----------

train_df = spark.table("workspace.default.telco_train_raw")
print("Training rows:", train_df.count())
display(train_df.select("customerID", "Churn", "tenure", "MonthlyCharges").limit(5))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load the fitted preprocessing pipeline
# MAGIC
# MAGIC The pipeline from Notebook 03 already learned the training-data median, category mappings, and scaling statistics. `PipelineModel.load()` retrieves those learned rules. `transform()` applies them without fitting again.
# MAGIC
# MAGIC **Checkpoint:** 20 pipeline stages and a `features` column with 60 positions.
# MAGIC

# COMMAND ----------

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

pipeline_path = "/Volumes/workspace/default/telco_data/telco_preprocessing_pipeline"
loaded_pipeline = PipelineModel.load(pipeline_path)
train_prepared = loaded_pipeline.transform(train_df)

print("Pipeline stages:", len(loaded_pipeline.stages))
print("Training rows:", train_prepared.count())
print("Feature positions:", train_prepared.select(F.size(vector_to_array("features")).alias("n")).first()["n"])
display(train_prepared.select("customerID", "Churn", "features").limit(5))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create the target label
# MAGIC
# MAGIC **Features** are the customer characteristics the model uses. The **label** is the answer we want the model to learn. Spark Logistic Regression expects a numeric label: `0.0` for `Churn = No`, and `1.0` for `Churn = Yes`.
# MAGIC
# MAGIC We explicitly map both values rather than letting an automatic indexer decide their order. Unexpected values become null so we can detect them before training.
# MAGIC
# MAGIC **Checkpoint:** Zero invalid labels.
# MAGIC

# COMMAND ----------

train_labeled = train_prepared.withColumn(
    "label",
    F.when(F.col("Churn") == "No", 0.0)
     .when(F.col("Churn") == "Yes", 1.0)
     .otherwise(None)
     .cast("double")
)

invalid_labels = train_labeled.filter(F.col("label").isNull()).count()
print("Invalid labels:", invalid_labels)
if invalid_labels != 0:
    raise ValueError("Unexpected or missing churn labels found")
display(train_labeled.groupBy("Churn", "label").count().orderBy("label"))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train a Logistic Regression model
# MAGIC
# MAGIC **Training** means letting an algorithm learn a relationship between the `features` vector and the `label`. Logistic Regression estimates a churn probability between 0 and 1. A default threshold of 0.5 converts that probability into a predicted class.
# MAGIC
# MAGIC `fit()` learns model coefficients from **training data only**. `maxIter=100` allows up to 100 optimization iterations.
# MAGIC
# MAGIC **Checkpoint:** The model fits without an error.
# MAGIC

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0
)

lr_model = lr.fit(train_labeled)
print("Logistic Regression model trained successfully")
print("Number of coefficients:", len(lr_model.coefficients))
print("Intercept:", lr_model.intercept)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Preview predictions on training data
# MAGIC
# MAGIC `transform()` now uses the trained classifier to make predictions. The `probability` vector contains probabilities for class 0 (stay) and class 1 (churn). `prediction` is the chosen class.
# MAGIC
# MAGIC These are **training-set predictions**, useful only as a quick sanity check. They do **not** tell us how well the model generalizes. Notebook 05 will evaluate the held-out test set.
# MAGIC

# COMMAND ----------

train_predictions = lr_model.transform(train_labeled)
display(train_predictions.select(
    "customerID", "Churn", "label", "probability", "prediction"
).limit(10))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save the trained classifier
# MAGIC
# MAGIC Saving the model lets Notebook 05 load it without retraining. We store it in the same Unity Catalog volume as the preprocessing pipeline. The saved model expects the 60-position `features` vector produced by that pipeline.
# MAGIC
# MAGIC **Caution:** `overwrite()` replaces the previous practice model at this path.
# MAGIC
# MAGIC **Checkpoint:** The model saves and reloads successfully.
# MAGIC

# COMMAND ----------

from pyspark.ml.classification import LogisticRegressionModel

model_path = "/Volumes/workspace/default/telco_data/telco_logistic_regression_model"
lr_model.write().overwrite().save(model_path)

reloaded_model = LogisticRegressionModel.load(model_path)
print("Saved model:", model_path)
print("Reloaded coefficient count:", len(reloaded_model.coefficients))


# COMMAND ----------

# MAGIC %md
# MAGIC ## What comes next?
# MAGIC
# MAGIC Notebook 05 will load the **saved raw test split**, apply the **same fitted preprocessing pipeline**, create the numeric label, and use this **saved classifier** to evaluate predictions. We will calculate a confusion matrix, precision, recall, F1, and ROC-AUC. We will add MLflow experiment tracking as a separate learning step so the first training run stays easy to understand.
# MAGIC

# COMMAND ----------


# Load the raw testing dataset saved in Notebook 03.
test_df = spark.table(
    "workspace.default.telco_test_raw"
)

# Verify the number of testing records.
print("Testing records:", test_df.count())

# Display five customer records.
display(
    test_df.select(
        "customerID",
        "Churn",
        "tenure",
        "MonthlyCharges"
    ).limit(5)
)
