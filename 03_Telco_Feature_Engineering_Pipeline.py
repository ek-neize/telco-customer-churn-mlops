# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Reusable Feature Engineering Pipeline
# MAGIC **Goal:** Fit all preprocessing steps on training data only, then apply the same fitted transformations to held-out test data.
# MAGIC
# MAGIC A **pipeline** is a repeatable chain of transformations. Fitting on training data only avoids **data leakage**, where test information accidentally influences model preparation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load raw data and fix the numeric type
# MAGIC We start from the original Bronze table, **not** Notebook 02's fitted outputs.

# COMMAND ----------

from pyspark.sql import functions as F
telco_df = spark.table(
    "workspace.default.telco_raw_bronze"
    )
pipeline_input_df = telco_df.withColumn(
    "TotalCharges",
    F.when(F.trim(F.col("TotalCharges").cast("string")) == "", None)
     .otherwise(F.col("TotalCharges")).cast("double"))
     
train_df, test_df = pipeline_input_df.randomSplit([0.8, 0.2], seed=42)
print("Training:", train_df.count(), "Testing:", test_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define all transformations
# MAGIC `Imputer` learns the median; `StringIndexer` learns category labels; `OneHotEncoder` creates binary vectors; `StandardScaler` learns training means and standard deviations; `VectorAssembler` builds the final feature vector.
# MAGIC
# MAGIC We also include `SeniorCitizen` as a numeric feature, as in our planned complete pipeline.

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
categorical_cols = ['gender', 
                    'Partner', 
                    'Dependents', 
                    'PhoneService', 
                    'MultipleLines', 
                    'InternetService', 
                    'OnlineSecurity', 
                    'OnlineBackup', 
                    'DeviceProtection', 
                    'TechSupport', 
                    'StreamingTV', 
                    'StreamingMovies', 
                    'Contract', 
                    'PaperlessBilling', 
                    'PaymentMethod']
imputer = Imputer(
    inputCols=["TotalCharges"], 
    outputCols=["TotalCharges_imputed"], 
    strategy="median")
indexers = [StringIndexer(
    inputCol=c, 
    outputCol=f"{c}_index", handleInvalid="keep"
    )
          for c in categorical_cols]
encoder = OneHotEncoder(
    inputCols=[f"{c}_index" for c in categorical_cols],
    outputCols=[f"{c}_encoded" for c in categorical_cols], 
    handleInvalid="keep"
    )
num_assembler = VectorAssembler(
    inputCols=["tenure", 
               "MonthlyCharges", 
               "TotalCharges_imputed", 
               "SeniorCitizen"],
    outputCol="numerical_features"
    )
scaler = StandardScaler(
    inputCol="numerical_features", 
    outputCol="scaled_numerical_features",
    withMean=True, 
    withStd=True
    )
final_assembler = VectorAssembler(
    inputCols=["scaled_numerical_features"] + [f"{c}_encoded" for c in categorical_cols],
    outputCol="features"
    )
pipeline = Pipeline(
    stages=[imputer, *indexers, encoder, num_assembler, scaler, final_assembler]
    )
print("Pipeline stages:", len(pipeline.getStages()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit on train, transform both sets
# MAGIC **Fit** learns parameters. **Transform** applies learned parameters. We never fit separately on the test set.

# COMMAND ----------

pipeline_model = pipeline.fit(train_df)
train_prepared = pipeline_model.transform(train_df)
test_prepared = pipeline_model.transform(test_df)
print("Training:", train_prepared.count(), "Testing:", test_prepared.count())
display(train_prepared.select("customerID", "Churn", "features").limit(5))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verify feature length
# MAGIC One-hot vectors expand categories into multiple positions, so the final feature vector is longer than the 15 categorical columns.

# COMMAND ----------

from pyspark.ml.functions import vector_to_array
feature_check = train_prepared.select(F.size(vector_to_array("features")).alias("feature_count"))
print("Number of final vector positions:", feature_check.first()["feature_count"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Save train/test splits and fitted pipeline
# MAGIC The raw split tables let later notebooks reproduce the same observations. The saved pipeline ensures future preprocessing uses the **training-fitted** statistics.
# MAGIC
# MAGIC **Caution:** `overwrite` replaces the existing practice artifacts. Do not rerun after training a model unless you intend to regenerate the entire experiment.

# COMMAND ----------

train_table = "workspace.default.telco_train_raw"
test_table = "workspace.default.telco_test_raw"
train_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(train_table)
test_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(test_table)
pipeline_path = "/Volumes/workspace/default/telco_data/telco_preprocessing_pipeline"
pipeline_model.write().overwrite().save(pipeline_path)
print("Saved raw splits:", train_table, test_table)
print("Saved fitted pipeline:", pipeline_path)

# COMMAND ----------


from pyspark.ml import PipelineModel

# Load the previously saved preprocessing pipeline.
pipeline_path = (
    "/Volumes/workspace/default/telco_data/"
    "telco_preprocessing_pipeline"
)

loaded_pipeline = PipelineModel.load(pipeline_path)

# Load the saved raw testing dataset.
test_df = spark.table(
    "workspace.default.telco_test_raw"
)

# Apply the saved transformations.
test_prepared = loaded_pipeline.transform(test_df)

# Confirm the number of records.
print("Test records:", test_prepared.count())

# Confirm the pipeline has 20 stages.
print("Pipeline stages:", len(loaded_pipeline.stages))

# Examine the generated features.
display(
    test_prepared.select(
        "customerID",
        "Churn",
        "features"
    ).limit(5)
)
