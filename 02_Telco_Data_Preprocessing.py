# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Data Preprocessing
# MAGIC **Goal:** Clean missing numeric values, encode categories, and standardize numbers.
# MAGIC
# MAGIC **Learning-only note:** Here we fit transformations on the whole dataset to see how they work. **Do not use these fitted transformations to evaluate a predictive model**; Notebook 03 will fit them on training data only.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read Bronze data
# MAGIC Notebooks do not share Python variables automatically. Reading the saved table makes this notebook self-contained.

# COMMAND ----------

from pyspark.sql import functions as F
telco_df = spark.table("workspace.default.telco_raw_bronze")
print("Rows:", telco_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Convert TotalCharges to a number
# MAGIC `TotalCharges` may be text because some records contain spaces. We trim whitespace, replace empty values with `null`, then cast to `double` (a decimal numeric type).

# COMMAND ----------


from pyspark.sql import functions as F

telco_clean_df = telco_df.withColumn(
    "TotalCharges",
    F.when(
        F.trim(F.col("TotalCharges").cast("string")) == "",
        None
    ).otherwise(F.col("TotalCharges")).cast("double")
)

telco_clean_df.select(
    "customerID", "tenure", "MonthlyCharges", "TotalCharges"
).show(10)


# COMMAND ----------


telco_clean_df.select(
    F.count(
        F.when(F.col("TotalCharges").isNull(), 1)
    ).alias("Missing_TotalCharges")
).show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Impute missing charges
# MAGIC **Imputation** means filling missing values. The median is the middle value after sorting; it is often less affected by extreme values than the mean. The original column stays visible for comparison.

# COMMAND ----------


from pyspark.ml.feature import Imputer

imputer = Imputer(
    inputCols=["TotalCharges"],
    outputCols=["TotalCharges_imputed"],
    strategy="median"
)

imputer_model = imputer.fit(telco_clean_df)

telco_imputed_df = imputer_model.transform(telco_clean_df)

display(
    telco_imputed_df.select(
        "customerID",
        "TotalCharges",
        "TotalCharges_imputed"
    ).filter(F.col("TotalCharges").isNull())
)



# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Select categorical columns
# MAGIC A **categorical feature** holds labels, such as `Month-to-month` or `Electronic check`. We exclude `customerID` (an identifier) and `Churn` (the prediction target).

# COMMAND ----------


categorical_cols = [
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

print("Categorical features:", len(categorical_cols))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Convert category labels to indices
# MAGIC `StringIndexer` learns an integer code for each category. The codes are labels, **not meaningful numeric distances**. `handleInvalid="keep"` handles unseen or null categories.

# COMMAND ----------


from pyspark.ml.feature import StringIndexer

indexers = [
    StringIndexer(
        inputCol=col,
        outputCol=f"{col}_index",
        handleInvalid="keep"
    )
    for col in categorical_cols
]

indexed_df = telco_imputed_df

for indexer in indexers:
    indexed_df = indexer.fit(indexed_df).transform(indexed_df)

display(
    indexed_df.select(
        "Contract",
        "Contract_index",
        "PaymentMethod",
        "PaymentMethod_index"
    ).limit(10)
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. One-hot encode category indices
# MAGIC `OneHotEncoder` makes a vector of zeroes and ones for each category. Sparse vectors store only nonzero positions. For example, `(4,[0],[1.0])` means a length-four vector with 1 at index 0.

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder
encoder = OneHotEncoder(
    inputCols=[f"{c}_index" for c in categorical_cols],
    outputCols=[f"{c}_encoded" for c in categorical_cols],
    handleInvalid="keep"
)
encoder_model = encoder.fit(indexed_df)

telco_encoded_df = encoder_model.transform(indexed_df)

encoded_cols = [f"{c}_encoded" for c in categorical_cols]

print("Encoded columns:", len(encoded_cols))

telco_encoded_df.select("Contract", "Contract_encoded", "PaymentMethod", "PaymentMethod_encoded").show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Assemble numeric features
# MAGIC A **feature vector** is an ordered list of numbers for one customer. `VectorAssembler` combines three numeric columns into one vector.

# COMMAND ----------


from pyspark.ml.feature import VectorAssembler

numerical_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges_imputed"
]

numerical_assembler = VectorAssembler(
    inputCols=numerical_cols,
    outputCol="numerical_features"
)

telco_assembled_df = numerical_assembler.transform(
    telco_encoded_df
)

display(
    telco_assembled_df.select(
       *numerical_cols,
        "numerical_features"
    ).limit(5)
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Standardize numerical values
# MAGIC **Standardization** subtracts the feature mean and divides by its standard deviation. This places features with different units (months and dollars) on comparable scales. `withMean=True` centers values at zero; `withStd=True` scales their spread.

# COMMAND ----------

from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(
    inputCol="numerical_features",
    outputCol="scaled_numerical_features",
    withMean=True, 
    withStd=True
)
scaler_model = scaler.fit(telco_assembled_df)

telco_scaled_df = scaler_model.transform(telco_assembled_df)

print("Columns:", numerical_cols)
print("Means:", scaler_model.mean)
print("Standard deviations:", scaler_model.std)
telco_scaled_df.select("numerical_features", "scaled_numerical_features").show(5, truncate=False)