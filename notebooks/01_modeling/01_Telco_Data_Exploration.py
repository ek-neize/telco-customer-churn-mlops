# Databricks notebook source
# DBTITLE 1,Lab overview
# MAGIC %md
# MAGIC # 01 — Telco Data Exploration
# MAGIC
# MAGIC **Goal:** Find the data, load it, inspect its structure, understand churn, and save an unchanged Bronze copy.
# MAGIC
# MAGIC Run cells from top to bottom. A **DataFrame** is a table-like collection of rows and columns; Spark processes it across compute resources.

# COMMAND ----------

# DBTITLE 1,Verify your workspace
# MAGIC %md
# MAGIC ## 1. Verify your workspace
# MAGIC
# MAGIC `spark` is the Databricks Spark session. The catalog and schema identify where Unity Catalog tables are stored.

# COMMAND ----------


# Databricks Machine Learning Lab
print("Welcome to Telco Customer Churn Analysis")

# Check Spark version
print("Spark version:", spark.version)

# Check Unity Catalog configuration
spark.sql("""
SELECT
    current_catalog() AS catalog,
    current_schema() AS schema
""").show()


# COMMAND ----------

# DBTITLE 1,Inspect the uploaded files
# MAGIC %md
# MAGIC ## 2. Inspect the uploaded files
# MAGIC
# MAGIC A **volume** stores files. This command lists the contents; it does not read the CSV into a table.

# COMMAND ----------

volume_path = "/Volumes/workspace/default/telco_data/"
display(dbutils.fs.ls(volume_path))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Read the CSV
# MAGIC
# MAGIC `header=true` uses the first row as column names.  
# MAGIC `inferSchema=true` guesses numeric types.  
# MAGIC `multiLine` and `escape` help with quoted CSV values.
# MAGIC
# MAGIC **Checkpoint:** 7,043 rows and 21 columns for the standard dataset.
# MAGIC

# COMMAND ----------

# dataset_path = volume_path + "WA_Fn-UseC_-Telco-Customer-Churn.csv"
# telco_df = (spark.read.option("header", "true")
#     .option("inferSchema", "true")
#     .option("multiLine", "true")
#     .option("escape", '"')
#     .csv(dataset_path))
# print("Rows:", telco_df.count())
# print("Columns:", len(telco_df.columns))
# display(telco_df.limit(10))

# COMMAND ----------


# Path to the uploaded Telco dataset
dataset_path = "/Volumes/workspace/default/telco_data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Load dataset into Spark DataFrame
telco_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("multiLine", "true")
    .option("escape", '"')
    .csv(dataset_path)
)

# Check dimensions
print("Number of rows:", telco_df.count())
print("Number of columns:", len(telco_df.columns))

# Display first 10 records
display(telco_df.limit(10))


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 4. Inspect the schema
# MAGIC The **schema** lists each column and its data type. In particular, check whether `TotalCharges` is a string because it contains blanks.

# COMMAND ----------


# Examine column names and data types
telco_df.printSchema()



# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Count churn and calculate percentages
# MAGIC `groupBy` creates one group per churn category; `count` measures each group. The percentage is group size divided by all customers, multiplied by 100. We add a total row for reporting only.

# COMMAND ----------


from pyspark.sql import functions as F

# Total number of customers
total_customers = telco_df.count()

# Calculate customer count and percentage
churn_summary = (
    telco_df
    .groupBy("Churn")
    .agg(F.count("*").alias("Customers"))
    .withColumn(
        "Percentage",
        F.round(
            F.col("Customers") / total_customers * 100,
            2
        )
    )
)

# Create a total row
total_row = spark.createDataFrame(
    [("Total", total_customers, 100.0)],
    ["Churn", "Customers", "Percentage"]
)

# Combine and order results
churn_summary = (
    churn_summary
    .unionByName(total_row)
    .withColumn(
        "sort_order",
        F.when(F.col("Churn") == "No", 1)
         .when(F.col("Churn") == "Yes", 2)
         .otherwise(3)
    )
    .orderBy("sort_order")
    .drop("sort_order")
)

display(churn_summary)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Check nulls AND blank strings
# MAGIC A **null** is a missing database value. A blank string (`""`) looks empty but is not necessarily null. We check both. The standard file has 11 blank `TotalCharges` entries.

# COMMAND ----------


from pyspark.sql import functions as F

# Count null and blank values in each column
missing_values = telco_df.select([
    F.sum(
        F.when(
            F.col(c).isNull() |
            (F.trim(F.col(c).cast("string")) == ""),
            1
        ).otherwise(0)
    ).alias(c)
    for c in telco_df.columns
])

display(missing_values)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summarize numeric features
# MAGIC `describe()` shows counts, means, standard deviations and ranges. It helps spot implausible values.

# COMMAND ----------

display(telco_df.select("tenure", "MonthlyCharges").describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save the raw Bronze Delta table
# MAGIC **Bronze** means the raw ingested data, before cleaning. We keep it unchanged so later experiments can restart reliably. `overwrite` replaces an existing table of the same name—use only for this practice table.

# COMMAND ----------

bronze_table = "workspace.default.telco_raw_bronze"
(telco_df.write.format("delta").mode("overwrite")
 .option("overwriteSchema", "true").saveAsTable(bronze_table))
bronze_df = spark.table(bronze_table)
print("Saved:", bronze_table)
print("Rows:", bronze_df.count(), "Columns:", len(bronze_df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. View Delta history
# MAGIC Delta records table writes as versions. This is useful for reproducibility, although older data files are not retained forever.

# COMMAND ----------

display(spark.sql("DESCRIBE HISTORY workspace.default.telco_raw_bronze"))
