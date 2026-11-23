# Databricks notebook source
# MAGIC %md
# MAGIC # 05A — Telco Customer Churn: Findings & Model Evaluation Report
# MAGIC
# MAGIC **Goal:** Present the results of Notebooks 01–05 as a readable, reproducible report using Markdown tables, charts, and a workflow diagram. This report comes **after Notebook 05** and **before Notebook 06 — Model Optimization**.
# MAGIC
# MAGIC **Dataset:** IBM Telco Customer Churn (7,043 customer records).  
# MAGIC **Model:** PySpark Logistic Regression.  
# MAGIC **Evaluation:** 1,409 held-out testing records.  
# MAGIC
# MAGIC **How to use:** Run the Python cells from top to bottom in Databricks. The findings already measured in earlier notebooks are documented below; additional charts and the confusion matrix are recalculated from saved data and models. No retraining is required.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Executive summary
# MAGIC
# MAGIC This project investigates whether customer characteristics can help predict telecommunications customer churn. The original data contains **7,043 customers**; **1,869 (26.54%)** churned and **5,174 (73.46%)** did not.
# MAGIC
# MAGIC The fitted Logistic Regression model achieved **80.06% accuracy** and **0.8385 ROC-AUC** on the held-out test set. However, **52.48% recall** means that it identified only about half of actual churners at the model's current classification threshold.
# MAGIC
# MAGIC **Main takeaway:** The model is a useful starting point for churn-risk ranking, but improving churn recall and understanding the business cost of false positives and false negatives should come before operational deployment.
# MAGIC
# MAGIC | Measure | Observed result |
# MAGIC |---|---:|
# MAGIC | Customers in source data | 7,043 |
# MAGIC | Training customers | 5,634 |
# MAGIC | Testing customers | 1,409 |
# MAGIC | Preprocessing pipeline stages | 20 |
# MAGIC | Numerical feature positions | 60 |
# MAGIC | Model type | Logistic Regression |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Business problem and evaluation question
# MAGIC
# MAGIC **Business question:** Can historical customer attributes help us identify customers likely to cancel service, so a retention team can prioritize outreach?
# MAGIC
# MAGIC - **Input features** include customer tenure, charges, contract type, service subscriptions, and payment method.
# MAGIC - **Target:** `Churn` (`Yes` = 1; `No` = 0).
# MAGIC - **Prediction:** Estimated churn probability and a predicted class.
# MAGIC - **Success criterion:** Identify churners reliably while keeping unnecessary outreach manageable.
# MAGIC
# MAGIC **Important:** The dataset shows historical relationships. It does not establish that any individual feature *causes* churn, or that outreach will prevent it.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data overview and churn distribution
# MAGIC
# MAGIC The original dataset has **21 columns** and **7,043 rows**. The churn distribution is imbalanced: approximately three-quarters of customers stayed, while about one-quarter churned.
# MAGIC
# MAGIC | Churn status | Customers | Percentage |
# MAGIC |---|---:|---:|
# MAGIC | No | 5,174 | 73.46% |
# MAGIC | Yes | 1,869 | 26.54% |
# MAGIC | **Total** | **7,043** | **100.00%** |
# MAGIC
# MAGIC **Why this matters:** Accuracy alone can be misleading when one outcome is more common. A model predicting `No` for everyone would achieve 73.46% accuracy on the **full original dataset**, while finding no churners.

# COMMAND ----------

# Chart 1 — Churn distribution from the saved, unchanged Bronze table.
# We calculate counts directly instead of hardcoding them into the chart.
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

bronze_df = spark.table("workspace.default.telco_raw_bronze")
churn_rows = bronze_df.groupBy("Churn").count().collect()
churn_counts = {row["Churn"]: row["count"] for row in churn_rows}
labels = ["No", "Yes"]
counts = [churn_counts.get(x, 0) for x in labels]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(labels, counts)
ax.set(title="Customer churn distribution", xlabel="Churn", ylabel="Number of customers")
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, count + 35,
            f"{count:,} ({count/sum(counts):.1%})", ha="center")
ax.set_ylim(0, max(counts) * 1.17)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Data quality findings
# MAGIC
# MAGIC The missing-value audit examined both database `null` values and blank strings. **11 `TotalCharges` values were blank**; the other original columns had no detected missing or blank values.
# MAGIC
# MAGIC | Data-quality finding | Result | Treatment |
# MAGIC |---|---:|---|
# MAGIC | Blank `TotalCharges` | 11 | Convert blanks to null, then median-impute |
# MAGIC | Other missing/blank values | 0 detected | No imputation required |
# MAGIC | Original `TotalCharges` data type | String | Convert to `double` for modeling |
# MAGIC
# MAGIC **Why median imputation?** It provides a numeric replacement for missing charges and is generally less sensitive to extreme values than the mean. In the reusable pipeline, imputation is fitted on **training data only**, then applied to the test data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data preparation and model workflow
# MAGIC
# MAGIC This diagram summarizes the stages already completed in Notebooks 01–05.
# MAGIC
# MAGIC ```text
# MAGIC                  IBM TELCO CUSTOMER CHURN CSV
# MAGIC                            7,043 rows
# MAGIC                                |
# MAGIC                                v
# MAGIC                     01 — DATA EXPLORATION
# MAGIC                   Raw Bronze Delta table
# MAGIC                                |
# MAGIC                                v
# MAGIC                   02 — PREPROCESSING LAB
# MAGIC           Missing values, encoding, scaling concepts
# MAGIC                                |
# MAGIC                                v
# MAGIC                  03 — FEATURE PIPELINE
# MAGIC                Split into training and testing
# MAGIC                   /                       \
# MAGIC                  v                         v
# MAGIC         TRAIN: 5,634 rows           TEST: 1,409 rows
# MAGIC                  |                         |
# MAGIC        Fit preprocessing          Apply fitted pipeline
# MAGIC           (20 stages)               (no refitting)
# MAGIC                  |                         |
# MAGIC                  v                         v
# MAGIC            60 features                60 features
# MAGIC                  |                         |
# MAGIC                  v                         |
# MAGIC        04 — TRAIN LOGISTIC                    |
# MAGIC           REGRESSION MODEL                   |
# MAGIC                  |                         |
# MAGIC                  +------------+------------+
# MAGIC                               |
# MAGIC                               v
# MAGIC                     05 — MODEL EVALUATION
# MAGIC               Accuracy, precision, recall,
# MAGIC                  F1-score and ROC-AUC
# MAGIC ```
# MAGIC
# MAGIC **Leakage prevention:** The imputer, category indexers, encoders and scaler learn their parameters from the training split, not the test split.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Evaluation metrics: recorded results
# MAGIC
# MAGIC These are the **actual scores** you obtained in Notebook 05, not illustrative values.
# MAGIC
# MAGIC | Metric | Score | Meaning |
# MAGIC |---|---:|---|
# MAGIC | Accuracy | **80.06%** | Proportion of all test customers correctly classified |
# MAGIC | Precision (Churn) | **67.00%** | Among customers flagged as churners, proportion who churned |
# MAGIC | Recall (Churn) | **52.48%** | Among actual churners, proportion identified |
# MAGIC | F1-score (Churn) | **58.86%** | Harmonic mean of churn precision and recall |
# MAGIC | ROC-AUC | **83.85%** | Ability to rank churners ahead of non-churners across thresholds |
# MAGIC
# MAGIC **Interpretation:** Recall is the clearest weakness. Approximately **47.52% of actual churners** were missed at the current threshold. The ROC-AUC suggests that threshold tuning may be worth investigating, but it does not guarantee a better business outcome.

# COMMAND ----------

# Chart 2 — Model evaluation scores reported in Notebook 05.
# These values are the results you already measured.
metrics = {
    "Accuracy": 0.8005677785663591,
    "Precision": 0.67,
    "Recall": 0.5248041775456919,
    "F1-score": 0.5885797950219619,
    "ROC-AUC": 0.8384534733991994,
}

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.barh(list(metrics), [100 * x for x in metrics.values()])
ax.invert_yaxis()
ax.set(xlim=(0, 100), xlabel="Score (%)", title="Held-out test-set evaluation metrics")
for bar, value in zip(bars, metrics.values()):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f"{value:.2%}", va="center")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Confusion matrix: what kinds of mistakes occur?
# MAGIC
# MAGIC A **confusion matrix** counts correct and incorrect classifications. It is especially useful for seeing how many churners were missed.
# MAGIC
# MAGIC | Actual outcome | Predicted no churn | Predicted churn |
# MAGIC |---|---|---|
# MAGIC | **No churn** | True negative (TN) | False positive (FP) |
# MAGIC | **Churn** | False negative (FN) | True positive (TP) |
# MAGIC
# MAGIC **Business interpretation:** False negatives are churners the retention team may fail to contact. False positives are customers who might receive unnecessary retention outreach.
# MAGIC
# MAGIC The following code **recalculates the actual counts** from the saved model and held-out test data; we do not invent counts from rounded summary metrics.

# COMMAND ----------

# Reload the saved pipeline and classifier; no retraining is needed.
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel

volume_root = "/Volumes/workspace/default/telco_data/"
loaded_pipeline = PipelineModel.load(volume_root + "telco_preprocessing_pipeline")
loaded_model = LogisticRegressionModel.load(volume_root + "telco_logistic_regression_model")
test_df = spark.table("workspace.default.telco_test_raw")

# Generate the features, the true numeric label and the model prediction.
test_prepared = loaded_pipeline.transform(test_df)
test_labeled = test_prepared.withColumn(
    "label",
    F.when(F.col("Churn") == "Yes", 1.0)
     .when(F.col("Churn") == "No", 0.0)
     .otherwise(None).cast("double")
)
assert test_labeled.filter(F.col("label").isNull()).count() == 0
predictions = loaded_model.transform(test_labeled)

# Counts: rows are actual outcomes, columns are predicted outcomes.
confusion_rows = predictions.groupBy("label", "prediction").count().collect()
counts_by_pair = {(int(r["label"]), int(r["prediction"])): r["count"]
                  for r in confusion_rows}
tn = counts_by_pair.get((0, 0), 0)
fp = counts_by_pair.get((0, 1), 0)
fn = counts_by_pair.get((1, 0), 0)
tp = counts_by_pair.get((1, 1), 0)
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print("Total:", tn + fp + fn + tp)
assert tn + fp + fn + tp == 1409

fig, ax = plt.subplots(figsize=(6, 5))
values = [[tn, fp], [fn, tp]]
image = ax.imshow(values, cmap="Blues")
ax.set_xticks([0, 1], ["No churn", "Churn"])
ax.set_yticks([0, 1], ["No churn", "Churn"])
ax.set(xlabel="Predicted", ylabel="Actual", title="Test-set confusion matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(values[i][j]), ha="center", va="center")
fig.colorbar(image, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Compare with a simple baseline
# MAGIC
# MAGIC A **majority-class baseline** always predicts the most common class (`No churn`). It provides a useful minimum reference for accuracy but has **zero churn recall**.
# MAGIC
# MAGIC We calculate this baseline on the **held-out test set** rather than assuming its class distribution exactly matches the full dataset.
# MAGIC
# MAGIC **Checkpoint:** The baseline accuracy will be the fraction of `No` records in the 1,409-row test split.

# COMMAND ----------

# Accuracy of a model that always predicts "No churn".
test_total = test_df.count()
no_churn_count = test_df.filter(F.col("Churn") == "No").count()
baseline_accuracy = no_churn_count / test_total
print(f"Test-set majority-class baseline accuracy: {baseline_accuracy:.2%}")
print("Baseline churn recall: 0.00%")
print(f"Logistic Regression accuracy (recorded): {metrics['Accuracy']:.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Findings, limitations, and next actions
# MAGIC
# MAGIC ### Findings
# MAGIC
# MAGIC 1. **Churn is a minority outcome:** 26.54% of customers in the source data churned.
# MAGIC 2. **The raw data needed limited missing-value repair:** 11 blank `TotalCharges` entries were detected.
# MAGIC 3. **The pipeline is reproducible:** 20 fitted preprocessing stages produce 60 feature positions.
# MAGIC 4. **The model provides useful risk discrimination:** ROC-AUC was 0.8385.
# MAGIC 5. **Churn recall needs attention:** At the evaluated threshold, about 47.52% of churners were missed.
# MAGIC
# MAGIC ### Limitations
# MAGIC
# MAGIC - These results are based on one train/test split, not repeated or time-based validation.
# MAGIC - A score is not a causal explanation or proof that a retention offer would work.
# MAGIC - No outreach cost, customer lifetime value, calibration, or fairness analysis has yet been reported.
# MAGIC - We have already examined the test results; any later optimization informed by them should be treated as exploratory until assessed on a fresh holdout.
# MAGIC
# MAGIC ### Recommended next work
# MAGIC
# MAGIC Proceed to **06 — Telco Model Optimization**. Use a separate validation split to compare thresholds and precision/recall trade-offs. Consider class weighting, calibration, and business-cost analysis before choosing a deployment threshold.
# MAGIC
# MAGIC **Report conclusion:** The first model is a credible baseline for further experimentation, not yet a validated production retention decision system.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Threshold optimization: experiment design
# MAGIC
# MAGIC Notebook 06 split the original training records into **development training** and **validation** subsets, fitted fresh preprocessing on development training only, and selected a threshold by **highest validation F1-score** among thresholds 0.25–0.60. It then applied the chosen threshold to the original saved Logistic Regression model on the test set.
# MAGIC
# MAGIC **Model selection:** The validation-selected threshold was **0.40**, compared with the original **0.50**.
# MAGIC
# MAGIC **Important limitation:** The test split had already been evaluated in Notebook 05. Although threshold selection used a separate validation set, the test set is not a completely untouched final benchmark for the overall project. A fresh external or future-period dataset would provide stronger confirmation.
# MAGIC
# MAGIC ### Workflow diagram
# MAGIC
# MAGIC ```text
# MAGIC Original dataset (7,043 rows)
# MAGIC             |
# MAGIC        Train / test split
# MAGIC             |
# MAGIC      +------+----------------------+
# MAGIC      |                             |
# MAGIC Train data (5,634)             Test data (1,409)
# MAGIC      |                             |
# MAGIC Development / validation            |  Final comparison only
# MAGIC      |                             |
# MAGIC Refit preprocessing                 |
# MAGIC Train candidate Logistic Reg.      |
# MAGIC Choose threshold by validation F1  |
# MAGIC      |                             |
# MAGIC      +-------- selected 0.40 ------+
# MAGIC                                    |
# MAGIC                        Saved full-training model
# MAGIC                                    |
# MAGIC                        Compare 0.50 vs 0.40
# MAGIC ```
# MAGIC
# MAGIC **Checkpoint:** The selection rule is documented before interpreting test results.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Optimization results: original versus validation-selected threshold
# MAGIC
# MAGIC These are the **observed test-set results you reported** from Notebook 06. The precision–recall trade-off is the primary finding.
# MAGIC
# MAGIC | Metric | Original threshold 0.50 | Selected threshold 0.40 | Change (percentage points) |
# MAGIC |---|---:|---:|---:|
# MAGIC | Accuracy | 80.06% | 79.21% | −0.85 |
# MAGIC | Precision (churn) | 67.00% | 61.48% | −5.52 |
# MAGIC | **Recall (churn)** | **52.48%** | **62.92%** | **+10.44** |
# MAGIC | **F1-score (churn)** | **58.86%** | **62.19%** | **+3.34** |
# MAGIC
# MAGIC **Interpretation:** At threshold 0.40, the classifier catches approximately **63 rather than 52 of every 100 actual churners**, at the cost of flagging more non-churners. ROC-AUC is unchanged by threshold-only adjustments to the same probability scores; Notebook 05 reported ROC-AUC **0.83845**.
# MAGIC
# MAGIC The next code cell reads the saved optimization Delta table and checks that the expected two rows exist. It does not overwrite your results.

# COMMAND ----------

# Read actual experiment results from Notebook 06.
optimization_table = "workspace.default.telco_optimization_results"
optimization_df = spark.table(optimization_table)
print("Optimization comparison rows:", optimization_df.count())
print("Columns:", optimization_df.columns)
display(optimization_df)
assert optimization_df.count() == 2, "Expected two model/threshold comparison rows."

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Visual comparison: accuracy, precision, recall and F1
# MAGIC
# MAGIC The following grouped bar chart uses the **recorded results from Notebook 06**. A higher bar is not always preferable in isolation: accuracy and precision fell while recall and F1 improved.
# MAGIC
# MAGIC **Checkpoint:** You should see four metric groups, each with original and selected-threshold bars.

# COMMAND ----------

# Chart — model performance by threshold (observed test results).
import matplotlib.pyplot as plt
import numpy as np

metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]
original_scores = [0.8005677785663591, 0.67, 0.5248041775456919, 0.5885797950219619]
selected_scores = [0.7920511000709723, 0.6147959183673469, 0.6292428198433421, 0.6219354838709678]

x = np.arange(len(metric_names))
fig, ax = plt.subplots(figsize=(10, 5))
w = 0.36
bars_original = ax.bar(x - w/2, np.array(original_scores) * 100, w, label="Original (0.50)")
bars_selected = ax.bar(x + w/2, np.array(selected_scores) * 100, w, label="Validation-selected (0.40)")
ax.set_xticks(x, metric_names)
ax.set_ylim(0, 100)
ax.set_ylabel("Score (%)")
ax.set_title("Telco churn: model evaluation before and after threshold optimization")
ax.legend()
ax.grid(axis="y", alpha=0.2)
for bars in (bars_original, bars_selected):
    ax.bar_label(bars, fmt="%.1f", padding=3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Business implications and decision trade-off
# MAGIC
# MAGIC A lower classification threshold identifies **more customers who are likely to churn**, potentially giving a retention team more opportunities to intervene. The benefit is not automatic: lower precision also means more retention outreach to customers who would have stayed.
# MAGIC
# MAGIC | Business consideration | Threshold 0.50 | Threshold 0.40 |
# MAGIC |---|---|---|
# MAGIC | Actual churners detected | Lower (~52 per 100) | Higher (~63 per 100) |
# MAGIC | False negatives (missed churners) | More | Fewer |
# MAGIC | False positives (unnecessary outreach) | Fewer | More |
# MAGIC | Operational implication | Smaller, more selective contact list | Broader contact list; greater contact cost |
# MAGIC
# MAGIC **Decision:** Keep **0.40 as a candidate threshold** because it improved recall by **10.44 percentage points** and F1 by **3.34 points**. Do not call it a proven financial optimum until outreach costs, retention effectiveness, and customer value are estimated.
# MAGIC
# MAGIC A useful next experiment is a **cost-sensitive decision analysis**: estimate expected net benefit for each threshold, given the cost of contacting a customer and the value of preventing churn.
# MAGIC
# MAGIC ### Limitations
# MAGIC
# MAGIC - The test set has already informed earlier analysis; assess generalization with a new future-period or external holdout.
# MAGIC - No customer-contact costs or treatment effectiveness were provided, so financial ROI cannot be calculated.
# MAGIC - The results are specific to this dataset and fitted Logistic Regression model.
# MAGIC - Classification-threshold tuning does **not** change the model's ranking ability or ROC-AUC.
# MAGIC
# MAGIC **Next notebook:** `07_Telco_Advanced_Model_Training_and_Comparison` — compare Logistic Regression, Random Forest, and Gradient-Boosted Trees under consistent train/validation/test rules.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Saved artifacts and reproducibility
# MAGIC
# MAGIC | Asset | Location |
# MAGIC |---|---|
# MAGIC | Original Bronze table | `workspace.default.telco_raw_bronze` |
# MAGIC | Training split | `workspace.default.telco_train_raw` |
# MAGIC | Test split | `workspace.default.telco_test_raw` |
# MAGIC | Original evaluation metrics | `workspace.default.telco_model_evaluation` |
# MAGIC | Threshold optimization comparison | `workspace.default.telco_optimization_results` |
# MAGIC | Fitted preprocessing pipeline | `/Volumes/workspace/default/telco_data/telco_preprocessing_pipeline` |
# MAGIC | Original Logistic Regression model | `/Volumes/workspace/default/telco_data/telco_logistic_regression_model` |
# MAGIC
# MAGIC **Report conclusion:** The original classifier achieved **0.83845 ROC-AUC** and **80.06% accuracy** at threshold 0.50. Selecting threshold 0.40 using validation F1 improved test churn recall from **52.48% to 62.92%** and F1 from **58.86% to 62.19%**, with lower accuracy and precision. This trade-off merits further business-cost analysis and independent validation.