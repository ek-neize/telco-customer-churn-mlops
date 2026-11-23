# Databricks notebook source
# MAGIC %md
# MAGIC # 07A — Telco Advanced Model Comparison & Findings Report
# MAGIC
# MAGIC **Purpose:** Document the actual Notebook 07 validation and test results, interpret the business trade-offs, and provide reproducible Databricks visualizations. This report **does not retrain models**.
# MAGIC
# MAGIC **Sources:** User-reported outputs from Notebooks 05, 06, and 07; saved comparison table `workspace.default.telco_07_model_comparison`. Figures are rounded for display only.
# MAGIC
# MAGIC **Important limitation:** The preprocessing pipeline was fitted before the Notebook 07 validation split, so validation scores are exploratory. The test set has also been examined in earlier notebooks; future unbiased evaluation requires a fresh holdout or prospective data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment and reproducibility checkpoint
# MAGIC
# MAGIC Confirm Spark is available and the saved Notebook 07 comparison table exists. The remaining visualizations use reported values and optionally check the persisted table.

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt

comparison_table = "workspace.default.telco_07_model_comparison"
try:
    saved_comparison = spark.table(comparison_table)
    print("Saved comparison table:", comparison_table)
    display(saved_comparison)
except Exception as exc:
    print("Could not load the saved comparison table; report charts will use recorded Notebook 07 results.")
    print(str(exc)[:400])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Executive summary
# MAGIC
# MAGIC Three models were trained and compared on **1,126 validation customers**, with **4,508 fit customers** and **60 features**. Logistic Regression led on validation ROC-AUC and F1. The final candidate was evaluated on **1,409 test customers**.
# MAGIC
# MAGIC **Decision:** Retain Logistic Regression as the provisional leading algorithm; investigate threshold choice and business costs before deployment.

# COMMAND ----------

summary = pd.DataFrame([
    ("Fit records", 4508), ("Validation records", 1126),
    ("Test records", 1409), ("Features", 60),
    ("Preprocessing pipeline stages", 20)
], columns=["Project checkpoint", "Value"])
display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Model comparison on validation data
# MAGIC
# MAGIC The table below reproduces the **actual Section 7 outputs**. All metrics here were computed at the models' evaluation thresholds (normally 0.50). Do not compare these directly with the test metrics because they use different data.

# COMMAND ----------

validation = pd.DataFrame([
    ["Logistic Regression",0.7912966252220248,0.6478260869565218,0.49174917491749176,0.5590994371482176,0.8315167482726348],
    ["Gradient-Boosted Trees",0.7841918294849023,0.6260504201680672,0.49174917491749176,0.5508317929759704,0.8304340154550073],
    ["Random Forest",0.7806394316163411,0.6794871794871795,0.34983498349834985,0.4618736383442266,0.8301332563389973]
], columns=["Model","Accuracy","Precision","Recall","F1","ROC-AUC"])
display(validation.style.format({m:"{:.2%}" for m in validation.columns if m != "Model"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Validation performance chart
# MAGIC
# MAGIC Compare precision, recall, F1 and ROC-AUC visually. Random Forest has the highest precision but the lowest recall; Logistic Regression leads in F1 and ROC-AUC.

# COMMAND ----------

ax = validation.set_index("Model")[["Precision","Recall","F1","ROC-AUC"]].mul(100).plot(kind="bar",figsize=(12,5),rot=10)
ax.set_ylabel("Validation score (%)")
ax.set_ylim(0,100)
ax.set_title("Notebook 07: validation performance by model")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. How close are the ROC-AUC results?
# MAGIC
# MAGIC The validation ROC-AUC values range from approximately **0.8301 to 0.8315**. The largest gap is only about **0.0014**, so the observed ranking is not evidence of a large performance advantage. Confidence intervals or repeated cross-validation would help assess stability.

# COMMAND ----------

ax = validation.plot(x="Model",y="ROC-AUC",kind="bar",legend=False,figsize=(10,4),rot=10)
ax.set_ylim(0.80,0.85)
ax.set_ylabel("Validation ROC-AUC (zoomed scale)")
ax.set_title("Small ROC-AUC differences — note truncated vertical axis")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.4f}",(p.get_x()+p.get_width()/2,p.get_height()),ha="center",va="bottom")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Model selection rationale
# MAGIC
# MAGIC Logistic Regression was selected by **highest validation ROC-AUC** (0.831517). It also had the highest F1 (0.559099). This is a *provisional* choice because validation preprocessing leakage and small score differences limit the strength of the conclusion.

# COMMAND ----------

winner = validation.loc[validation["ROC-AUC"].idxmax()]
print("Validation ROC-AUC winner:", winner["Model"])
print("ROC-AUC:", round(winner["ROC-AUC"],6))
print("F1:", round(winner["F1"],6))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Final candidate performance on test data
# MAGIC
# MAGIC The selected Logistic Regression candidate achieved the following metrics on the 1,409-record test set. These exactly match the earlier Notebook 05 reported metrics; matching aggregate metrics do not prove that the fitted models or probabilities are identical.

# COMMAND ----------

test_metrics = pd.DataFrame([
    ("Accuracy",0.8005677785663591),
    ("Precision (churn)",0.67),
    ("Recall (churn)",0.5248041775456919),
    ("F1 (churn)",0.5885797950219619),
    ("ROC-AUC",0.8384534733991994)
],columns=["Metric","Score"])
display(test_metrics.style.format({"Score":"{:.2%}"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Test confusion matrix and business meaning
# MAGIC
# MAGIC Rows are **actual** outcomes and columns are **predicted** outcomes, where 1 = churn. Of **383 actual churners**, **201 were detected** and **182 were missed**. The 99 false positives are customers who were flagged for retention but did not churn in the observed labels.

# COMMAND ----------

import numpy as np
matrix = np.array([[927,99],[182,201]])
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(matrix,cmap="Blues")
for (i,j), value in np.ndenumerate(matrix):
    ax.text(j,i,str(value),ha="center",va="center",fontsize=16,
            color="white" if value > 500 else "black")
ax.set_xticks([0,1],labels=["No churn","Churn"])
ax.set_yticks([0,1],labels=["No churn","Churn"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Logistic Regression: final test confusion matrix")
fig.colorbar(im,ax=ax)
plt.tight_layout()
plt.show()
print("Total:",matrix.sum(),"Actual churners:",matrix[1].sum())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Notebook 06 threshold optimization: original vs 0.40
# MAGIC
# MAGIC The earlier experiment lowered the classification threshold from **0.50 to 0.40**. On its reported test evaluation, recall increased by about **10.44 percentage points** and F1 by **3.34 points**, while precision and accuracy fell. The 0.40 threshold was validation-selected, but test data has already been reviewed, so treat this as exploratory evidence.

# COMMAND ----------

thresholds = pd.DataFrame([
    ["Original (0.50)",0.50,0.8005677785663591,0.67,0.5248041775456919,0.5885797950219619],
    ["Validation-selected (0.40)",0.40,0.7920511000709723,0.6147959183673469,0.6292428198433421,0.6219354838709678]
],columns=["Setting","Threshold","Accuracy","Precision","Recall","F1"])
display(thresholds.style.format({"Threshold":"{:.2f}",**{m:"{:.2%}" for m in ["Accuracy","Precision","Recall","F1"]}}))
print("Recall gain (percentage points):",round(100*(thresholds.loc[1,"Recall"]-thresholds.loc[0,"Recall"]),2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Threshold trade-off chart
# MAGIC
# MAGIC A lower threshold flags more customers as at risk. This raises recall but also increases false positives. Whether this is valuable depends on outreach cost, retention-offer cost, and the financial impact of losing a customer.

# COMMAND ----------

ax = thresholds.set_index("Setting")[["Accuracy","Precision","Recall","F1"]].mul(100).T.plot(kind="bar",figsize=(10,5),rot=0)
ax.set_ylabel("Test score (%)")
ax.set_ylim(0,100)
ax.set_title("Notebook 06: impact of threshold selection")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Workflow diagram and limitations
# MAGIC
# MAGIC ```text
# MAGIC IBM Telco CSV (7,043 customers)
# MAGIC           |
# MAGIC           v
# MAGIC Bronze Delta + data exploration
# MAGIC           |
# MAGIC           v
# MAGIC Preprocessing pipeline (20 stages, 60 features)
# MAGIC           |
# MAGIC           v
# MAGIC Train / validation: 4,508 / 1,126
# MAGIC           |
# MAGIC           +--> Logistic Regression ------+
# MAGIC           +--> Random Forest ------------+--> Validation comparison
# MAGIC           +--> Gradient-Boosted Trees ---+           |
# MAGIC                                                     v
# MAGIC                                        Select Logistic Regression
# MAGIC                                                     |
# MAGIC                                                     v
# MAGIC                                        Test evaluation (1,409)
# MAGIC                                                     |
# MAGIC                                                     v
# MAGIC                                       Threshold/business analysis
# MAGIC ```
# MAGIC
# MAGIC **Limitations:** preprocessing was fitted before the validation split; the test set was used repeatedly; a single split does not measure variability; the model comparison did not establish statistically meaningful differences; threshold and cost choices need further validation.

# COMMAND ----------

limitations = pd.DataFrame([
    ("Preprocessing leakage", "Refit imputers, encoders, and scalers inside each training fold"),
    ("Repeated test inspection", "Reserve a new untouched holdout or prospective dataset"),
    ("Single validation split", "Use stratified cross-validation and uncertainty intervals"),
    ("Business trade-off", "Estimate cost per outreach, false alarm, and lost customer"),
    ("Deployment readiness", "Check calibration, fairness, drift, and operational monitoring")
],columns=["Risk","Recommended action"])
display(limitations)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Conclusions and recommended next steps
# MAGIC
# MAGIC **Finding 1:** Logistic Regression is the best *observed* candidate on validation ROC-AUC and F1, but only narrowly.
# MAGIC
# MAGIC **Finding 2:** At threshold 0.50, test recall is **52.48%** (201/383 churners detected). The model misses 182 churners.
# MAGIC
# MAGIC **Finding 3:** The earlier 0.40 threshold experiment raised reported recall to **62.92%**, at the cost of lower precision and accuracy.
# MAGIC
# MAGIC **Recommendations:** (1) Rebuild a leakage-free training/validation pipeline, (2) run stratified cross-validation and tune thresholds on out-of-fold predictions, (3) select a threshold based on business value, (4) evaluate once on fresh unseen data, and (5) use Notebook 08 to track experiments and register approved models with MLflow.

# COMMAND ----------

print("Report complete: Notebook 07A")
print("Provisional leading model: Logistic Regression")
print("Saved Notebook 07 comparison table:",comparison_table)
print("Next recommended notebook: 08 — MLflow Experiment Tracking and Model Registry")

# COMMAND ----------

# MAGIC %md
# MAGIC