# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Telco Customer Churn Prediction
# MAGIC ## Notebook 10A — Automated Batch Scoring, Operational Monitoring and Feature Drift Findings
# MAGIC
# MAGIC **Project:** End-to-End Telco Customer Churn Prediction and MLOps  
# MAGIC **Platform:** Databricks, Apache Spark, MLflow, Unity Catalog and Delta Lake  
# MAGIC **Model:** Logistic Regression — Registered Model Version 1  
# MAGIC **Execution:** Databricks Workflow using Serverless Compute  
# MAGIC **Report Date:** November 17, 2026
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 1. Executive Summary
# MAGIC
# MAGIC ### 1.1 Business Context
# MAGIC
# MAGIC Customer churn is a significant challenge for telecommunications companies because losing existing subscribers can reduce recurring revenue and increase customer acquisition costs.
# MAGIC
# MAGIC This project develops a machine learning solution to identify customers at risk of leaving a telecommunications provider.
# MAGIC
# MAGIC The objective is to support targeted retention campaigns by assigning churn probabilities to customers and prioritizing those who may benefit from proactive engagement.
# MAGIC
# MAGIC ### 1.2 Solution Overview
# MAGIC
# MAGIC An end-to-end machine learning workflow was implemented using Databricks.
# MAGIC
# MAGIC The solution integrates:
# MAGIC
# MAGIC - A reusable Spark ML feature engineering pipeline.
# MAGIC - A Logistic Regression model registered in MLflow and Unity Catalog.
# MAGIC - Automated batch inference through a Databricks Workflow.
# MAGIC - Delta Lake tables for persistent prediction history.
# MAGIC - Operational data quality monitoring.
# MAGIC - Feature distribution drift detection using Population Stability Index (PSI) and Total Variation Distance (TVD).
# MAGIC - Batch-specific overwrite operations to support safe reprocessing.
# MAGIC
# MAGIC ### 1.3 Key Results
# MAGIC
# MAGIC | Metric | Result |
# MAGIC |---|---:|
# MAGIC | Customers scored | 1,409 |
# MAGIC | Unique customers | 1,409 |
# MAGIC | Missing churn probabilities | 0 |
# MAGIC | Average predicted churn probability | 25.99% |
# MAGIC | Customers flagged for potential retention | 400 |
# MAGIC | Low-risk customers | 1,009 |
# MAGIC | Medium-risk customers | 311 |
# MAGIC | High-risk customers | 89 |
# MAGIC | Features monitored for drift | 6 |
# MAGIC | Features exceeding configured drift thresholds | 0 |
# MAGIC
# MAGIC ### 1.4 Workflow Validation
# MAGIC
# MAGIC The Databricks Workflow was executed successfully using serverless compute.
# MAGIC
# MAGIC A controlled repeat-execution test was also performed using the same batch identifier.
# MAGIC
# MAGIC After reprocessing, the Delta tables contained:
# MAGIC
# MAGIC | Output Table | Verified Records |
# MAGIC |---|---:|
# MAGIC | Prediction history | 1,409 |
# MAGIC | Batch monitoring | 1 |
# MAGIC | Feature drift | 6 |
# MAGIC
# MAGIC The results confirm that batch-level replacement preserved the expected record counts without accumulating duplicate rows for the tested batch.
# MAGIC
# MAGIC ### 1.5 Key Findings
# MAGIC
# MAGIC 1. The registered machine learning model can be loaded and executed through a Databricks Workflow.
# MAGIC 2. The scoring pipeline generated valid churn probabilities for all 1,409 customers.
# MAGIC 3. A total of 400 customers were identified for potential retention outreach using the configured decision threshold.
# MAGIC 4. All six monitored features were classified as NORMAL under the project's drift thresholds.
# MAGIC 5. The controlled repeat-execution test demonstrated successful batch-level replacement in the three output tables.
# MAGIC
# MAGIC ### 1.6 Limitations
# MAGIC
# MAGIC The workflow currently uses historical training and test datasets rather than a live production data feed.
# MAGIC
# MAGIC Consequently, the drift results demonstrate the monitoring implementation but do not establish stability across future customer populations.
# MAGIC
# MAGIC The workflow has been validated through manual Job executions. Its table-update trigger remains paused, and recurring production scheduling has not been enabled.
# MAGIC
# MAGIC The project is therefore a validated MLOps prototype rather than a live production deployment.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Executive Summary Conclusion:** The project demonstrates a reproducible batch scoring and monitoring workflow that connects machine learning predictions with operational quality controls and customer retention prioritization.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2. MLOps System Architecture
# MAGIC
# MAGIC ### 2.1 Architecture Overview
# MAGIC
# MAGIC The Telco Customer Churn MLOps solution uses Databricks to integrate data preparation, model management, batch inference, and operational monitoring.
# MAGIC
# MAGIC The architecture separates model development from model execution. A previously trained and registered Logistic Regression model is reused during batch scoring, rather than retrained whenever new scoring data is processed.
# MAGIC
# MAGIC The implementation uses the following components:
# MAGIC
# MAGIC 1. **Unity Catalog:** Governance and storage for source tables, Delta outputs, and the registered machine learning model.
# MAGIC 2. **Apache Spark ML:** Reusable feature preprocessing and transformation.
# MAGIC 3. **MLflow Model Registry:** Versioned model registration and retrieval.
# MAGIC 4. **Databricks Workflows:** Execution of the batch scoring notebook.
# MAGIC 5. **Serverless Compute:** Managed compute for executing the scoring workflow.
# MAGIC 6. **Delta Lake:** Persistent prediction history, monitoring metrics, and feature drift results.
# MAGIC
# MAGIC ### 2.2 End-to-End Data Flow
# MAGIC
# MAGIC **Stage 1 — Source Data**
# MAGIC
# MAGIC The batch scoring workflow reads customer records from:
# MAGIC
# MAGIC `workspace.default.telco_test_raw`
# MAGIC
# MAGIC The drift monitoring process uses the historical training dataset as its reference:
# MAGIC
# MAGIC `workspace.default.telco_train_raw`
# MAGIC
# MAGIC **Stage 2 — Feature Preprocessing**
# MAGIC
# MAGIC A previously fitted Spark ML preprocessing pipeline is loaded from:
# MAGIC
# MAGIC `/Volumes/workspace/default/telco_data/telco_preprocessing_pipeline`
# MAGIC
# MAGIC The pipeline transforms raw customer attributes into the numerical feature representation required by the model.
# MAGIC
# MAGIC The resulting feature vector contains 60 dimensions.
# MAGIC
# MAGIC **Stage 3 — Registered Model Loading**
# MAGIC
# MAGIC The workflow retrieves the registered Logistic Regression model from Unity Catalog using MLflow.
# MAGIC
# MAGIC Model URI:
# MAGIC
# MAGIC `models:/workspace.default.telco_churn_logistic_regression/1`
# MAGIC
# MAGIC This enables model versioning and reproducible inference without retraining during each scoring run.
# MAGIC
# MAGIC **Stage 4 — Batch Prediction**
# MAGIC
# MAGIC The model generates churn probabilities for each customer.
# MAGIC
# MAGIC The workflow applies the previously selected decision threshold of 0.40 to identify customers for potential retention outreach.
# MAGIC
# MAGIC Customers are also assigned Low, Medium, or High risk classifications.
# MAGIC
# MAGIC **Stage 5 — Prediction Persistence**
# MAGIC
# MAGIC Predictions are saved to:
# MAGIC
# MAGIC `workspace.default.telco_10_prediction_history`
# MAGIC
# MAGIC The table stores batch-specific scoring results, allowing predictions from different executions to be inspected and compared.
# MAGIC
# MAGIC **Stage 6 — Operational Monitoring**
# MAGIC
# MAGIC The workflow calculates operational metrics, including:
# MAGIC
# MAGIC - Total records scored.
# MAGIC - Number of unique customers.
# MAGIC - Missing churn probabilities.
# MAGIC - Average predicted churn probability.
# MAGIC - Number of customers flagged for retention.
# MAGIC - Distribution of customers across risk categories.
# MAGIC
# MAGIC Monitoring results are saved to:
# MAGIC
# MAGIC `workspace.default.telco_10_batch_monitoring`
# MAGIC
# MAGIC **Stage 7 — Feature Drift Detection**
# MAGIC
# MAGIC The workflow compares the scoring dataset with the historical reference dataset.
# MAGIC
# MAGIC Two drift metrics are used:
# MAGIC
# MAGIC - **Population Stability Index (PSI):** Measures changes in numerical feature distributions.
# MAGIC - **Total Variation Distance (TVD):** Measures changes in categorical feature distributions.
# MAGIC
# MAGIC Drift results are saved to:
# MAGIC
# MAGIC `workspace.default.telco_10_feature_drift`
# MAGIC
# MAGIC ### 2.3 Databricks Workflow Configuration
# MAGIC
# MAGIC | Configuration | Value |
# MAGIC |---|---|
# MAGIC | Job name | Telco_Churn_Batch_Scoring_Monitoring |
# MAGIC | Job ID | 738907313537928 |
# MAGIC | Task name | telco_batch_scoring |
# MAGIC | Task type | Notebook |
# MAGIC | Compute | Databricks Serverless |
# MAGIC | Model version | 1 |
# MAGIC | Batch identifier | `telco_{{job.run_id}}` |
# MAGIC | Maximum automatic retries | 3 |
# MAGIC | Table-update trigger | Paused |
# MAGIC
# MAGIC The batch identifier is passed to the notebook through a Databricks task parameter.
# MAGIC
# MAGIC This allows each new Job run to receive a distinct batch identifier while task retries within the same Job run can reuse the same identifier.
# MAGIC
# MAGIC ### 2.4 Batch Persistence and Retry Strategy
# MAGIC
# MAGIC The three output tables use Delta Lake batch-specific replacement operations.
# MAGIC
# MAGIC The implementation combines:
# MAGIC
# MAGIC - Delta format.
# MAGIC - Overwrite mode.
# MAGIC - A `replaceWhere` condition on `batch_id`.
# MAGIC - Partitioning by `batch_id`.
# MAGIC
# MAGIC This design allows an existing batch to be reprocessed without appending another copy of its records.
# MAGIC
# MAGIC A controlled repeat-execution test confirmed that the tested batch retained:
# MAGIC
# MAGIC - 1,409 prediction records.
# MAGIC - 1 monitoring record.
# MAGIC - 6 feature drift records.
# MAGIC
# MAGIC This demonstrates batch-level replacement under the tested conditions. It does not establish transactional atomicity across all three tables or guarantee correctness under concurrent executions.
# MAGIC
# MAGIC ### 2.5 Architecture Benefits
# MAGIC
# MAGIC **Reproducibility:** The registered model and saved preprocessing pipeline can be reused across scoring runs.
# MAGIC
# MAGIC **Traceability:** Batch identifiers and model versions help associate outputs with specific executions.
# MAGIC
# MAGIC **Operational Visibility:** Monitoring tables provide information about scoring completeness and risk distributions.
# MAGIC
# MAGIC **Data Quality:** Validation checks detect missing predictions and duplicate customer identifiers.
# MAGIC
# MAGIC **Model Monitoring:** Drift metrics provide an initial mechanism for identifying changes in input distributions.
# MAGIC
# MAGIC **Extensibility:** The architecture can be extended to ingest new customer data, support retention decisioning, and implement scheduled monitoring.
# MAGIC
# MAGIC ### 2.6 Current Architecture Limitations
# MAGIC
# MAGIC The current workflow processes a historical test dataset rather than continuously arriving production records.
# MAGIC
# MAGIC The preprocessing pipeline and registered prediction model are loaded as separate artifacts, which requires their compatibility to be maintained.
# MAGIC
# MAGIC Drift monitoring compares historical training and test datasets rather than observations from different production time periods.
# MAGIC
# MAGIC The table-update trigger remains paused, and the workflow has been validated through manually initiated executions.
# MAGIC
# MAGIC Future production improvements should include automated data ingestion, artifact version management, operational alerting, and monitoring against genuinely new customer populations.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Architecture Conclusion:** The implemented solution demonstrates a modular, reproducible MLOps workflow connecting registered machine learning models with batch inference, Delta Lake persistence, operational monitoring, and feature drift detection.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3. Batch Scoring Results and Customer Risk Segmentation
# MAGIC
# MAGIC ### 3.1 Business Objective
# MAGIC
# MAGIC The objective of batch scoring is to identify telecommunications customers who are at elevated risk of churn and provide actionable information for customer retention planning.
# MAGIC
# MAGIC Instead of treating all customers equally, the model assigns each customer a predicted churn probability.
# MAGIC
# MAGIC These probabilities can help the retention team prioritize outreach, allocate campaign resources, and identify customer segments requiring additional investigation.
# MAGIC
# MAGIC ### 3.2 Batch Scoring Summary
# MAGIC
# MAGIC The registered Logistic Regression model was applied to 1,409 customer records using a Databricks Workflow.
# MAGIC
# MAGIC | Metric | Result |
# MAGIC |---|---:|
# MAGIC | Customers scored | 1,409 |
# MAGIC | Unique customers | 1,409 |
# MAGIC | Missing churn probabilities | 0 |
# MAGIC | Average predicted churn probability | 25.99% |
# MAGIC | Customers flagged for retention | 400 |
# MAGIC | Percentage flagged | 28.39% |
# MAGIC | Registered model version | 1 |
# MAGIC | Optimized decision threshold | 0.40 |
# MAGIC
# MAGIC All 1,409 customer records received valid churn probabilities.
# MAGIC
# MAGIC The workflow also confirmed that each customer appeared only once within the validated scoring batch.
# MAGIC
# MAGIC ### 3.3 Customer Risk Segmentation
# MAGIC
# MAGIC Customers were classified into three risk categories based on their predicted churn probabilities.
# MAGIC
# MAGIC | Risk Category | Customers | Percentage |
# MAGIC |---|---:|---:|
# MAGIC | Low Risk | 1,009 | 71.61% |
# MAGIC | Medium Risk | 311 | 22.07% |
# MAGIC | High Risk | 89 | 6.32% |
# MAGIC | **Total** | **1,409** | **100%** |
# MAGIC
# MAGIC The Low Risk group represents the majority of the customer population.
# MAGIC
# MAGIC The Medium Risk group represents customers who may warrant additional analysis or targeted retention communications.
# MAGIC
# MAGIC The High Risk group represents customers with relatively elevated predicted churn risk and may be considered for prioritized retention assessment.
# MAGIC
# MAGIC These categories represent model-based risk estimates rather than confirmed future churn events.
# MAGIC
# MAGIC ### 3.4 Retention Targeting Using the Optimized Threshold
# MAGIC
# MAGIC The project uses a decision threshold of 0.40.
# MAGIC
# MAGIC Customers with predicted churn probabilities greater than or equal to this threshold are classified as candidates for retention outreach.
# MAGIC
# MAGIC **Decision rule:**
# MAGIC
# MAGIC - Predicted churn probability >= 0.40: Flag for potential retention outreach.
# MAGIC - Predicted churn probability < 0.40: Do not flag under the current decision rule.
# MAGIC
# MAGIC The workflow flagged 400 of the 1,409 customers, representing approximately 28.39% of the scoring population.
# MAGIC
# MAGIC The threshold was selected during the earlier model optimization stage to improve churn recall compared with the default 0.50 threshold.
# MAGIC
# MAGIC The threshold should ultimately be evaluated against actual retention costs, intervention effectiveness, and customer lifetime value.
# MAGIC
# MAGIC ### 3.5 Model Performance Context
# MAGIC
# MAGIC Earlier evaluation on the held-out test dataset produced the following results:
# MAGIC
# MAGIC | Metric | Threshold 0.50 | Threshold 0.40 |
# MAGIC |---|---:|---:|
# MAGIC | Accuracy | 80.06% | 79.21% |
# MAGIC | Precision (Churn) | 67.00% | 61.48% |
# MAGIC | Recall (Churn) | 52.48% | 62.92% |
# MAGIC | F1 Score | 58.86% | 62.19% |
# MAGIC
# MAGIC Reducing the classification threshold increased the proportion of actual churners detected, at the cost of lower precision and a small reduction in overall accuracy.
# MAGIC
# MAGIC This trade-off may be desirable when the business cost of failing to identify a likely churner exceeds the cost of contacting a customer who would otherwise remain.
# MAGIC
# MAGIC However, the financial advantage of the lower threshold has not yet been established.
# MAGIC
# MAGIC ### 3.6 Business Interpretation
# MAGIC
# MAGIC The batch scoring results provide a practical foundation for retention campaign planning.
# MAGIC
# MAGIC **Targeted outreach:** The retention team could prioritize the 400 flagged customers rather than contact all 1,409 customers.
# MAGIC
# MAGIC **Resource allocation:** Risk probabilities can be combined with customer value and campaign costs to determine which customers should receive higher-cost interventions.
# MAGIC
# MAGIC **Campaign measurement:** Future campaigns should record which customers were contacted, the intervention delivered, and subsequent churn outcomes.
# MAGIC
# MAGIC **Decision optimization:** Model predictions should be combined with financial assumptions and measured intervention effectiveness before estimating campaign profitability.
# MAGIC
# MAGIC ### 3.7 Limitations
# MAGIC
# MAGIC The current scoring batch uses historical test data rather than a newly arriving production customer population.
# MAGIC
# MAGIC The reported customer risk groups are based on model predictions and do not establish the causal effect of any retention intervention.
# MAGIC
# MAGIC No randomized retention experiment has yet been conducted to measure whether outreach reduces churn.
# MAGIC
# MAGIC Customer lifetime value, retention offer costs, and campaign response probabilities have not yet been incorporated into the targeting strategy.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Section 3 Conclusion:** The batch scoring workflow successfully generated customer-level churn probabilities and identified 400 potential retention candidates. The results demonstrate how machine learning predictions can support targeted decision-making, while further business-value analysis is required before recommending an operational retention campaign.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 4. Operational Monitoring and Data Quality Validation
# MAGIC
# MAGIC ### 4.1 Monitoring Objective
# MAGIC
# MAGIC A machine learning model must produce reliable predictions consistently before its outputs can support business decisions.
# MAGIC
# MAGIC Model accuracy alone is insufficient. A production-oriented MLOps workflow should also monitor data completeness, scoring coverage, customer uniqueness, and batch execution results.
# MAGIC
# MAGIC The Telco Customer Churn workflow implements operational monitoring to validate each scoring batch and persist quality metrics for subsequent inspection.
# MAGIC
# MAGIC ### 4.2 Monitoring Architecture
# MAGIC
# MAGIC The workflow reads the saved batch predictions from:
# MAGIC
# MAGIC `workspace.default.telco_10_prediction_history`
# MAGIC
# MAGIC It calculates aggregate operational metrics and stores them in:
# MAGIC
# MAGIC `workspace.default.telco_10_batch_monitoring`
# MAGIC
# MAGIC Each monitoring record includes:
# MAGIC
# MAGIC - Batch identifier
# MAGIC - Model version
# MAGIC - Number of records scored
# MAGIC - Number of unique customers
# MAGIC - Number of missing churn probabilities
# MAGIC - Average predicted churn probability
# MAGIC - Number of customers flagged for retention
# MAGIC - Customer risk distribution
# MAGIC - Monitoring timestamp
# MAGIC
# MAGIC This design supports batch-level traceability and historical monitoring.
# MAGIC
# MAGIC ### 4.3 Validated Batch Results
# MAGIC
# MAGIC The following results were verified during the controlled Databricks Workflow execution.
# MAGIC
# MAGIC | Monitoring Metric | Observed Value | Validation |
# MAGIC |---|---:|---|
# MAGIC | Total scored records | 1,409 | PASS |
# MAGIC | Unique customer IDs | 1,409 | PASS |
# MAGIC | Duplicate customer IDs | 0 | PASS |
# MAGIC | Missing churn probabilities | 0 | PASS |
# MAGIC | Average churn probability | 0.2599 | Recorded |
# MAGIC | Customers flagged for retention | 400 | Recorded |
# MAGIC | Low-risk customers | 1,009 | Recorded |
# MAGIC | Medium-risk customers | 311 | Recorded |
# MAGIC | High-risk customers | 89 | Recorded |
# MAGIC
# MAGIC The scoring output contained one record per customer, and all customer records received non-null churn probabilities.
# MAGIC
# MAGIC The risk category counts also reconcile to the total number of scored customers.
# MAGIC
# MAGIC ### 4.4 Data Quality Validation Rules
# MAGIC
# MAGIC **Rule 1 — Customer Uniqueness**
# MAGIC
# MAGIC The number of scored records must equal the number of distinct customer identifiers.
# MAGIC
# MAGIC Expected condition:
# MAGIC
# MAGIC `records == unique_customers`
# MAGIC
# MAGIC This check helps detect duplicate customer records within a scoring batch.
# MAGIC
# MAGIC **Rule 2 — Probability Completeness**
# MAGIC
# MAGIC Every scored customer must have a non-null churn probability.
# MAGIC
# MAGIC Expected condition:
# MAGIC
# MAGIC `null_probabilities == 0`
# MAGIC
# MAGIC This helps detect incomplete scoring outputs.
# MAGIC
# MAGIC **Rule 3 — Batch Record Reconciliation**
# MAGIC
# MAGIC The number of saved prediction records must equal the number of input customer records.
# MAGIC
# MAGIC Expected condition:
# MAGIC
# MAGIC `saved_records == input_records`
# MAGIC
# MAGIC This provides an additional check that the batch was persisted with the expected record count.
# MAGIC
# MAGIC **Rule 4 — Risk Distribution Reconciliation**
# MAGIC
# MAGIC The combined number of Low, Medium, and High Risk customers should equal the total number of scored records.
# MAGIC
# MAGIC Expected condition:
# MAGIC
# MAGIC `low_risk + medium_risk + high_risk == records`
# MAGIC
# MAGIC The observed counts satisfy this condition:
# MAGIC
# MAGIC `1009 + 311 + 89 = 1409`
# MAGIC
# MAGIC This reconciliation is supported by the observed monitoring output; it is not currently implemented as a separate explicit assertion in the monitoring code reviewed.
# MAGIC
# MAGIC ### 4.5 Batch-Level Persistence
# MAGIC
# MAGIC Monitoring results are written to a Delta Lake table using batch-specific replacement.
# MAGIC
# MAGIC The write operation uses:
# MAGIC
# MAGIC - Delta format
# MAGIC - Overwrite mode
# MAGIC - A `replaceWhere` condition on `batch_id`
# MAGIC - Partitioning by `batch_id`
# MAGIC
# MAGIC This allows the monitoring record for an existing batch to be replaced during reprocessing rather than duplicated through an append operation.
# MAGIC
# MAGIC ### 4.6 Controlled Retry Validation
# MAGIC
# MAGIC A repeat-execution test was conducted using the existing batch identifier:
# MAGIC
# MAGIC `telco_788040999362398`
# MAGIC
# MAGIC The workflow completed successfully.
# MAGIC
# MAGIC After reprocessing, the persisted output counts were:
# MAGIC
# MAGIC | Output | Expected | Observed | Result |
# MAGIC |---|---:|---:|---|
# MAGIC | Prediction history | 1,409 | 1,409 | PASS |
# MAGIC | Batch monitoring | 1 | 1 | PASS |
# MAGIC | Feature drift | 6 | 6 | PASS |
# MAGIC
# MAGIC These results demonstrate successful batch-level replacement for the tested execution.
# MAGIC
# MAGIC The test does not establish atomicity across all output tables or validate simultaneous executions using the same batch identifier.
# MAGIC
# MAGIC ### 4.7 Operational Monitoring Limitations
# MAGIC
# MAGIC The current monitoring implementation validates important aspects of batch completeness and customer uniqueness, but additional checks would strengthen production readiness.
# MAGIC
# MAGIC Recommended future checks include:
# MAGIC
# MAGIC - Churn probability values remain between 0 and 1.
# MAGIC - Required input columns are present and have valid data types.
# MAGIC - Risk category counts reconcile automatically.
# MAGIC - Incoming customer records meet schema and business validation rules.
# MAGIC - Batch execution failures generate notifications.
# MAGIC - Scoring duration and throughput are monitored over time.
# MAGIC - Model version and preprocessing artifact compatibility are verified.
# MAGIC - Monitoring results are compared against historical operational baselines.
# MAGIC
# MAGIC ### 4.8 Business Significance
# MAGIC
# MAGIC Operational monitoring improves confidence that model outputs are complete and suitable for further analysis.
# MAGIC
# MAGIC For a telecommunications retention team, incomplete predictions or duplicated customer records could lead to missed outreach opportunities, repeated customer contacts, and inaccurate campaign planning.
# MAGIC
# MAGIC By validating scoring outputs and maintaining a persistent monitoring history, the workflow provides a foundation for more reliable model-driven customer retention operations.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Section 4 Conclusion:** The Databricks Workflow successfully monitored 1,409 customer predictions, verified customer uniqueness and probability completeness, and demonstrated batch-level replacement during a controlled repeat-execution test. The implementation provides foundational operational quality controls, with further validation and alerting required for production use.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 5. Feature Drift Detection and Statistical Interpretation
# MAGIC
# MAGIC ### 5.1 Business Objective
# MAGIC
# MAGIC A machine learning model is trained using historical data. Over time, the characteristics of customers may change because of pricing adjustments, new service offerings, customer behaviour, or market conditions.
# MAGIC
# MAGIC These changes can cause the statistical distributions of incoming model features to differ from those observed during training.
# MAGIC
# MAGIC This phenomenon is known as **feature drift** or **data drift**.
# MAGIC
# MAGIC Monitoring feature drift helps identify changes that may require investigation, additional model evaluation, or retraining.
# MAGIC
# MAGIC Feature drift does not automatically mean that model performance has deteriorated.
# MAGIC
# MAGIC ### 5.2 Reference and Incoming Datasets
# MAGIC
# MAGIC The drift monitoring workflow compares two datasets stored in Unity Catalog.
# MAGIC
# MAGIC | Dataset | Unity Catalog Table | Records |
# MAGIC |---|---|---:|
# MAGIC | Reference dataset | `workspace.default.telco_train_raw` | 5,634 |
# MAGIC | Incoming scoring dataset | `workspace.default.telco_test_raw` | 1,409 |
# MAGIC
# MAGIC The reference dataset represents the historical training population.
# MAGIC
# MAGIC The incoming dataset represents the held-out test population used to simulate a new scoring batch.
# MAGIC
# MAGIC Because both datasets originate from the same historical Telco dataset, the comparison validates the drift-monitoring implementation rather than demonstrating real-world temporal drift.
# MAGIC
# MAGIC ### 5.3 Drift Detection Methods
# MAGIC
# MAGIC Two statistical measures are implemented.
# MAGIC
# MAGIC #### A. Population Stability Index (PSI)
# MAGIC
# MAGIC Population Stability Index measures changes between two distributions.
# MAGIC
# MAGIC In this project, PSI is used for numerical customer attributes.
# MAGIC
# MAGIC The monitored numerical features are:
# MAGIC
# MAGIC - tenure
# MAGIC - MonthlyCharges
# MAGIC - TotalCharges
# MAGIC
# MAGIC PSI is calculated by comparing the proportions of observations falling into corresponding distribution bins.
# MAGIC
# MAGIC The general formula is:
# MAGIC
# MAGIC PSI = SUM[(Actual_i - Expected_i) * ln(Actual_i / Expected_i)]
# MAGIC
# MAGIC Where:
# MAGIC
# MAGIC - Expected_i is the proportion in bin i for the reference dataset.
# MAGIC - Actual_i is the proportion in the corresponding bin for the incoming dataset.
# MAGIC - ln represents the natural logarithm.
# MAGIC
# MAGIC A PSI value close to zero indicates that the two binned distributions are similar.
# MAGIC
# MAGIC Larger PSI values indicate greater distributional differences.
# MAGIC
# MAGIC The interpretation depends on the binning strategy, treatment of missing values, and monitoring thresholds.
# MAGIC
# MAGIC #### B. Total Variation Distance (TVD)
# MAGIC
# MAGIC Total Variation Distance measures the difference between two probability distributions.
# MAGIC
# MAGIC In this project, TVD is used for categorical customer attributes.
# MAGIC
# MAGIC The monitored categorical features are:
# MAGIC
# MAGIC - Contract
# MAGIC - InternetService
# MAGIC - PaymentMethod
# MAGIC
# MAGIC The general formula is:
# MAGIC
# MAGIC TVD = 0.5 * SUM[ABS(P_i - Q_i)]
# MAGIC
# MAGIC Where:
# MAGIC
# MAGIC - P_i is the reference proportion for category i.
# MAGIC - Q_i is the incoming proportion for the same category.
# MAGIC - ABS represents absolute value.
# MAGIC
# MAGIC TVD ranges from 0 to 1.
# MAGIC
# MAGIC A value of 0 indicates identical categorical distributions.
# MAGIC
# MAGIC Larger values indicate greater differences in category proportions.
# MAGIC
# MAGIC ### 5.4 Configured Drift Thresholds
# MAGIC
# MAGIC The project applies the following monitoring rules.
# MAGIC
# MAGIC **Numerical Features — PSI**
# MAGIC
# MAGIC | PSI Value | Classification | Interpretation |
# MAGIC |---|---|---|
# MAGIC | Below 0.10 | NORMAL | Small observed distributional difference |
# MAGIC | 0.10 to below 0.25 | REVIEW | Distributional change warrants investigation |
# MAGIC | 0.25 or higher | ALERT | Larger distributional change warrants attention |
# MAGIC
# MAGIC **Categorical Features — TVD**
# MAGIC
# MAGIC | TVD Value | Classification | Interpretation |
# MAGIC |---|---|---|
# MAGIC | Below 0.10 | NORMAL | Small observed distributional difference |
# MAGIC | 0.10 or higher | REVIEW | Distributional change warrants investigation |
# MAGIC
# MAGIC These are project-specific operational thresholds, not universal statistical significance tests.
# MAGIC
# MAGIC ### 5.5 Observed Feature Drift Results
# MAGIC
# MAGIC The workflow monitored six customer features.
# MAGIC
# MAGIC | Feature | Metric | Drift Score | Status |
# MAGIC |---|---|---:|---|
# MAGIC | tenure | PSI | 0.016046 | NORMAL |
# MAGIC | MonthlyCharges | PSI | 0.007876 | NORMAL |
# MAGIC | TotalCharges | PSI | 0.009381 | NORMAL |
# MAGIC | Contract | TVD | 0.019799 | NORMAL |
# MAGIC | InternetService | TVD | 0.012720 | NORMAL |
# MAGIC | PaymentMethod | TVD | 0.011989 | NORMAL |
# MAGIC
# MAGIC All six monitored features were classified as NORMAL under the configured thresholds.
# MAGIC
# MAGIC No monitored feature exceeded its review threshold.
# MAGIC
# MAGIC ### 5.6 Statistical Interpretation
# MAGIC
# MAGIC **Numerical features**
# MAGIC
# MAGIC The PSI values for tenure, MonthlyCharges, and TotalCharges are all below 0.10.
# MAGIC
# MAGIC This indicates relatively small differences between the binned numerical feature distributions in the reference and scoring datasets.
# MAGIC
# MAGIC **Categorical features**
# MAGIC
# MAGIC The TVD values for Contract, InternetService, and PaymentMethod are all below 0.10.
# MAGIC
# MAGIC This indicates that the observed category proportions differ only modestly between the two datasets.
# MAGIC
# MAGIC Contract has the highest observed TVD among the monitored categorical features, at approximately 0.0198.
# MAGIC
# MAGIC However, this remains below the configured review threshold.
# MAGIC
# MAGIC **Overall interpretation**
# MAGIC
# MAGIC The observed drift scores suggest that the historical training and test populations have similar distributions for the six monitored features.
# MAGIC
# MAGIC This result is consistent with the datasets being partitions of the same historical source.
# MAGIC
# MAGIC It does not establish that future production data will have similar distributions or that predictive performance will remain stable.
# MAGIC
# MAGIC ### 5.7 Missing-Value Considerations
# MAGIC
# MAGIC The drift calculations use available non-null observations for each monitored feature.
# MAGIC
# MAGIC For TotalCharges:
# MAGIC
# MAGIC | Dataset | Total Records | Non-Null Records |
# MAGIC |---|---:|---:|
# MAGIC | Reference | 5,634 | 5,627 |
# MAGIC | Incoming | 1,409 | 1,405 |
# MAGIC
# MAGIC Consequently, the TotalCharges PSI calculation is based on fewer observations than the full dataset sizes.
# MAGIC
# MAGIC A production monitoring solution should separately track missing-value rates, because changes in missingness may indicate data quality problems even when the observed non-null distributions remain similar.
# MAGIC
# MAGIC ### 5.8 Drift Monitoring Persistence
# MAGIC
# MAGIC Drift results are saved in the Delta table:
# MAGIC
# MAGIC `workspace.default.telco_10_feature_drift`
# MAGIC
# MAGIC Each saved record includes:
# MAGIC
# MAGIC - Batch identifier
# MAGIC - Feature name
# MAGIC - Drift metric
# MAGIC - Calculated drift score
# MAGIC - Monitoring status
# MAGIC - Reference non-null count
# MAGIC - Incoming non-null count
# MAGIC - Registered model version
# MAGIC - Monitoring timestamp
# MAGIC
# MAGIC The table uses batch-specific replacement through Delta Lake `overwrite` and `replaceWhere` operations.
# MAGIC
# MAGIC This supports historical inspection and controlled reprocessing of individual batches.
# MAGIC
# MAGIC ### 5.9 Recommended Production Improvements
# MAGIC
# MAGIC The following enhancements would strengthen the drift-monitoring system:
# MAGIC
# MAGIC 1. Compare genuinely new scoring data against a clearly defined reference population.
# MAGIC 2. Monitor missing-value rates independently of distribution drift.
# MAGIC 3. Monitor additional categorical and numerical model features.
# MAGIC 4. Add alerts when PSI or TVD exceeds configured thresholds.
# MAGIC 5. Track drift trends across multiple scoring periods.
# MAGIC 6. Monitor prediction-score distributions.
# MAGIC 7. Evaluate actual churn outcomes when labels become available.
# MAGIC 8. Investigate changes in model precision, recall, calibration, and business performance before making retraining decisions.
# MAGIC
# MAGIC ### 5.10 Business Implications
# MAGIC
# MAGIC Feature drift monitoring provides an early-warning mechanism for changes in the customer population.
# MAGIC
# MAGIC For a telecommunications company, significant changes in contract types, pricing, or service adoption may affect how well a historical churn model generalizes to new customers.
# MAGIC
# MAGIC By monitoring these changes, data science teams can prioritize investigations and determine whether additional validation or model updates are necessary.
# MAGIC
# MAGIC However, drift alerts should be interpreted alongside model performance, data quality, and business context rather than treated as automatic evidence of model failure.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Section 5 Conclusion:** The Databricks workflow successfully calculated and persisted PSI and TVD metrics for six customer features. All six were classified as NORMAL under the project's monitoring thresholds. These results validate the feature drift monitoring implementation on historical data and establish a foundation for future monitoring against genuinely new customer populations.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 6. Workflow Reliability, Retry Safety and Production Readiness
# MAGIC
# MAGIC ### 6.1 Objective
# MAGIC
# MAGIC A reliable machine learning workflow must do more than generate accurate predictions.
# MAGIC
# MAGIC It should also support:
# MAGIC
# MAGIC - Repeatable batch execution.
# MAGIC - Traceable processing runs.
# MAGIC - Safe reprocessing of previously scored batches.
# MAGIC - Persistent monitoring and prediction outputs.
# MAGIC - Recovery from temporary execution failures.
# MAGIC - Operational visibility into workflow results.
# MAGIC
# MAGIC The Telco Customer Churn project implements several of these capabilities using Databricks Workflows, Delta Lake, MLflow, and Unity Catalog.
# MAGIC
# MAGIC ### 6.2 Databricks Workflow Configuration
# MAGIC
# MAGIC | Configuration | Value |
# MAGIC |---|---|
# MAGIC | Job name | Telco_Churn_Batch_Scoring_Monitoring |
# MAGIC | Job ID | 738907313537928 |
# MAGIC | Task name | telco_batch_scoring |
# MAGIC | Compute | Databricks Serverless |
# MAGIC | Environment | telco_batch_scoring_environment |
# MAGIC | Environment version | 6 |
# MAGIC | Notebook | 10_Telco_Batch_Scoring_Automation_Monitoring_and_Drift |
# MAGIC | Model | Logistic Regression |
# MAGIC | Registered model version | 1 |
# MAGIC | Maximum automatic retries | 3 |
# MAGIC | Retry interval | Immediate |
# MAGIC | Table-update trigger | Paused |
# MAGIC
# MAGIC The workflow executes batch inference, operational monitoring, and feature drift detection within a single notebook task.
# MAGIC
# MAGIC The current implementation has been validated through manually initiated Job runs.
# MAGIC
# MAGIC ### 6.3 Workflow Execution History
# MAGIC
# MAGIC Three successful Databricks Job executions were recorded on November 17, 2026.
# MAGIC
# MAGIC | Execution | Job Run ID | Duration | Result |
# MAGIC |---|---|---|---|
# MAGIC | Run 1 | 420241103909817 | 2m 50s | SUCCESS |
# MAGIC | Run 2 | 788040999362398 | 2m 56s | SUCCESS |
# MAGIC | Run 3 | 403340602056524 | 3m 04s | SUCCESS |
# MAGIC
# MAGIC All three executions completed successfully.
# MAGIC
# MAGIC These results demonstrate that the workflow can execute repeatedly on the configured serverless environment.
# MAGIC
# MAGIC They do not establish a production service-level agreement or guarantee future execution reliability.
# MAGIC
# MAGIC ### 6.4 Dynamic Batch Identification
# MAGIC
# MAGIC The Databricks Workflow uses the task parameter:
# MAGIC
# MAGIC `batch_id = telco_{{job.run_id}}`
# MAGIC
# MAGIC Databricks resolves the dynamic reference to the Job run identifier.
# MAGIC
# MAGIC For example, Job Run 2 used:
# MAGIC
# MAGIC `telco_788040999362398`
# MAGIC
# MAGIC A new Job run receives a different identifier.
# MAGIC
# MAGIC Automatic task retries within the same Job run can reuse the same identifier, allowing the workflow to target the same batch during reprocessing.
# MAGIC
# MAGIC When the notebook is executed interactively without a supplied batch identifier, the notebook can generate an identifier using a UTC timestamp and a UUID suffix.
# MAGIC
# MAGIC ### 6.5 Batch-Specific Delta Lake Replacement
# MAGIC
# MAGIC The workflow persists results in three Delta Lake tables:
# MAGIC
# MAGIC | Table | Purpose |
# MAGIC |---|---|
# MAGIC | `workspace.default.telco_10_prediction_history` | Customer-level churn predictions |
# MAGIC | `workspace.default.telco_10_batch_monitoring` | Operational monitoring metrics |
# MAGIC | `workspace.default.telco_10_feature_drift` | Feature drift measurements |
# MAGIC
# MAGIC The implementation uses Delta Lake overwrite operations with a batch-specific `replaceWhere` predicate.
# MAGIC
# MAGIC Conceptually, the write operation follows this pattern:
# MAGIC
# MAGIC `overwrite WHERE batch_id = current_batch_id`
# MAGIC
# MAGIC This allows records belonging to a previously processed batch to be replaced without overwriting unrelated batches.
# MAGIC
# MAGIC Each table is updated independently.
# MAGIC
# MAGIC ### 6.6 Controlled Repeat-Execution Test
# MAGIC
# MAGIC To test batch-level replacement, the workflow was manually executed again using an existing batch identifier.
# MAGIC
# MAGIC **Original batch identifier:**
# MAGIC
# MAGIC `telco_788040999362398`
# MAGIC
# MAGIC **Controlled reprocessing Job run:**
# MAGIC
# MAGIC `403340602056524`
# MAGIC
# MAGIC The third Job run was configured temporarily to reuse the original batch identifier.
# MAGIC
# MAGIC After successful execution, the following records were verified:
# MAGIC
# MAGIC | Output Table | Expected Records | Actual Records | Result |
# MAGIC |---|---:|---:|---|
# MAGIC | Prediction history | 1,409 | 1,409 | PASS |
# MAGIC | Batch monitoring | 1 | 1 | PASS |
# MAGIC | Feature drift | 6 | 6 | PASS |
# MAGIC
# MAGIC The observed results confirm that the tested batch was reprocessed without accumulating duplicate records in the three output tables.
# MAGIC
# MAGIC This provides evidence of batch-level idempotent replacement under the tested conditions.
# MAGIC
# MAGIC ### 6.7 Automatic Retry Configuration
# MAGIC
# MAGIC The Databricks task is configured to allow up to three automatic retries after a failure.
# MAGIC
# MAGIC Automatic retries are useful for recovering from transient problems such as temporary infrastructure or service interruptions.
# MAGIC
# MAGIC However, the controlled repeat-execution test was performed through a separate manually initiated Job run.
# MAGIC
# MAGIC An actual automatic retry following a task failure was not observed or validated during this test.
# MAGIC
# MAGIC Therefore, automatic retry behaviour remains a configured capability rather than an experimentally verified recovery scenario.
# MAGIC
# MAGIC ### 6.8 Workflow Configuration Restored
# MAGIC
# MAGIC After completing the controlled repeat-execution test, the temporary batch identifier was removed.
# MAGIC
# MAGIC The task parameter was restored to:
# MAGIC
# MAGIC `telco_{{job.run_id}}`
# MAGIC
# MAGIC The updated task configuration was saved.
# MAGIC
# MAGIC This ensures that future Job runs can receive their own dynamically generated batch identifiers.
# MAGIC
# MAGIC ### 6.9 Reliability Limitations
# MAGIC
# MAGIC Although the workflow completed successfully, several limitations remain.
# MAGIC
# MAGIC **Independent table writes**
# MAGIC
# MAGIC The prediction, monitoring, and drift tables are written separately.
# MAGIC
# MAGIC A failure between writes could leave some outputs updated while others remain unchanged.
# MAGIC
# MAGIC **Concurrent execution**
# MAGIC
# MAGIC The workflow has not been tested with simultaneous executions targeting the same batch identifier.
# MAGIC
# MAGIC **Failure recovery**
# MAGIC
# MAGIC No controlled task failure was introduced to validate automatic retry recovery.
# MAGIC
# MAGIC **Historical scoring data**
# MAGIC
# MAGIC The workflow currently processes a historical test dataset rather than a continuously updated production source.
# MAGIC
# MAGIC **Paused trigger**
# MAGIC
# MAGIC The configured table-update trigger remains paused.
# MAGIC
# MAGIC Consequently, the workflow is not currently operating as an automatically triggered production pipeline.
# MAGIC
# MAGIC ### 6.10 Production Readiness Roadmap
# MAGIC
# MAGIC The following improvements are recommended before production deployment.
# MAGIC
# MAGIC | Priority | Improvement | Expected Benefit |
# MAGIC |---|---|---|
# MAGIC | High | Integrate new customer data ingestion | Enable scoring of current customers |
# MAGIC | High | Activate an appropriate trigger or schedule | Automate batch execution |
# MAGIC | High | Implement failure notifications | Improve incident response |
# MAGIC | High | Add explicit data quality assertions | Detect invalid scoring outputs |
# MAGIC | High | Test automatic retry recovery | Validate failure handling |
# MAGIC | Medium | Implement batch completion status tracking | Identify partial executions |
# MAGIC | Medium | Add concurrent-run controls | Reduce conflicting batch writes |
# MAGIC | Medium | Monitor execution duration and cost | Improve operational efficiency |
# MAGIC | Medium | Track actual churn outcomes | Measure ongoing model performance |
# MAGIC | Medium | Monitor model and preprocessing versions | Strengthen reproducibility |
# MAGIC | Future | Integrate retention campaign outcomes | Evaluate business effectiveness |
# MAGIC
# MAGIC ### 6.11 Overall Reliability Assessment
# MAGIC
# MAGIC The project demonstrates a functional and repeatable batch-scoring workflow.
# MAGIC
# MAGIC Its strongest validated reliability capabilities are:
# MAGIC
# MAGIC 1. Successful execution on Databricks Serverless.
# MAGIC 2. Persistent Delta Lake outputs.
# MAGIC 3. Batch-level traceability using identifiers.
# MAGIC 4. Controlled reprocessing without duplicate accumulation.
# MAGIC 5. Operational monitoring and feature drift persistence.
# MAGIC
# MAGIC Additional failure testing, automated triggering, alerting, and recovery controls are required before the solution can be considered production-ready.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Section 6 Conclusion:** Three successful Databricks Job runs and a controlled same-batch reprocessing test demonstrate repeatable execution and batch-level replacement. The workflow provides a strong MLOps prototype, while automatic failure recovery, concurrent execution safety, and production scheduling remain areas for further validation.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 7. Business Impact, Project Limitations and Production Roadmap
# MAGIC
# MAGIC ### 7.1 Business Objective
# MAGIC
# MAGIC The primary business objective of the Telco Customer Churn project is to support customer retention by identifying subscribers with elevated predicted churn risk.
# MAGIC
# MAGIC The implemented machine learning workflow enables a telecommunications company to:
# MAGIC
# MAGIC - Estimate customer-level churn probabilities.
# MAGIC - Identify potential candidates for retention outreach.
# MAGIC - Prioritize customers based on predicted churn risk.
# MAGIC - Maintain historical records of scoring results.
# MAGIC - Monitor operational data quality and feature distributions.
# MAGIC
# MAGIC These capabilities provide a foundation for data-driven customer retention planning.
# MAGIC
# MAGIC ### 7.2 Potential Business Impact
# MAGIC
# MAGIC The validated scoring batch contained 1,409 customers.
# MAGIC
# MAGIC Using the selected classification threshold of 0.40, the workflow flagged 400 customers for potential retention outreach.
# MAGIC
# MAGIC | Business Metric | Observed Result |
# MAGIC |---|---:|
# MAGIC | Customers evaluated | 1,409 |
# MAGIC | Customers flagged | 400 |
# MAGIC | Percentage flagged | 28.39% |
# MAGIC | Customers not flagged | 1,009 |
# MAGIC | Average predicted churn probability | 25.99% |
# MAGIC
# MAGIC Rather than treating all customers as equally likely to churn, the model provides a risk-based approach to customer prioritization.
# MAGIC
# MAGIC A retention team could use these predictions to focus its initial investigation or outreach planning on a smaller subset of customers.
# MAGIC
# MAGIC However, flagging a customer does not establish that contacting them will prevent churn.
# MAGIC
# MAGIC ### 7.3 From Predictive Analytics to Business Decisions
# MAGIC
# MAGIC A churn prediction model estimates the probability that a customer will leave.
# MAGIC
# MAGIC A retention decision requires additional information, including:
# MAGIC
# MAGIC 1. Customer lifetime value.
# MAGIC 2. Cost of retention outreach.
# MAGIC 3. Cost of incentives or discounts.
# MAGIC 4. Probability that an intervention will prevent churn.
# MAGIC 5. Available campaign capacity.
# MAGIC 6. Customer eligibility and communication preferences.
# MAGIC
# MAGIC A customer with a high predicted churn probability is not necessarily the most profitable customer to target.
# MAGIC
# MAGIC For example, a customer with moderate churn risk but high customer lifetime value and a strong likelihood of responding to an offer may represent a better retention opportunity.
# MAGIC
# MAGIC A future decision framework should therefore combine churn probabilities with customer value and estimated intervention effectiveness.
# MAGIC
# MAGIC ### 7.4 Illustrative Retention Economics
# MAGIC
# MAGIC The following assumptions are provided only to illustrate how business value could be evaluated.
# MAGIC
# MAGIC They are not observed financial results from the Telco dataset.
# MAGIC
# MAGIC | Illustrative Parameter | Assumed Value |
# MAGIC |---|---:|
# MAGIC | Customers targeted | 400 |
# MAGIC | Customer value at risk | $500 |
# MAGIC | Intervention cost per customer | $50 |
# MAGIC | Assumed probability intervention prevents churn | 20% |
# MAGIC
# MAGIC Under these simplified assumptions, the expected net benefit for a customer can be expressed as:
# MAGIC
# MAGIC Expected Net Benefit = P(Churn) * P(Prevent Churn | Intervention, Churn) * Value Preserved - Intervention Cost
# MAGIC
# MAGIC For a customer with predicted churn probability of 0.60:
# MAGIC
# MAGIC Expected Net Benefit = 0.60 * 0.20 * $500 - $50
# MAGIC
# MAGIC Expected Net Benefit = $10
# MAGIC
# MAGIC This example illustrates the importance of considering both churn risk and intervention effectiveness.
# MAGIC
# MAGIC It assumes the $500 represents net economic value preserved, the intervention costs $50, and the assumed 20% prevention probability is applicable to that customer.
# MAGIC
# MAGIC Actual business value would require validated financial inputs and credible estimates of the causal effect of retention interventions.
# MAGIC
# MAGIC ### 7.5 Current Project Limitations
# MAGIC
# MAGIC **Historical data**
# MAGIC
# MAGIC The scoring workflow processes the historical Telco test dataset rather than newly arriving customer records.
# MAGIC
# MAGIC **Limited feature drift validation**
# MAGIC
# MAGIC Drift monitoring compares historical training and test partitions from the same source dataset.
# MAGIC
# MAGIC This does not demonstrate temporal drift monitoring in a live telecommunications environment.
# MAGIC
# MAGIC **No retention campaign integration**
# MAGIC
# MAGIC The workflow generates retention candidates but does not execute customer outreach.
# MAGIC
# MAGIC **No observed intervention outcomes**
# MAGIC
# MAGIC The project does not contain evidence that a retention offer prevented churn.
# MAGIC
# MAGIC **No realized financial impact**
# MAGIC
# MAGIC Revenue saved, campaign profitability, and return on investment have not been measured.
# MAGIC
# MAGIC **No automated production trigger**
# MAGIC
# MAGIC The Databricks table-update trigger remains paused.
# MAGIC
# MAGIC **Incomplete failure-recovery validation**
# MAGIC
# MAGIC The controlled reprocessing test validated batch-level replacement but did not exercise automatic task retry following a failure.
# MAGIC
# MAGIC **Model monitoring scope**
# MAGIC
# MAGIC Operational quality and selected feature distributions are monitored, but ongoing model performance cannot be established without new, reliable churn outcome labels.
# MAGIC
# MAGIC ### 7.6 Production Roadmap
# MAGIC
# MAGIC The following roadmap describes how the prototype could evolve into a production-oriented churn prediction system.
# MAGIC
# MAGIC #### Phase 1 — Production Data Integration
# MAGIC
# MAGIC - Connect to an operational customer data source.
# MAGIC - Establish a repeatable ingestion process.
# MAGIC - Validate schema, customer identifiers, and required features.
# MAGIC - Implement incremental batch processing.
# MAGIC - Define data freshness and quality requirements.
# MAGIC
# MAGIC **Expected outcome:** Reliable ingestion of new customer data for scoring.
# MAGIC
# MAGIC #### Phase 2 — Workflow Automation and Reliability
# MAGIC
# MAGIC - Enable an appropriate Databricks Workflow trigger or schedule.
# MAGIC - Configure operational alerts for failed tasks.
# MAGIC - Test automatic retry and recovery behaviour.
# MAGIC - Introduce batch completion tracking.
# MAGIC - Prevent conflicting concurrent executions.
# MAGIC - Monitor runtime, resource consumption, and cost.
# MAGIC
# MAGIC **Expected outcome:** A more reliable and observable automated scoring process.
# MAGIC
# MAGIC #### Phase 3 — Advanced Model Monitoring
# MAGIC
# MAGIC - Monitor feature drift across genuine production time periods.
# MAGIC - Track prediction probability distributions.
# MAGIC - Monitor missing values and schema changes.
# MAGIC - Collect delayed actual churn labels.
# MAGIC - Evaluate precision, recall, ROC-AUC, and calibration over time.
# MAGIC - Define investigation and retraining criteria.
# MAGIC
# MAGIC **Expected outcome:** Improved visibility into model health and changing customer populations.
# MAGIC
# MAGIC #### Phase 4 — Business Decision Optimization
# MAGIC
# MAGIC - Incorporate customer lifetime value.
# MAGIC - Estimate intervention costs.
# MAGIC - Analyze decile lift and cumulative gains.
# MAGIC - Evaluate alternative targeting thresholds.
# MAGIC - Design retention campaign experiments.
# MAGIC - Estimate treatment effectiveness using controlled comparisons.
# MAGIC
# MAGIC **Expected outcome:** Retention targeting informed by expected business value rather than churn probability alone.
# MAGIC
# MAGIC #### Phase 5 — Governance and Continuous Improvement
# MAGIC
# MAGIC - Maintain versioned model and preprocessing artifacts.
# MAGIC - Document model assumptions and limitations.
# MAGIC - Implement appropriate access controls.
# MAGIC - Establish model approval and deployment processes.
# MAGIC - Maintain monitoring and incident-response procedures.
# MAGIC - Evaluate fairness, privacy, and customer communication requirements.
# MAGIC
# MAGIC **Expected outcome:** A more accountable, maintainable, and auditable MLOps system.
# MAGIC
# MAGIC ### 7.7 Recommended Next Analytical Enhancements
# MAGIC
# MAGIC The following enhancements are planned for subsequent project notebooks.
# MAGIC
# MAGIC | Notebook | Planned Enhancement | Business Relevance |
# MAGIC |---|---|---|
# MAGIC | 11 | Cost-sensitive threshold optimization | Evaluate targeting decisions using economic assumptions |
# MAGIC | 12 | Decile lift and cumulative gains | Measure how effectively the model ranks churners |
# MAGIC | 13 | Model explainability | Understand feature contributions to predictions |
# MAGIC | 14 | Retention strategy simulation | Compare hypothetical retention targeting strategies |
# MAGIC
# MAGIC These notebooks will extend the project beyond predictive modeling toward decision-support analytics.
# MAGIC
# MAGIC ### 7.8 Portfolio and Interview Relevance
# MAGIC
# MAGIC The project demonstrates practical experience across the machine learning lifecycle.
# MAGIC
# MAGIC **Data Science**
# MAGIC
# MAGIC Feature engineering, classification, model evaluation, threshold optimization, and statistical drift monitoring.
# MAGIC
# MAGIC **Data Engineering**
# MAGIC
# MAGIC Apache Spark transformations, Unity Catalog tables, Delta Lake persistence, and batch data processing.
# MAGIC
# MAGIC **MLOps**
# MAGIC
# MAGIC MLflow model registration, model reuse, Databricks Workflow execution, batch traceability, and operational monitoring.
# MAGIC
# MAGIC **Business Analytics**
# MAGIC
# MAGIC Customer risk segmentation, retention prioritization, and the design of cost-sensitive decision frameworks.
# MAGIC
# MAGIC **Production Engineering**
# MAGIC
# MAGIC Retry configuration, controlled reprocessing, batch-level replacement, and identification of reliability limitations.
# MAGIC
# MAGIC Together, these capabilities demonstrate how machine learning models can be integrated into operational workflows while maintaining appropriate validation and governance controls.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Section 7 Conclusion:** The Telco Customer Churn project demonstrates how predictive analytics and MLOps can support customer retention prioritization. The workflow successfully identified 400 potential retention candidates and implemented foundational operational monitoring. Realized financial impact and production reliability remain unverified. The proposed roadmap outlines the additional data, experimentation, automation, and governance capabilities required for production deployment.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 8. Final Conclusions, Key Achievements and GitHub Portfolio Summary
# MAGIC
# MAGIC ### 8.1 Project Overview
# MAGIC
# MAGIC The Telco Customer Churn Prediction project demonstrates an end-to-end machine learning and MLOps implementation using Databricks, Apache Spark, MLflow, Unity Catalog, and Delta Lake.
# MAGIC
# MAGIC The solution covers data exploration, preprocessing, feature engineering, model development, model evaluation, threshold optimization, model registration, batch inference, operational monitoring, and feature drift detection.
# MAGIC
# MAGIC The final implementation provides a repeatable batch scoring workflow that generates customer churn probabilities and persists monitoring results for further analysis.
# MAGIC
# MAGIC ### 8.2 Key Technical Achievements
# MAGIC
# MAGIC **1. End-to-End Machine Learning Pipeline**
# MAGIC
# MAGIC Developed a reusable Spark ML preprocessing pipeline that transforms raw telecommunications customer attributes into a 60-dimensional feature vector.
# MAGIC
# MAGIC Trained and evaluated classification models, including Logistic Regression, Gradient-Boosted Trees, and Random Forest.
# MAGIC
# MAGIC Selected Logistic Regression based on the validation results.
# MAGIC
# MAGIC **2. Model Evaluation and Threshold Optimization**
# MAGIC
# MAGIC Evaluated the selected model using a held-out test dataset.
# MAGIC
# MAGIC At the default threshold of 0.50, the model achieved:
# MAGIC
# MAGIC - Accuracy: 80.06%
# MAGIC - Precision: 67.00%
# MAGIC - Recall: 52.48%
# MAGIC - F1 Score: 58.86%
# MAGIC - ROC-AUC: 0.8385
# MAGIC
# MAGIC Reducing the classification threshold to 0.40 improved recall to 62.92%, while precision decreased to 61.48%.
# MAGIC
# MAGIC This demonstrated the importance of aligning classification thresholds with business priorities rather than relying exclusively on accuracy.
# MAGIC
# MAGIC **3. MLflow and Unity Catalog Integration**
# MAGIC
# MAGIC Registered the selected Logistic Regression model in Unity Catalog through MLflow.
# MAGIC
# MAGIC Implemented model retrieval using a versioned model URI.
# MAGIC
# MAGIC Separated reusable feature preprocessing from the registered prediction model.
# MAGIC
# MAGIC **4. Automated Batch Scoring Workflow**
# MAGIC
# MAGIC Implemented a Databricks Workflow using serverless compute.
# MAGIC
# MAGIC Successfully scored 1,409 customer records and generated churn probabilities for every customer.
# MAGIC
# MAGIC Identified 400 customers as potential retention candidates using the selected threshold.
# MAGIC
# MAGIC **5. Operational Monitoring**
# MAGIC
# MAGIC Implemented batch-level monitoring for:
# MAGIC
# MAGIC - Scoring completeness
# MAGIC - Customer uniqueness
# MAGIC - Missing predictions
# MAGIC - Average churn probability
# MAGIC - Retention flag counts
# MAGIC - Customer risk distributions
# MAGIC
# MAGIC Persisted monitoring results in a Delta Lake table.
# MAGIC
# MAGIC **6. Statistical Feature Drift Monitoring**
# MAGIC
# MAGIC Implemented numerical feature drift monitoring using Population Stability Index (PSI).
# MAGIC
# MAGIC Implemented categorical feature drift monitoring using Total Variation Distance (TVD).
# MAGIC
# MAGIC Monitored six customer features and persisted their drift results.
# MAGIC
# MAGIC All six features were classified as NORMAL under the configured thresholds.
# MAGIC
# MAGIC **7. Workflow Reliability and Batch Reprocessing**
# MAGIC
# MAGIC Successfully executed the Databricks Workflow three times.
# MAGIC
# MAGIC Implemented dynamic batch identification using Databricks Job run IDs.
# MAGIC
# MAGIC Validated controlled same-batch reprocessing using Delta Lake batch-specific replacement.
# MAGIC
# MAGIC Verified that the tested batch retained the expected number of records without accumulating duplicates.
# MAGIC
# MAGIC ### 8.3 Consolidated Project Results
# MAGIC
# MAGIC | Category | Metric | Result |
# MAGIC |---|---|---:|
# MAGIC | Data | Original dataset records | 7,043 |
# MAGIC | Feature Engineering | Feature vector dimensions | 60 |
# MAGIC | Model Evaluation | Test ROC-AUC | 0.8385 |
# MAGIC | Model Evaluation | Recall at threshold 0.40 | 62.92% |
# MAGIC | Batch Inference | Customers scored | 1,409 |
# MAGIC | Batch Inference | Customers flagged | 400 |
# MAGIC | Data Quality | Missing churn probabilities | 0 |
# MAGIC | Data Quality | Duplicate customers | 0 |
# MAGIC | Drift Monitoring | Features monitored | 6 |
# MAGIC | Drift Monitoring | Features exceeding thresholds | 0 |
# MAGIC | Workflow | Successful Job runs | 3 |
# MAGIC | Reliability | Controlled same-batch reprocessing | Successful |
# MAGIC
# MAGIC ### 8.4 Business Value
# MAGIC
# MAGIC The project demonstrates how customer churn predictions can support more focused retention planning.
# MAGIC
# MAGIC Instead of treating every customer as equally likely to churn, the model provides customer-level risk estimates that can be used to prioritize investigation and outreach.
# MAGIC
# MAGIC The workflow also introduces operational controls to help ensure that scoring outputs are complete, traceable, and suitable for downstream analysis.
# MAGIC
# MAGIC However, the project has not measured actual retention campaign effectiveness, prevented churn, or realized financial savings.
# MAGIC
# MAGIC Future work will incorporate cost-sensitive optimization, customer value assumptions, and retention strategy simulations.
# MAGIC
# MAGIC ### 8.5 Current Deployment Status
# MAGIC
# MAGIC The project is a validated MLOps prototype rather than a live production deployment.
# MAGIC
# MAGIC The workflow has demonstrated:
# MAGIC
# MAGIC - Repeatable execution on Databricks Serverless.
# MAGIC - Registered model loading and batch inference.
# MAGIC - Persistent Delta Lake outputs.
# MAGIC - Operational data quality monitoring.
# MAGIC - Feature drift calculations.
# MAGIC - Controlled same-batch reprocessing.
# MAGIC
# MAGIC The workflow currently uses historical data, and its table-update trigger remains paused.
# MAGIC
# MAGIC Automatic recovery from an actual task failure, concurrent-run safety, and production monitoring against new customer populations have not yet been validated.
# MAGIC
# MAGIC ### 8.6 Recommended Next Steps
# MAGIC
# MAGIC The next phase of the project will focus on advanced business analytics and model interpretability.
# MAGIC
# MAGIC **Notebook 11 — Cost-Sensitive Threshold Optimization**
# MAGIC
# MAGIC Evaluate classification thresholds using illustrative customer value, intervention costs, and retention effectiveness assumptions.
# MAGIC
# MAGIC **Notebook 12 — Decile Lift and Cumulative Gains**
# MAGIC
# MAGIC Evaluate how effectively the model ranks actual churners and estimate the proportion captured by targeting the highest-risk customers.
# MAGIC
# MAGIC **Notebook 13 — Model Explainability**
# MAGIC
# MAGIC Investigate which customer attributes contribute to Logistic Regression churn predictions and explain the results in business terms.
# MAGIC
# MAGIC **Notebook 14 — Retention Strategy Simulation**
# MAGIC
# MAGIC Compare hypothetical customer targeting strategies under alternative budget and intervention assumptions.
# MAGIC
# MAGIC These enhancements will extend the project from predictive modeling and MLOps toward decision-focused customer retention analytics.
# MAGIC
# MAGIC ### 8.7 GitHub Portfolio Summary
# MAGIC
# MAGIC **Project Title:**
# MAGIC
# MAGIC End-to-End Telco Customer Churn Prediction and MLOps on Databricks
# MAGIC
# MAGIC **Project Description:**
# MAGIC
# MAGIC Developed an end-to-end telecommunications customer churn prediction solution using Databricks, Apache Spark, MLflow, Unity Catalog, and Delta Lake.
# MAGIC
# MAGIC Implemented reusable feature engineering, classification model comparison, threshold optimization, registered model inference, batch scoring, operational monitoring, and statistical feature drift detection.
# MAGIC
# MAGIC The selected Logistic Regression model achieved a test ROC-AUC of 0.8385. The batch scoring workflow processed 1,409 customers, identified 400 potential retention candidates, and monitored six customer features for distribution changes.
# MAGIC
# MAGIC Validated three successful Databricks Workflow executions and demonstrated controlled batch-level reprocessing without duplicate accumulation for the tested batch.
# MAGIC
# MAGIC The project demonstrates practical machine learning engineering, statistical analysis, MLOps workflow development, and business-oriented decision support.
# MAGIC
# MAGIC ### 8.8 Final Conclusion
# MAGIC
# MAGIC The Telco Customer Churn project successfully demonstrates the integration of machine learning development, model management, batch inference, operational monitoring, and statistical drift detection within a unified Databricks environment.
# MAGIC
# MAGIC The project extends beyond model training by addressing how predictions are generated, stored, monitored, and reproduced through an operational workflow.
# MAGIC
# MAGIC Its current implementation establishes a foundation for production-oriented MLOps, while further work is required to validate automated failure recovery, live-data monitoring, and measurable business outcomes.
# MAGIC
# MAGIC The next phase will focus on translating churn predictions into more economically informed customer retention decisions.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **End of Notebook 10A — MLOps Findings Report**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC