# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Telco Advanced Model Training and Comparison
# MAGIC
# MAGIC **Goal:** Compare Logistic Regression, Random Forest, and Gradient-Boosted Trees using a consistent validation split, then assess the chosen candidate on the existing held-out test set.
# MAGIC
# MAGIC **Important:** Earlier notebooks have already inspected test-set results. Treat this as an exploratory comparison; a fresh untouched test set is needed for an unbiased final production estimate.
# MAGIC
# MAGIC **Data:** `workspace.default.telco_train_raw` and `workspace.default.telco_test_raw`. **Preprocessing pipeline:** `/Volumes/workspace/default/telco_data/telco_preprocessing_pipeline`.
# MAGIC
# MAGIC Run cells in order. Each numbered section explains the next step before showing code.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Import libraries
# MAGIC
# MAGIC Load PySpark ML models and evaluation tools. We will fit fresh preprocessing only on the training portion to prevent validation leakage.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import pandas as pd
import matplotlib.pyplot as plt
print('Imports complete')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load existing raw training and test tables
# MAGIC
# MAGIC The raw training table is the source for training and validation. Do not tune models on the test table.

# COMMAND ----------

train_raw = spark.table('workspace.default.telco_train_raw')
test_raw = spark.table('workspace.default.telco_test_raw')
print('Raw train:', train_raw.count(), 'Raw test:', test_raw.count())
print('Columns:', train_raw.columns)
assert 'Churn' in train_raw.columns and 'Churn' in test_raw.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create training and validation subsets
# MAGIC
# MAGIC Use a reproducible 80/20 split of the previously saved training data. The test set stays separate.

# COMMAND ----------

fit_raw, valid_raw = train_raw.randomSplit([0.8, 0.2], seed=42)
print('Fit:', fit_raw.count(), 'Validation:', valid_raw.count())
assert fit_raw.count() > 0 and valid_raw.count() > 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Fit preprocessing without leakage
# MAGIC
# MAGIC The pipeline saved in Notebook 03 was fitted on the original training set. To avoid leaking validation information, reconstruct its *unfitted* stages and fit them on the new fit subset. This works for standard Spark ML transformers/estimators. If a custom stage cannot be reconstructed, stop and review that stage rather than using the already fitted pipeline for model selection.

# COMMAND ----------

saved_pipeline = PipelineModel.load('/Volumes/workspace/default/telco_data/telco_preprocessing_pipeline')
print('Saved pipeline stages:', len(saved_pipeline.stages))
for i, stage in enumerate(saved_pipeline.stages):
    print(i, type(stage).__name__)
# Inspect the stages above before running Section 5.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Rebuild and fit preprocessing on fit subset
# MAGIC
# MAGIC Convert fitted stages such as StringIndexerModel, OneHotEncoderModel and StandardScalerModel back to corresponding estimators with matching parameters. This avoids using category mappings or scaling statistics learned from validation data.

# COMMAND ----------

# DBTITLE 1,untitled
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, StandardScaler, Imputer

fresh_stages = []
for stage in saved_pipeline.stages:
    cls = type(stage).__name__
    if cls == 'StringIndexerModel':
        estimator = StringIndexer(inputCol=stage.getInputCol(), outputCol=stage.getOutputCol(), handleInvalid=stage.getHandleInvalid(), stringOrderType='frequencyDesc')
    elif cls == 'OneHotEncoderModel':
        estimator = OneHotEncoder(inputCols=stage.getInputCols(), outputCols=stage.getOutputCols(), dropLast=stage.getDropLast(), handleInvalid=stage.getHandleInvalid())
    elif cls == 'StandardScalerModel':
        estimator = StandardScaler(inputCol=stage.getInputCol(), outputCol=stage.getOutputCol(), withMean=stage.getWithMean(), withStd=stage.getWithStd())
    elif cls == 'ImputerModel':
        estimator = Imputer(inputCols=stage.getInputCols(), outputCols=stage.getOutputCols(), strategy=stage.getStrategy(), missingValue=stage.getMissingValue())
    else:
        # Stateless transformer stages can be reused; fitted unknown stages must be reviewed.
        if cls.endswith('Model'):
            raise ValueError(f'Review fitted pipeline stage before proceeding: {cls}')
        estimator = stage.copy({})
    fresh_stages.append(estimator)

preprocessor = Pipeline(stages=fresh_stages).fit(fit_raw)
fit_features = preprocessor.transform(fit_raw).withColumn('label', F.when(F.col('Churn') == 'Yes', 1.0).otherwise(0.0))
valid_features = preprocessor.transform(valid_raw).withColumn('label', F.when(F.col('Churn') == 'Yes', 1.0).otherwise(0.0))
print('Fit features:', fit_features.count(), 'Validation features:', valid_features.count())
print('Feature vector length:', fit_features.select('features').first()['features'].size)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Train the three candidate algorithms
# MAGIC
# MAGIC Use the same training features for every model. Tree-based models do not need scaled features, but using the shared feature vector ensures a comparable experiment.

# COMMAND ----------

models = {
    'Logistic Regression': LogisticRegression(featuresCol='features', labelCol='label', maxIter=100, regParam=0.01),
    'Random Forest': RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=100, maxDepth=6, seed=42),
    'Gradient-Boosted Trees': GBTClassifier(featuresCol='features', labelCol='label', maxIter=50, maxDepth=4, seed=42)
}
trained = {}
for name, estimator in models.items():
    print('Training:', name)
    trained[name] = estimator.fit(fit_features)
print('All candidate models trained')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Evaluate validation performance
# MAGIC
# MAGIC At a fixed 0.50 threshold, calculate accuracy, churn precision, churn recall, churn F1 and ROC-AUC. ROC-AUC measures ranking quality independent of the threshold.

# COMMAND ----------

auc_eval = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='rawPrediction', metricName='areaUnderROC')

def evaluate(pred):
    counts = {(int(r['label']), int(r['prediction'])): r['count'] for r in pred.groupBy('label', 'prediction').count().collect()}
    tp, fp = counts.get((1,1),0), counts.get((0,1),0)
    fn, tn = counts.get((1,0),0), counts.get((0,0),0)
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {'Accuracy':(tp+tn)/(tp+tn+fp+fn), 'Precision':precision, 'Recall':recall, 'F1':f1, 'ROC-AUC':auc_eval.evaluate(pred)}

rows=[]
for name, model in trained.items():
    metrics=evaluate(model.transform(valid_features))
    rows.append({'Model':name, **metrics})
validation_results=pd.DataFrame(rows).sort_values('ROC-AUC', ascending=False)
display(validation_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Visualize the model comparison
# MAGIC
# MAGIC Compare the models on the validation data. High recall matters for retention campaigns, but precision controls how many unnecessary contacts are made.

# COMMAND ----------

ax=validation_results.set_index('Model')[['Precision','Recall','F1','ROC-AUC']].plot(kind='bar', figsize=(11,5), ylim=(0,1), rot=15)
ax.set_title('Validation performance by algorithm')
ax.set_ylabel('Score')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Select a candidate using validation ROC-AUC
# MAGIC
# MAGIC Select the model with the highest validation ROC-AUC as the ranking baseline. A different metric may be more appropriate when campaign costs are known.

# COMMAND ----------

winner_name = validation_results.iloc[0]['Model']
print('Validation ROC-AUC winner:', winner_name)
print(validation_results.iloc[0].to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Retrain selected algorithm on all original training data
# MAGIC
# MAGIC Refit preprocessing on the entire original training set, then refit the selected estimator. This uses all available training records without fitting anything on the test set.

# COMMAND ----------

full_preprocessor = Pipeline(stages=fresh_stages).fit(train_raw)
full_train = full_preprocessor.transform(train_raw).withColumn('label', F.when(F.col('Churn')=='Yes',1.0).otherwise(0.0))
full_test = full_preprocessor.transform(test_raw).withColumn('label', F.when(F.col('Churn')=='Yes',1.0).otherwise(0.0))
selected_estimator = models[winner_name].copy({})
final_model = selected_estimator.fit(full_train)
print('Final candidate trained:', winner_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Evaluate selected candidate on the held-out test set
# MAGIC
# MAGIC Evaluate once after choosing the algorithm on validation data. Since earlier notebooks already examined this test set, interpret this as an exploratory result, not an untouched final benchmark.

# COMMAND ----------

final_predictions = final_model.transform(full_test)
final_metrics = evaluate(final_predictions)
print('Selected model:', winner_name)
display(pd.DataFrame([{'Model':winner_name, **final_metrics}]))
final_predictions.groupBy('label','prediction').count().orderBy('label','prediction').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Save artifacts and comparison results
# MAGIC
# MAGIC Save the selected model and its matching preprocessing pipeline together. The model expects features produced by this specific pipeline. Also save the validation comparison as a Delta table.

# COMMAND ----------

base='/Volumes/workspace/default/telco_data'
full_preprocessor.write().overwrite().save(base+'/telco_07_preprocessing_pipeline')
final_model.write().overwrite().save(base+'/telco_07_selected_model')
comparison_spark=spark.createDataFrame(validation_results.to_dict('records'))
comparison_spark.write.mode('overwrite').format('delta').saveAsTable('workspace.default.telco_07_model_comparison')
print('Saved selected model:', winner_name)
print('Saved validation metrics table: workspace.default.telco_07_model_comparison')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion and checkpoint
# MAGIC
# MAGIC Record which model won on validation ROC-AUC and whether it improved churn recall and F1 relative to the earlier Logistic Regression model. **Do not assume** a tree ensemble must outperform Logistic Regression. In the next notebook we can tune hyperparameters and thresholds on validation data and estimate campaign costs.
# MAGIC
# MAGIC
