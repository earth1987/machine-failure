# Predicting machine failure

Predicting machine failure through machine learning is paramount for efficient industrial operations. By analysing data patterns, ML models can forecast potential breakdowns, allowing for proactive maintenance and minimising downtime. This predictive approach not only enhances equipment reliability but also optimises resource allocation.

## File structure

<pre>
|- machine-failure/
   |- custom_funcs.py
   |- config.py
|- data/
   |- raw/
   |- cleaned/    
|- notebooks/
   |- data_exploration.ipynb
   |- predicting_failure.ipynb
   |- figures/
|- .gitignore
|- LICENSE
|- README.md
</pre>

## Scope

**_Context_**

* [The AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) is a synthetic dataset that reflects real predictive maintenance data encountered in industry.

* It will be assumed that faulty machinery would lead to decreased productivity and efficiency, quality issues, defective products, and potentially safety concerns for workers.

**_Model_**

* The machine failure consists of five independent failure modes. However, the problem will initially be framed as a binary classification problem.

* No baseline model is available for benchmarking. The model will therefore be compared to a no-skill classifier.

**_Technical requirements_**

* In reality, the model would likely be deployed to an online endpoint so predictions could be generated in real-time (via a REST API).

* At this stage, the model will not be deployed. However, the low latency requirements of online deployment does place constraints on infrastructure and model complexity. For this reason, neural networks will not be tested as they can be slow at inference time due to the large number of operations.

**_Data requirements_**

* No further data collection is required.

* No personal data is involved. Consequently, there are no apparent legal or ethical constraints (e.g., GDPR).

## Model performance metrics

* The performance metrics should closely align with the specific business problem at hand.

* While a single metric simplifies ranking model performance, the dataset's imbalance makes overall accuracy unsuitable.

* It is assumed the cost of a false negative prediction (incorrectly identifying something as okay when it has failed) outweighs the cost of a false positive prediction.

* The primary metric is therefore recall for the minority (failure) class, reflecting the model's ability to identify failed machinery and aiming to minimise false negatives.

    $Recall = \frac{TP}{TP + FN}$

* A secondary metric is precision for the minority (failure) class, minimising false positives and ensuring high accuracy in failed predictions. The optimisation target is a nominal constraint of 50% precision.

    $Precision = \frac{TP}{TP + FP}$

* "Striking a balance involves achieving a reasonable level of precision to prevent unnecessary disruptions while maintaining a sufficiently high recall for effective fault detection.

    Note: AUC on the Precision-Recall will also allow for the comparison of different models in terms of their ability to balance precision and recall.

## Model selection

* It's a binary classification problem.

* The following models are tested:

    - Naive Bayes (a generative probabilistic model)
  
    - Logistic regression (a discriminative probabilistic model and linear classifier)

    - Support Vector Machine (a non-linear classifier)

    - Random Forest (a discriminative probabilistic model and non-linear classifier)

    - XGBoost (a discriminative probabilistic model and non-linear classifier)
 
## Conclusion

The goal was to maximise recall whilst ensuring precision is at least $0.5$. The below table shows the performance of each model given this constraint after hyperparameter optimisation (via 5-fold cross validation). The Random Forest model performs best. SMOTE did not significantly improve performance.

| Model | Threshold | Recall | Precision | AUC |
|----------|----------|----------|----------|----------|
| Naive Bayes   | 0.5   |0.21   | 0.5   | 0.37|
| Logistic regression   | 0.5   | 0.35   | 0.5   | 0.37   |
| SVM   | 0.23   | 0.51   | 0.5   | 0.51 |
| Random Forest   | 0.26   | 0.57   | 0.5   | 0.62   |
| XGBoost   | 0.26   | 0.52   | 0.5   | 0.58|
| Random Forest with SMOTE   | 0.42 |0.56  | 0.5   | 0.61    |

The below classification report and confusion matrix shows the test set performance of the random forest model after retraining with the entire training set. The recall and precision are $0.56$ and $0.55$ respectively.

<img src="figures/classification_report.png" align="center" width="300" />

<img src="figures/confusion_matrix.png" align="center" width="300" />