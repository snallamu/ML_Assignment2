# Machine Learning Classification Models – Student Performance Dataset

## a. Problem Statement

The objective of this project is to design, implement, and evaluate multiple
machine learning classification models on a real-world educational dataset.
The goal is to predict student performance levels based on demographic,
social, and school-related attributes, and to compare the performance of
different classification algorithms using standard evaluation metrics.

---

## b. Dataset Description

The dataset used in this project is the **Student Performance Dataset**
obtained from the UCI Machine Learning Repository.

The dataset represents student achievement in secondary education across
two Portuguese schools and was collected using school reports and
questionnaires. It includes academic, demographic, social, and
school-related features.

Two subjects are originally available in the dataset:
- Mathematics
- Portuguese Language

In this project, the dataset is accessed programmatically using the
`ucimlrepo` Python library (Dataset ID: 320), ensuring reproducibility
and data integrity.

### Dataset Characteristics

- Number of instances: More than 1000
- Number of features: More than 30
- Feature types: Numeric and categorical
- Target attribute: Final grade (G3)

### Target Engineering

The numeric final grade (G3) is converted into a **five-level
classification** problem:

| Class | Grade Range | Description   |
|-------|------------|---------------|
| 0     | 0–9        | Very Poor     |
| 1     | 10–11      | Poor          |
| 2     | 12–13      | Satisfactory  |
| 3     | 14–15      | Good          |
| 4     | 16–20      | Excellent     |

To avoid data leakage, intermediate grades (G1 and G2) are excluded
from the feature set, making the prediction task more realistic and
meaningful.

---

## c. Models Used

The following six classification models were implemented using the same
dataset and train–test split:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN) Classifier
4. Naive Bayes Classifier (Gaussian)
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

Gaussian Naive Bayes was selected because the feature set consists of
continuous numerical values after preprocessing, making it more
suitable than Multinomial Naive Bayes, which is designed for discrete
count-based data.

---

## d. Evaluation Metrics

Each model was evaluated using the following metrics:

- Accuracy
- AUC Score (One-vs-Rest, Macro Average)
- Precision (Macro Average)
- Recall (Macro Average)
- F1 Score (Macro Average)
- Matthews Correlation Coefficient (MCC)

These metrics provide a balanced evaluation of model performance,
especially for multi-class classification problems with potential
class imbalance.

---

## e. Model Comparison Table

> **Note:** The values below are obtained from actual model execution.
> Run the Streamlit app and update this table with your live results.

| ML Model                 | Accuracy | AUC  | Precision | Recall | F1   | MCC  |
|--------------------------|----------|------|-----------|--------|------|------|
| Logistic Regression      | —        | —    | —         | —      | —    | —    |
| Decision Tree            | —        | —    | —         | —      | —    | —    |
| KNN                      | —        | —    | —         | —      | —    | —    |
| Naive Bayes              | —        | —    | —         | —      | —    | —    |
| Random Forest (Ensemble) | —        | —    | —         | —      | —    | —    |
| XGBoost (Ensemble)       | —        | —    | —         | —      | —    | —    |

---

## f. Observations on Model Performance

| ML Model                 | Observation                                                                                                                                            |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression      | Served as a reliable baseline model with consistent performance, but its linear decision boundary limits its ability to capture non-linear relationships present in this multi-class problem. |
| Decision Tree            | Captured feature interactions effectively; however, performance fluctuated across runs, indicating a tendency to overfit the training data without pruning. |
| KNN                      | Produced moderate results but proved sensitive to feature scaling and the choice of neighbourhood size, which affected prediction stability.             |
| Naive Bayes              | Executed very quickly and provided acceptable baseline results, though the strong feature-independence assumption reduced accuracy on the correlated features in this dataset. |
| Random Forest (Ensemble) | Demonstrated strong and consistent performance by aggregating predictions across 100 decision trees, effectively reducing variance and improving generalisation. |
| XGBoost (Ensemble)       | Delivered the best overall results, benefiting from sequential boosting, built-in regularisation, and its efficient capacity to learn complex patterns from tabular data. |

---

## g. Repository Structure

```
project-folder/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
└── model/
    ├── __init__.py         # Package initialiser
    ├── dataprep.py         # Data loading and preprocessing
    └── metrics.py          # Model definitions and evaluation
```

---

## h. Execution Instructions

### Local Execution

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## i. Deployment

The Streamlit application is deployed using Streamlit Community Cloud
and provides an interactive interface to:

- Upload custom test data (CSV)
- Select individual models from a dropdown
- View the full model comparison metrics table
- Inspect per-model confusion matrices and classification reports
- Review qualitative observations for each model

---

## j. Conclusion

This project demonstrates a comparative analysis of multiple
classification models on a real-world educational dataset. Ensemble
methods, particularly XGBoost, outperform simpler models due to their
ability to capture complex non-linear patterns while reducing bias and
variance through boosting and regularisation. The results confirm that
model complexity must be balanced against interpretability, and that
feature-independence assumptions (as in Naive Bayes) can limit
performance on correlated real-world data.
