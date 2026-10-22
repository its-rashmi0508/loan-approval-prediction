# 🏦 AI Loan Approval & Risk Prediction System

An AI-powered machine learning application that predicts loan approval outcomes based on applicant information. The project uses a Kaggle loan approval dataset and compares multiple classification algorithms to identify the best-performing model.

## 🎯 Objective

The main objective is to develop a machine learning system that predicts whether a loan application is likely to be approved or rejected based on applicant details.

The project also explores different classification algorithms, evaluates their performance, and provides a foundation for an interactive loan prediction application using Streamlit.

## 🧠 Technologies Used

- **Programming Language:** Python
- **Environment:** Jupyter Notebook, Anaconda
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Joblib
- **Web Application:** Streamlit
- **Dataset:** Kaggle Loan Approval Prediction Competition
- **Version Control:** Git and GitHub

## 📊 Dataset

This project uses the [Kaggle Loan Approval Prediction dataset](https://www.kaggle.com/competitions/playground-series-s4e10/data).

The competition training dataset contains approximately **58,645 records**, providing substantially more training examples than the original dataset used in this project.

### Dataset files

- `loan_prediction.csv` — labeled records used for model training and validation.
- `test.csv` — records without target labels, used to generate predictions.
- `sample_submission.csv` — template defining the required Kaggle submission format.

**Note:** The competition dataset is synthetically generated. Although it is useful for experimenting with machine learning on a larger dataset, its performance may not directly reflect real-world loan approval outcomes.

## ⚙️ Machine Learning Workflow

The project follows these steps:

1. Load the Kaggle training dataset.
2. Inspect the dataset structure, features, and target distribution.
3. Separate input features from the target variable, `loan_status`.
4. Exclude the record ID from model training.
5. Split the labeled data into training and validation sets using an 80:20 stratified split.
6. Identify numerical and categorical features automatically.
7. Handle missing numerical values using median imputation.
8. Handle missing categorical values using most-frequent-value imputation.
9. Scale numerical features using `StandardScaler`.
10. Encode categorical features using `OneHotEncoder`.
11. Train and compare three machine learning classifiers.
12. Evaluate model performance using classification metrics.
13. Select the best model based on validation ROC-AUC.
14. Save the complete preprocessing and prediction pipeline.
15. Generate predictions for the competition test dataset when required.

Preprocessing is integrated into a Scikit-learn pipeline, helping prevent data leakage between training and validation data.

## 🤖 Machine Learning Models

Three supervised classification algorithms are compared.

### 1. Logistic Regression

Logistic Regression is used as a baseline classification model. It estimates the probability of an applicant belonging to a particular loan outcome class.

### 2. Random Forest Classifier

Random Forest combines predictions from multiple decision trees. It can capture nonlinear relationships and interactions between applicant features.

### 3. Extra Trees Classifier

Extra Trees, also known as Extremely Randomized Trees, uses randomized decision trees to learn patterns in the dataset and combine their predictions.

### Model Selection

The models are compared using the same training and validation split. The model with the highest validation ROC-AUC is selected as the final model.

The winning model and its measured performance are determined by running `train_model.py`; no model is assumed to be best before evaluation.

## 📈 Model Evaluation

The following metrics are used to evaluate the classifiers:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified applications. |
| Precision | Measures how many positive predictions are correct. |
| Recall | Measures how many actual positive cases are identified. |
| F1-score | Balances precision and recall. |
| ROC-AUC | Measures the model's ability to distinguish between the two classes across classification thresholds. |
| Confusion Matrix | Shows correct and incorrect predictions for each class. |

The training script generates a model comparison table and a classification report for the selected model.

### Model Comparison Results

The actual results are generated when the script runs. The following table should be updated with the values from `model/model_comparison.csv`.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9489 | 0.9011 |  0.7204 | 0.8007 | 0.9348 |
| Extra Trees | 0.9255 | 0.7298 | 0.7569 | 0.7431 | 0.9235 |
| Logistic Regression | 0.8451 | 0.4749 | 0.8341 |  0.6053 | 0.9068 |


## 📂 Project Structure

```text
loan-approval-prediction/
│
├── data/
│   ├── loan_prediction.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── model/
│   ├── loan_approval_pipeline.pkl
│   ├── feature_columns.pkl
│   ├── metrics.json
│   └── model_comparison.csv
│
├── train_model.py
├── app.py
├── loan_approval.ipynb
├── requirements.txt
└── README.md
```

## 🚀 Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version
- Anaconda and Jupyter Notebook
- GitHub
- A Kaggle account to obtain the dataset

### Step 1: Clone the repository

```bash
git clone https://github.com/its-rashmi0508/loan-approval-prediction.git
cd loan-approval-prediction
```

### Step 2: Install dependencies

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn streamlit
```

### Step 3: Download the dataset

Download the files from the [Kaggle competition Data page](https://www.kaggle.com/competitions/playground-series-s4e10/data).

Extract the files and place `loan_prediction.csv`, `test.csv`, and `sample_submission.csv` in the project directory or the configured data folder.

The supplied `loan_prediction_model.py` expects `train.csv` beside the script by default.

### Step 4: Train and compare models

Run the training script:

```bash
python loan_prediction_model.py
```

Alternatively, execute it from a Jupyter Notebook cell:

```python
%run loan_prediction_model.py
```

After training, the selected pipeline and model evaluation results are saved in the `model/` directory.

### Step 5: Run the Streamlit application

Once `app.py` has been updated to load `model/loan_approval_pipeline.pkl` and supply input features in the expected format, run:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to use the application.

## 💻 Application Features

The Streamlit application is intended to provide:

- An interactive form for entering applicant details.
- Loan approval prediction based on applicant information.
- Probability estimates for the predicted outcome.
- A simple and accessible user interface.
- Integration with the saved machine learning pipeline.

The application must use the same feature names and preprocessing pipeline as the trained model. The existing application may require updates before it works with the new Kaggle dataset.

## 📁 Generated Model Artifacts

| File | Description |
|---|---|
| `loan_approval_pipeline.pkl` | Selected classifier and preprocessing steps. |
| `feature_columns.pkl` | Ordered list of training feature names. |
| `metrics.json` | Evaluation metrics for the selected model. |
| `model_comparison.csv` | Validation results for all three classifiers. |

These files are generated by the updated training script.

## 🧪 Testing and Validation

The project uses a stratified training and validation split to maintain the target-class proportions.

The models are evaluated on held-out validation data. The Kaggle `test.csv` file is used to generate predictions rather than calculate accuracy because its true target labels are not provided.

Model performance should be assessed using multiple metrics rather than accuracy alone.

## 🔮 Future Enhancements

- Hyperparameter tuning using cross-validation.
- Threshold optimization based on the costs of different prediction errors.
- Additional classification algorithms and feature engineering.
- Explainable AI using SHAP or other feature-importance techniques.
- Improved Streamlit dashboard with prediction history and model evaluation charts.
- Testing on suitable real-world loan data, subject to availability and appropriate privacy safeguards.
- Automated model retraining and performance monitoring.


## 👩‍💻 Author

**Rashmi Kumari**

GitHub: [its-rashmi0508](https://github.com/its-rashmi0508)

---
