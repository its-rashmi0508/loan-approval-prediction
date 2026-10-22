# 🏦 Loan Approval Prediction Using Machine Learning

A Machine Learning project that predicts whether a loan application is likely to be **Approved or Rejected** based on applicant information, financial details, and credit history.

## 🎯 Project Objective

The objective of this project is to develop a machine learning model that analyzes applicant data and predicts loan eligibility. It demonstrates how data preprocessing, feature engineering, and classification algorithms can be used to support loan approval analysis.

## 📌 Problem Statement

Financial institutions evaluate several factors before approving a loan, including applicant income, employment status, credit history, loan amount, and repayment period.

Manually analyzing these factors can be time-consuming. This project explores the use of supervised machine learning to automate the prediction process and provide a preliminary assessment of loan eligibility.

## ✨ Key Features

- Predict loan approval status using machine learning.
- Analyze applicant financial and demographic information.
- Perform data cleaning and preprocessing.
- Handle categorical and numerical features.
- Split the dataset into training and testing sets.
- Train a classification model.
- Evaluate model performance using standard metrics.
- Generate predictions for loan applications.

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Model training and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data analysis |
| Jupyter Notebook | Experimentation and model development |
| Streamlit | Interactive web interface, if included |

## 📊 Dataset Description

The dataset contains applicant information used to train and evaluate the loan approval prediction model.

Depending on the dataset used in this project, the input features may include:

| Feature | Description |
|---|---|
| Gender | Gender of the applicant |
| Married | Marital status |
| Dependents | Number of dependents |
| Education | Educational qualification |
| Self_Employed | Employment status |
| ApplicantIncome | Applicant's income |
| CoapplicantIncome | Co-applicant's income |
| LoanAmount | Requested loan amount |
| Loan_Amount_Term | Loan repayment period |
| Credit_History | Credit history information |
| Property_Area | Residential property area |
| Loan_Status | Target variable indicating approval status |

**Target variable:** `Loan_Status`

The model predicts one of two outcomes: **Approved** or **Rejected**.

*Note: The exact columns depend on the dataset included in the repository.*

## ⚙️ Machine Learning Workflow

The project follows these steps:

1. **Data Collection:** Load the loan application dataset.
2. **Data Understanding:** Inspect the dataset, column types, distributions, and target variable.
3. **Data Preprocessing:** Handle missing values, duplicate records, and inconsistent data where necessary.
4. **Exploratory Data Analysis:** Examine patterns and relationships between applicant features and loan approval.
5. **Feature Engineering:** Encode categorical variables and prepare numerical features.
6. **Data Splitting:** Divide the dataset into training and testing sets.
7. **Model Training:** Train a classification algorithm on the prepared training data.
8. **Model Evaluation:** Measure performance using appropriate classification metrics.
9. **Prediction:** Use the trained model to predict the approval status of new applications.

## 🤖 Machine Learning Model

The project uses supervised learning to solve a binary classification problem.

The model learns patterns from historical loan application data and uses those patterns to predict the target class for new records.

The specific algorithm should match the implementation in the repository. Common algorithms for this task include:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

## 📈 Model Evaluation

The model can be evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | 0.8049 |
| Precision | 0.8427 |
| Recall | 0.8824 |
| F1-Score | 0.8621 |
| Confusion Matrix | 0.8115 |

The final evaluation results should be reported from the actual test dataset. No accuracy value is assumed in this README.

## 📂 Project Structure

The following is a suggested structure. Keep only the files and folders that actually exist in your repository.

```text
loan-approval-prediction/
│
├── data/
│   └── loan_approval_dataset.csv
│
├── notebooks/
│   └── loan_approval_prediction.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

**File descriptions**

- `data/` — Contains the dataset, if included.
- `notebooks/` — Contains exploratory data analysis, preprocessing, and model training.
- `app.py` — Application entry point, if a Streamlit interface is implemented.
- `requirements.txt` — Lists the Python dependencies.
- `README.md` — Documents the project and setup instructions.
- `.gitignore` — Specifies files Git should ignore.

## 🚀 Installation and Setup

### Prerequisites

Make sure you have installed:

- Python 3.10 or a compatible version for your dependencies
- Git
- A code editor such as Visual Studio Code or PyCharm
- Jupyter Notebook, if the project uses notebooks

### Step 1: Clone the repository

```bash
git clone https://github.com/its-rashmi0508/loan-approval-prediction.git
```

### Step 2: Navigate to the project directory

```bash
cd loan-approval-prediction
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### Step 4: Install dependencies

If `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the packages required by your implementation:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

If the project includes a Streamlit interface, install Streamlit as well:

```bash
pip install streamlit
```

## ▶️ How to Run the Project

### Option 1: Run the Jupyter Notebook

If your repository contains a notebook, launch Jupyter:

```bash
jupyter notebook
```

Open the loan approval prediction notebook and execute the cells in order, from data loading and preprocessing through model evaluation.

### Option 2: Run the Streamlit Application

If your repository contains `app.py` and the required trained model files, start the application using:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

Enter the required applicant details and submit them to view the model's prediction.

*Use only the execution method supported by the actual project files.*

## 🧪 Example Prediction

An application may contain information such as:

| Input | Example |
|---|---|
| Applicant income | 50,000 |
| Loan amount | 150,000 |
| Credit history | Available |
| Education | Graduate |
| Employment status | Salaried |

The model processes the applicant information and returns a predicted class:

- **Approved** — The model predicts that the application belongs to the approval class.
- **Rejected** — The model predicts that the application belongs to the rejection class.

These values are illustrative only. Actual input requirements and predictions depend on the trained model and its dataset.

## 🔮 Future Enhancements

- Compare multiple classification algorithms.
- Improve model performance through hyperparameter tuning.
- Add interactive visualizations for applicant data.
- Display prediction confidence where supported by the model.
- Build a user-friendly web application.
- Add model explainability to understand important prediction factors.
- Evaluate fairness and potential bias across applicant groups.
- Deploy the application for online access.

## ⚠️ Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- The model may not generalize to financial institutions with different lending policies.
- Historical data may contain biases that affect predictions.
- A predicted approval does not guarantee that a bank will approve the loan.

This project is intended for educational and demonstration purposes. Its predictions should not be used as the sole basis for real-world lending decisions.

## 👩‍💻 Author

**Rashmi Kumari**

- GitHub: [@its-rashmi0508](https://github.com/its-rashmi0508)
- Project: [Loan Approval Prediction](https://github.com/its-rashmi0508/loan-approval-prediction)

## 📄 License

No license is specified here. If you want others to reuse or modify this project, consider adding an appropriate open-source license to the repository.

---
