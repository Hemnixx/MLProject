
# 🎓 Student Performance Prediction System

## 📌 Project Overview

The **Student Performance Prediction System** is an end-to-end machine learning project designed to predict a student’s **Math score** based on demographic, academic, and preparation-related features.
This project demonstrates a complete ML pipeline including data ingestion, data transformation, model training, evaluation, and prediction on new/unseen data.

---

## 🚀 Key Features

* End-to-end ML pipeline (industry-style structure)
* Data ingestion and train-test split
* Data preprocessing using pipelines
* Multiple regression models comparison
* Best model selection based on **R² score**
* Model & preprocessor serialization
* Prediction pipeline for new data
* Clean and modular code structure

---

## 🧠 Problem Statement

Predict the **Math score of a student** using features such as:

* Gender
* Race/Ethnicity
* Parental level of education
* Lunch type
* Test preparation course
* Reading score
* Writing score

This is a **regression problem** since the target variable is a continuous numerical value (0–100).

---

## 🗂️ Project Structure

```
MLPROJECT/
│
├── artifacts/
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── logs/
│
├── notebook/
│   └── EDA.ipynb
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── venv/
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

---

## ⚙️ Tech Stack

* **Programming Language:** Python
* **Libraries:**

  * NumPy
  * Pandas
  * Scikit-learn
  * CatBoost
  * XGBoost
* **Tools:** Git, VS Code

---

## 📊 Machine Learning Models Used

* Linear Regression
* Random Forest Regressor
* Decision Tree Regressor
* Gradient Boosting Regressor
* K-Neighbors Regressor
* XGBoost Regressor
* CatBoost Regressor
* AdaBoost Regressor

The best model is selected based on the **R² score**.

---

## 🔄 Workflow

1. **Data Ingestion**

   * Read raw dataset
   * Split into train and test sets
   * Store data in `artifacts/`

2. **Data Transformation**

   * Handle categorical & numerical features
   * Apply scaling and encoding
   * Save preprocessing object

3. **Model Training**

   * Train multiple regression models
   * Evaluate using R² score
   * Save the best-performing model

4. **Prediction Pipeline**

   * Load trained model & preprocessor
   * Transform new data
   * Generate predictions

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd MLPROJECT
```

### 2️⃣ Create & activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run training pipeline

```bash
python -m src.components.data_ingestion
```

---

## 🔮 Predict on New Data

Example:

```python
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

data = CustomData(
    gender="female",
    race_ethnicity="group B",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="none",
    reading_score=72,
    writing_score=74
)

df = data.get_data_as_dataframe()
predictor = PredictPipeline()
result = predictor.predict(df)

print("Predicted Math Score:", result[0])
```

---

## 📈 Evaluation Metric

* **R² Score** (Coefficient of Determination)

---

## 📌 Future Improvements

* Add Flask / FastAPI for web deployment
* Build a Streamlit UI
* Dockerize the application
* Add CI/CD pipeline
* Deploy on cloud platforms (AWS / Render / Azure)

---

## 👨‍💻 Author

**Neeraj Kumar Gupta**
B.Tech (ECE), NIT Kurukshetra

---

## ⭐ Acknowledgements

* Dataset inspired by student performance datasets used for educational analytics.
* Scikit-learn & open-source ML community.



sab bana deta hoon 😄
