# Employee-Attrition-Predictor
A web application for predicting employee attrition
The system helps HR departments identify employees who may be at risk of leaving.

---

## Features

- Employee attrition prediction using Machine Learning
- Random Forest Pipeline model
- Interactive employee assessment form
- Real-time prediction results
- Risk probability and risk level display
- Interactive analytics dashboard
- Responsive user interface

---

## Technologies Used

### Backend
- Python
- Flask
- Scikit-learn

### Frontend
- HTML5
- CSS3
- JavaScript

### Machine Learning Model
- Random Forest Classifier

---

## Dataset

This project uses the **IBM HR Analytics Employee Attrition Dataset**, which contains employee demographic, job, and workplace-related information.

### Target Variable
- Attrition
  - Yes (Employee Left)
  - No (Employee Stayed)

---

## Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. One-Hot Encoding
4. Feature Selection
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Model Deployment
9. Prediction through Web Application

---

## Project Structure

```
Employee-Attrition-Prediction/
│
├── app.py
├── README.md
│
├── model/
│   └── attrition_model.pkl
│
├── static/
│   ├── css/
│   └── js/
│    
├── templates/
│   ├── index.html
│   └── assessment.html
|
└── training/
    ├── attrition_model.py
    └── employeedata.csv
```

---

## Model Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 77.89% |
| Decision Tree | 70.41% |
| **Random Forest** | **84.01%** |

### Random Forest Metrics

- Accuracy: **84.01%**
- ROC-AUC Score: **0.811**
