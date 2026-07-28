import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV


df=pd.read_csv('employeedata.csv')

x=df.drop(columns=["Attrition", 'PercentSalaryHike', 'Education', 'RelationshipSatisfaction'])  # all the independent features
y=df["Attrition"]               # dependent feature

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

print(x.columns.tolist())

#1.
p_lor = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=0.1, max_iter=1000, solver='saga', class_weight='balanced'))])
p_lor.fit(x_train, y_train)
pred_lor=p_lor.predict(x_test)
pred_lor_proba = p_lor.predict_proba(x_test)

#2
p_dt = Pipeline([("scaler", StandardScaler()), ("model", DecisionTreeClassifier(criterion='gini', max_depth=15, max_features='sqrt', min_samples_leaf=2, min_samples_split=5, random_state=42, class_weight='balanced'))])
p_dt.fit(x_train, y_train)
pred_dt=p_dt.predict(x_test)
pred_dt_proba = p_dt.predict_proba(x_test)


#3
p_rf = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(max_depth=10, max_features='sqrt', min_samples_leaf=2, min_samples_split=10, n_estimators=200, random_state=42, class_weight='balanced'))])
p_rf.fit(x_train, y_train)
pred_rf=p_rf.predict(x_test)
pred_rf_proba = p_rf.predict_proba(x_test)


print("------------LOR------------")
print("Accuracy:", accuracy_score(y_test, pred_lor))
print("Classification Report:\n", classification_report(y_test, pred_lor))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_lor))
print("ROC AUC Score:", roc_auc_score(y_test, pred_lor_proba[:,1]))

print("------------DT------------")
print("Accuracy:", accuracy_score(y_test, pred_dt))
print("Classification Report:\n", classification_report(y_test, pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_dt))
print("ROC AUC Score:", roc_auc_score(y_test, pred_dt_proba[:,1]))

print("------------Rf------------")
print("Accuracy:", accuracy_score(y_test, pred_rf))
print("Classification Report:\n", classification_report(y_test, pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_rf))
print("ROC AUC Score:", roc_auc_score(y_test, pred_rf_proba[:,1]))

