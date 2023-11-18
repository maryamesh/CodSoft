import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score, \
    ConfusionMatrixDisplay

# Load data
churn = pd.read_csv("C:/Users/dell/Desktop/Churn_Modelling.csv")
churn.columns = churn.columns.str.strip().str.lower()

# Data preprocessing
# ... (cleaning, handling duplicates, etc.)

# Visualizations
plt.figure(figsize=(15, 5))
sns.countplot(data=churn, x='exited')
plt.title('Count of Exited Customers')
plt.show()

# Feature engineering, handling categorical variables, etc.

# Resampling for balancing classes
churn_majority = churn[churn['exited'] == 0]
churn_minority = churn[churn['exited'] == 1]
churn_majority_downsample = resample(churn_majority, n_samples=2037, replace=False, random_state=42)
churn_df = pd.concat([churn_majority_downsample, churn_minority])

# Visualization after balancing classes
plt.figure(figsize=(15, 5))
sns.countplot(data=churn_df, x='exited')
plt.title('Balanced Count of Exited Customers')
plt.show()

# Feature selection, correlation analysis
# ...

# Model training and evaluation
x = churn_df.drop(['exited', 'rownumber', 'customerid', 'surname', 'geography', 'gender'], axis=1)
y = churn_df['exited']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=500),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
    'Support Vector Machine': SVC()
}

for name, model in models.items():
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    y_pred = model.predict(x_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Model: {name}")
    print(f"Train Accuracy: {train_score}")
    print(f"Test Accuracy: {test_score}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Confusion Matrix
    cmd = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=model.classes_)
    plt.figure(figsize=(6, 4))
    cmd.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.show()
