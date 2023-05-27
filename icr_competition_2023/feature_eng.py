"""
Import training data and create a data pipeline with feature eng.
"""
import pandas as pd
# import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

train_df = pd.read_csv('train.csv')
labels = train_df["Class"]
sample_ids = train_df["Id"]

# Create a pipeline for feature engineering
col_names = train_df.columns
col_names = col_names.drop(["Id", "Class", "EJ"])

# One hot encode categorical variables in column "EJ"
train_df["EJ_A"] = train_df["EJ"].apply(lambda x: 1 if x == "A" else 0)
train_df["EJ_B"] = train_df["EJ"].apply(lambda x: 1 if x == "B" else 0)
train_df.drop("EJ", axis=1, inplace=True)

# Fill missing data with the mean of the column
print("Imputing missing values...")
for col in col_names:
    train_df[col].fillna(train_df[col].mean(), inplace=True)
    # Verify that the minimum value is not 0
    if train_df[col].min() == 0.0:
        raise ValueError(f"Minimum value of column {col} is 0")

# Compute the log of the numerical feature values
print("Computing log features...")
for col in col_names:
    new_col_name = col + "_log"
    train_df[new_col_name] = train_df[col].apply(lambda x: np.log(x))

# Compute the ratio of two numerical features as a new feature
print("Computing ratio and product features...")
for col1 in col_names:
    for col2 in col_names:
        if col1 != col2:
            ratio_col_name = col1 + "_" + col2 + "_ratio"
            train_df[ratio_col_name] = train_df[col1] / train_df[col2]
            product_col_name = col1 + "_" + col2 + "_product"
            # Avoid duplicate product features
            if col2 + "_" + col1 + "_product" not in train_df.columns:
                train_df[product_col_name] = train_df[col1] * train_df[col2]

# Drop unnecessary columns from train_df
train_df.drop(["Id", "Class"], axis=1, inplace=True)

# Implement logistic regression with 5-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lr = LogisticRegression(max_iter=1000, penalty="l1", solver="liblinear")
accuracy_scores = []
for train_index, test_index in kf.split(train_df):
    X_train, X_test = train_df.iloc[train_index], train_df.iloc[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

print(f"Accuracy scores: {accuracy_scores}")
# train_df.info()
