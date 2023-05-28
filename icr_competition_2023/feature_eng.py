"""
Import training data and create a data pipeline with feature eng.
"""
import pandas as pd
# import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

def feature_eng(df):
    # Create a pipeline for feature engineering
    col_names = df.columns
    col_names = col_names.drop(["Id", "Class", "EJ"], errors="ignore")

    # One hot encode categorical variables in column "EJ"
    df["EJ_A"] = df["EJ"].apply(lambda x: 1 if x == "A" else 0)
    df["EJ_B"] = df["EJ"].apply(lambda x: 1 if x == "B" else 0)
    df.drop("EJ", axis=1, inplace=True)

    # Fill missing data with the mean of the column
    print("Imputing missing values...")
    for col in col_names:
        df[col].fillna(df[col].mean(), inplace=True)
        # Verify that the minimum value is not 0
        if df[col].min() == 0.0:
            # raise ValueError(f"Minimum value of column {col} is 0")
            df[col] = df[col].apply(lambda x: x + 0.0001)

    # Compute the log of the numerical feature values
    print("Computing log features...")
    series_dict = {}
    for col in col_names:
        new_col_name = col + "_log"
        series_dict[new_col_name] = df[col].apply(lambda x: np.log(x))
    log_df = pd.DataFrame(series_dict)
    df = pd.concat([df, log_df], axis=1)

    # Compute the ratio of two numerical features as a new feature
    print("Computing ratio and product features...")
    series_dict = {}
    for col1 in col_names:
        for col2 in col_names:
            if col1 != col2:
                ratio_col_name = col1 + "_" + col2 + "_ratio"
                series_dict[ratio_col_name] = df[col1] / df[col2]
                product_col_name = col1 + "_" + col2 + "_product"
                # Avoid duplicate product features
                if col2 + "_" + col1 + "_product" not in df.columns:
                    series_dict[product_col_name] = df[col1] * df[col2]
    ratio_prod_df = pd.DataFrame(series_dict)
    df = pd.concat([df, ratio_prod_df], axis=1)

    return df

train_df = pd.read_csv('train.csv')
labels = train_df["Class"]
sample_ids = train_df["Id"]

# Drop unnecessary columns from df
train_df.drop(["Id", "Class"], axis=1, inplace=True, errors="ignore")
train_df = feature_eng(train_df)

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

# Train the model on the entire training set
lr.fit(train_df, labels)

# Import test data and create a data pipeline with feature eng.
test_df = pd.read_csv('test.csv')
test_ids = test_df["Id"]
test_df = feature_eng(test_df)
test_df.drop(["Id"], axis=1, inplace=True, errors="ignore")
y_pred = lr.predict(test_df)

# Get the probability of each class
y_pred_proba = lr.predict_proba(test_df)

# Create a submission file
submission_df = pd.DataFrame({"Id": test_ids, "class_0": (y_pred == 0).astype(float), "class_1": (y_pred == 1).astype(float)})
submission_df.to_csv("submission.csv", index=False)
