# ICR - Identifying Age-Related Conditions - Project Notes

## First Attempts

### May 18, 2023

- Submitting a baseline test with 99.9% for all class 0 (no conditions) yields a weighted score of 3.41
  - Note that -ln(0.001) = 6.9078, so based on log loss function structure, there are more class 0 examples in the test set than class 1
- Using data explorer, many columns (about 40) of training set are skewed left, could benefit from log-smoothed values. Exceptions:
  - AZ has mean around 10 with a few outliers
  - BN has mean around 20
  - CR has mean around 0.8
  - CU has mean around 1.3
  - CW is bimodal near min (7) and about 37
  - DA has mean around 50
  - DH has mean around 0.4
  - DL has mean around 90
  - DN has mean around 25
  - EJ is a categorical feature - A or B
  - EL has most data points at maximum (109.13)
  - FI has mean around 10
  - GH has mean around 30
  - GL is bimodal near min (0) and max (21.978) values
- Basic models to try: Logistic Regression with L1 penalty, SVC with linear kernel (with L1 penalty)
  - Evaluate both in scripts before submitting a solution to Kaggle
  
### May 19, 2023
- Quick submission for the day, modify probabilities to 90% for class 0 and 10% for class 1
  - Weighted Score dropped to 1.2 -> 896 out of 934
  - Note that -ln(0.1) = 2.3026 and -ln(0.9) = 0.1054
  - We could also see what happens at 50-50 probabilities -> -ln(0.5) = 0.6931
    - **This is already listed in `sample_submission.csv` entry -> score = 0.69
  - At 66-33 split -> -ln(0.66) = 0.4155, -ln(0.33) = 1.1087
 - Leaderboard: 
   - 700th place = 0.37
   - 500th place = 0.29
   - 250th place = 0.22
   - 100th place = 0.19
   - 25th place = 0.18
   - 10th place = 0.17
   - 1st place = 0.14

### May 20, 2023
- Quick submission for the day, 80% for class 0 and 20% for class 1
  - Weighted score dropped to 0.91, still very low on the scoreboard
- Other models to try
  - Basic two-layer neural net
- Feature engineering; to add to DataFrame:
  - Ratios of two features
  - Products of two features
  - Log of certain features (as above)
  - Sums of two features

### May 28, 2023
- Completed a notebook with basic logistic regression with L1 penalty method
  - 5-fold cross validation did note accuracies upwards of 88%, surprisingly
  - Score = 6.67 on test set?!
    - ln 10^-15 is about -34.539, for reference; ln 0.9 is about -0.1 and ln 0.99 is about -0.01
  - Retry using the probabilities themselves rather than a simple binary 0-1 decision (resubmit tomorrow)
    - Done after 8pm; Score = 1.51 - still not good?!
- Leaderboard state:
  - 1st place = 0.13
  - 10th place = 0.14
  - 25th place = 0.15
  - 100th place = 0.16
  - 250th place = 0.17
- ideas for future exploration (based on Kaggle forum reading)
  - tabPFN package - transformer NN for learning tabular data
  - autosklearn - metalearning using an sklearn-like interface
  - try leveraging the `greeks.csv` data: predict greeks as an intermediate step to binary prediction
    - Alpha column specifies the specific illness - A if no condition; B, D, or G for specific condition
    - Beta, Gamma, and Delta are "experimental characteristics"

### June 8, 2023
- Investigate preprocessing utilities of scikit-learn
  - StandardScaler, PowerTransformer, RobustScaler, MinMaxScaler, MaxAbsScaler
- There's a [notebook](https://www.kaggle.com/code/shashanknecrothapa/icr-classification-logistic-regression) where logisitic regression gets a score of 0.19 using just StandardScaler and the product of features
- Leaderboard state:
  - 1st place = 0.10
