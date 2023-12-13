## Hyperparameter optimization results

Using whole dataset 
- note 1: parameters for production model are identical, but `n_iter = 500`
- note 2: rounded parameters to one or two significant digits to provide some robustness in the prediction
```
best_params = {
    'n_iter'           : 700,
    'verbose'          : -1,
    'objective'        : 'l2',
    'learning_rate'    : 0.08,
    'colsample_bytree' : 0.8,
    'colsample_bynode' : 0.6,
    'lambda_l1'        : 4.0,
    'lambda_l2'        : 4.5,
    'min_data_in_leaf' : 75,
    'max_depth'        : 8,
    'max_bin'          : 600,
}
```
Cross validation score ~= 62.4
Validation error on consumption predictor ~= 49.6
Validation error on production predictor ~= 79.3

Using `is_consumption = 0` subset selection
```
best_params = {
    'n_iter'           : 700,
    'verbose'          : -1,
    'objective'        : 'l2',
    'learning_rate'    : 0.04, 
    'colsample_bytree' : 0.55, 
    'colsample_bynode' : 0.65, 
    'lambda_l1'        : 2.0, 
    'lambda_l2'        : 6.0, 
    'min_data_in_leaf' : 60, 
    'max_depth'        : 5, 
    'max_bin'          : 1000
}
```
Cross validation score ~= 73.5 (Compare with 79.3 validation error on production predictor on whole dataset above)
Validation error on consumption predictor ~= 50.4
Validation error on production predictor ~= 77.1

Using `is_consumption = 1` subset selection
```
best_params = {
    'n_iter'           : 700,
    'verbose'          : -1,
    'objective'        : 'l2',
    'learning_rate'    : 0.05, 
    'colsample_bytree' : 0.55, 
    'colsample_bynode' : 0.90, 
    'lambda_l1'        : 4.5, 
    'lambda_l2'        : 0.5, 
    'min_data_in_leaf' : 200, 
    'max_depth'        : 10, 
    'max_bin'          : 650
}
```
Cross validation score ~= 49.1 (Compare with 49.6 validation error on consumption predictor on whole dataset above)
Validation error on consumption predictor ~= 49.4
Validation error on production predictor ~= 76.5

**Observe:** This last set of hyperparams is better than the hyperparams derived from the production model on both validations, so take these as my next guess.
