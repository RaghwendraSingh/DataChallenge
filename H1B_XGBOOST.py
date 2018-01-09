
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split

h1b = pd.read_csv('fractal/h1b_data/h1b_TRAIN.csv', header=0)
h1b = h1b[h1b['CASE_STATUS'].notnull()]
#column_list = ['CASE_STATUS', 'EMPLOYER_NAME', 'SOC_NAME', 'JOB_TITLE',
#       'FULL_TIME_POSITION', 'PREVAILING_WAGE', 'YEAR', 'WORKSITE', 'lon',
#       'lat']
h1b_train, h1b_test = train_test_split(h1b, test_size=0.2)

# We'll impute missing values using the median for numeric columns and the most
# common value for string columns.
# This is based on some code by 'sveitser' at http://stackoverflow.com/a/25562948
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

feature_columns_to_use = ['EMPLOYER_NAME', 'SOC_NAME', 'JOB_TITLE', 'FULL_TIME_POSITION', 'WORKSITE',
                          'YEAR', 'PREVAILING_WAGE']
nonnumeric_columns = ['EMPLOYER_NAME', 'SOC_NAME', 'JOB_TITLE', 'FULL_TIME_POSITION', 'WORKSITE']

big_X = h1b_train[feature_columns_to_use].append(h1b_test[feature_columns_to_use])
big_X_imputed = DataFrameImputer().fit_transform(big_X)

le = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

# Prepare the inputs for the model
train_X = big_X_imputed[0:h1b_train.shape[0]].as_matrix()
test_X = big_X_imputed[h1b_train.shape[0]::].as_matrix()
train_y = le.fit_transform(h1b_train['CASE_STATUS'])

# see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
predictions = gbm.predict(test_X)

# finding the accuracy of the trained model
res = pd.DataFrame(
    {
        'CASE_ID': h1b_test['CASE_ID'],
        'CASE_STATUS': le.fit_transform(h1b_test['CASE_STATUS']),
        'CASE_STATUS_PRED': predictions
    })
# correct predictions
(res['CASE_STATUS'] == res['CASE_STATUS_PRED']).sum()
# total observations
(res['CASE_STATUS'] == res['CASE_STATUS_PRED']).count()
# 87.20 % accuracy