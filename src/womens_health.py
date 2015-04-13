import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier
from xgbc import *

train = pd.read_csv('train_values.csv').fillna(-1)
test = pd.read_csv('test_values.csv').fillna(-1)
labels = pd.read_csv('train_labels.csv')
sample = pd.read_csv('SubmissionFormat.csv')

test_ids = test.id.values

train = train.drop('id', axis = 1)
test = test.drop('id', axis = 1)
labels = labels.drop('id', axis = 1)

lbl_enc = preprocessing.LabelEncoder()
train['release'] = lbl_enc.fit_transform(train.release.values)
test['release'] = lbl_enc.fit_transform(test.release.values)

for column in train.columns:
	if column[0] == 'c':
		lbl_enc.fit(list(train[column].values) + list(test[column].values))
		train[column] = lbl_enc.transform(train[column].values)
		test[column] = lbl_enc.transform(test[column].values)

xg = XGBoostClassifier(n_estimators=120, eta=0.1, max_depth=7, n_jobs=8)
bg = BaggingClassifier(base_estimator=xg, n_estimators=5, max_samples=0.9, max_features=0.9, random_state=42)

X = np.array(train)
X_test = np.array(test)
labels = np.array(labels)
predictions = np.zeros((X_test.shape[0], labels.shape[1]))

for i in range(labels.shape[1]):
	bg.fit(X, labels[:,i])
	predictions[:,i] = bg.predict_proba(X_test)[:,1]


predictions = pd.DataFrame(np.column_stack(test_ids, predictions), columns=sample.columns)
predictions.to_csv('submission.csv', index = False)
