import pandas as pd
import numpy as np
import os

# Loading the data


path = os.path.join('datasets', 'titanic')
train_path = os.path.join(path, 'train.csv')
test_path = os.path.join(path, 'test.csv')

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
# print(str(test_data))

train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")

# print(train_data.info())  # we have missing values in variables like Age and Cabin, we need imputation
# print(train_data.describe())  # just for fun

num_attribs = ["Age", "SibSp", "Parch", "Fare"]  # numerical attributes
cat_attribs = ["Pclass", "Sex", "Embarked"]  # categorical attributes

# Pipeline Building

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])  # numeric pipeline, imputing missing values via median and scaling numeric inputs

from sklearn.preprocessing import OneHotEncoder

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('cat_encoder', OneHotEncoder(sparse=False)),
])  # categorical pipeline

from sklearn.compose import ColumnTransformer

final_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs),
])

X_train = final_pipeline.fit_transform(train_data[num_attribs + cat_attribs])  # transformed train data
y_train = train_data['Survived']  # Labels
# print(X_train)


## Models: No.1 Random Forest

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)  # first try random forest classifier

from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
print('Random Forest: ', forest_scores.mean())  # ~80% with an random forest decision tree without any data manipulation

## Models No.2 SGD

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)

sgd_scores = cross_val_score(sgd_clf, X_train, y_train, cv=10)
print('Stochastic Gradient Decent: ', sgd_scores.mean())  # ~ 78% with Stochastic Gradient Descent

## Model No. 3 SVC

from sklearn.svm import SVC

svc_clf = SVC(gamma='auto')
svc_clf.fit(X_train, y_train)

svc_scores = cross_val_score(svc_clf, X_train, y_train, cv=10)
print('SVC: ', svc_scores.mean())  # ~ 82% with SCV

## Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

dectree_clf = DecisionTreeClassifier(random_state=42)
dectree_clf.fit(X_train, y_train)

dectree_scores = cross_val_score(dectree_clf, X_train, y_train, cv=10)
print('Decision Tree Classifier: ', dectree_scores.mean())  # ~ 78%


## Logistic Regression

from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
log_clf_scores = cross_val_score(log_clf, X_train, y_train, cv=10)
print('Logistic Regression:', log_clf_scores.mean())


## GridSearch and HyperparameterTuning

from sklearn.model_selection import GridSearchCV

param_grid_forest = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8, 10, 12]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 4, 6, 8]}
]

forest_clf_grid = GridSearchCV(forest_clf, param_grid_forest, cv=5, scoring='neg_mean_squared_error')
forest_clf_grid.fit(X_train, y_train)

print('Best Parameters are:', forest_clf_grid.best_estimator_)

forest_clf_best = RandomForestClassifier(max_features=12, n_estimators=10, random_state=42)
forest_clf_best.fit(X_train, y_train)

forest_clf_best_scores = cross_val_score(forest_clf_best, X_train, y_train, cv=10)
print('Best Random Forest: ', forest_clf_best_scores.mean())  # 81,5% slightly better

param_grid_SVC = [
    {'C': [1, 2, 3, 4, 5, 6], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['auto', 'scale'],
     'shrinking': [False]}
]

svc_clf_grid = GridSearchCV(svc_clf, param_grid_SVC, cv=5, scoring='neg_mean_squared_error')
svc_clf_grid.fit(X_train, y_train)

svc_clf_best = SVC(C=4, gamma='auto', shrinking=False)
svc_clf_best.fit(X_train, y_train)

svc_clf_best_scores = cross_val_score(svc_clf_best, X_train, y_train, cv=10)

print('Best Parameters are:', svc_clf_grid.best_estimator_)
print('Best SVC: ', svc_clf_best_scores.mean())  # 82% no increase


param_grid_LogReg = [
    {'penalty': ['l1', 'l2', 'elasticnet', 'none'], 'C': [1, 2, 3, 4, 5, 6],
     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'max_iter':[200, 2000, 5000]}
]

# log_clf_grid = GridSearchCV(log_clf, param_grid_LogReg, cv=5, scoring='neg_mean_squared_error')
# log_clf_grid.fit(X_train, y_train)

log_clf_best = LogisticRegression(C=1, max_iter=200, random_state=42, solver='newton-cg')
log_clf_best.fit(X_train, y_train)

log_clf_best_scores = cross_val_score(log_clf_best, X_train, y_train, cv=10)

print('Best Logistic Regression:', log_clf_best_scores.mean())


# Conclusion and File Submission

X_test = final_pipeline.fit_transform(test_data[num_attribs + cat_attribs])

predictions = svc_clf_best.predict(X_test)

test_data = pd.read_csv(test_path)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")