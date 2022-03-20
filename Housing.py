import os
import tarfile
from six.moves import urllib
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = os.path.join('datasets', 'housing')
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'


## Downloading and saving the data


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


fetch_housing_data()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


housing = load_housing_data()

# housing.hist(bins=50, figsize=(20,15))
# plt.show()

## Creating a test set

housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)  # stratification for median_income, see corr matrix
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)  # >50k median_income category 5

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # dividing the data via income_cat and
# 80-20 relation
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)  # deleting the artifical attribute income cat

## Visualization

housing = strat_train_set.copy()

housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1, s=housing['population'] / 100, label='population',
             figsize=(10, 7),
             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
# plt.show()


## Correlations

# corr_matrix = housing.corr()
# print(corr_matrix['median_house_value'].sort_values(ascending=False))

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
pd.plotting.scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

## Manipulations

housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']

corr_matrix = housing.corr()
# print(corr_matrix['median_house_value'].sort_values(ascending=False))

## Prep

housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()  # Labeling

imputer = SimpleImputer(strategy='median')  # Replacing missing values in bedrooms
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

housing_cat = housing['ocean_proximity']  # Factorize the categorical attribute
housing_cat_encoded, housing_categories = housing_cat.factorize()
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):  # does adding bedrooms to the learning process help?
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, bedrooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

## Pipeline

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


class DataFrameSelector(BaseEstimator, TransformerMixin):  # Reformates pd-DataFrames into Numpy
    def __init__(self, attribute_names):
        self.attributes_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attributes_names].values


num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),  # PD-Dataframes
    ('imputer', SimpleImputer(strategy='median')),  # Handling missing values
    ('attribs_adder', CombinedAttributesAdder()),  # Attribute Adder (see above)
    ('std_scaler', StandardScaler()),  # Scaling the data
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),  # see numeric pipeline
    ('cat_encoder', OneHotEncoder(categories='auto', sparse=False))  # Encoder for categorical data
])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])  # Full pipeline

housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared.shape)

# Model Selection and Training


## Linear Regression


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
# print('Predictions: ', lin_reg.predict(some_data_prepared))
# print('Labels: ', list(some_labels))

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
# print(round(lin_mse), round(lin_rmse))

## Decision Tree

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)

# Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    # print('Scores: ', scores)
    print('Mean ', scores.mean())
    print('Standard deviation: ', scores.std())


display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)

# Random Forest

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=2)
forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)

import joblib

joblib.dump(forest_reg, 'forest_reg.pkl')
joblib.dump(tree_reg, 'tree_reg.pkl')
joblib.dump(lin_reg, 'lin_reg.pkl')

# Grid Search for Random Forest, selecting best hyperparameters

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_)  # gives best hyperparameters in a give interval, see param grid

print(grid_search.best_estimator_)

cvres = grid_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)  # printing out directly with each iteration of cv search

# Ensemble Methods, Analyzing Errors

feature_importance = grid_search.best_estimator_.feature_importances_  # gives array of relative importance of each attribute
# print(feature_importance)

extra_attribs = ['rooms_per_household', 'pop_per_household', 'bedrooms_per_room']
cat_encoder = cat_pipeline.named_steps['cat_encoder']
cat_one_hot_attribs = list(cat_encoder.categories[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
attrib_list = sorted(zip(feature_importance, attributes), reverse=True)
print(attrib_list)

#  returns a sorted list of each atribute per name and its relative importance in the random tree


# Evaluaion on the Test Set

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)  # Pipelining test Set

final_predictions = final_model.predict(X_test_prepared)  # Predictions by Random Forest

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)  # Evaluattes RMSE on the Test Set with CV-Hyperparameter Trained Random Forest
