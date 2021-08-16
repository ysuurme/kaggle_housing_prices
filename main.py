import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score

from functions.functions import *


# Load Data:
fileName_train = 'sourceData/IowaHousing_Train.csv'
df_train = pd.read_csv(fileName_train)
fileName_data = 'sourceData/MelbourneHousing_Data.csv'
df_data = pd.read_csv(fileName_data)

# Understand the Data:
# df_train.describe()
# df_train.info()
# df_data.columns
missing_val_count_by_column(df_train)

# Select Data:
# num_features = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
y = df_train.SalePrice
X = df_train[['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Select numeric columns
num_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

# Select categorical columns
cat_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and
                        X_train[cname].dtype == "object"]

cols = num_cols + cat_cols
X_train = X_train[cols].copy()
X_valid = X_valid[cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

# Define model
gb_model = GradientBoostingClassifier(n_estimators=10)

# Fit model
gb_model.fit(X_train, y_train)

# Predict target
predictions = gb_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))












""" 
Pipelines
"""

# Preprocessing for numerical data
num_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='constant')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features)
#         ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
forest_model = RandomForestRegressor(n_estimators=50, random_state=1)
gb_model = GradientBoostingClassifier()  #todo n_iter_no_change + learning rate + n_jobs(PC Cores)

model = gb_model

forest_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)])

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(forest_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
print(f"Average MAE score (across experiments): {scores.mean()}")

results = {}
n_estimators = [x for x in range(50, 450, 50)]
for n in n_estimators:
    results[n] = get_forest_mae(n)

plt.plot(list(results.keys()), list(results.values()))
plt.show()

model.fit(X_train, y_train)
predictions = gb_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

""" 
The newest house in your data isn't that new. A few potential explanations for this:
a. They haven't built new houses where this data was collected.
b. The data was collected a long time ago. Houses built after the data publication wouldn't show up.
Do these potential explanations affect your trust in the model you build with this data?

How could you dig into the data to see which explanation is more plausible?
"""

# Clean the Data:
df_data_cleaned = df_data.dropna(axis=0)  # Dropping rows for which features have missing values todo Significant Data Loss
df_train_cleaned = df_train.dropna(axis=0, subset=['SalePrice'])
df_train_cleaned = df_train_cleaned.select_dtypes(exclude=['object'])

cols_with_missing = [col for col in df_train_cleaned.columns if df_train_cleaned[col].isnull().any()]
for col in cols_with_missing:
    df_train_cleaned[col] = df_train_cleaned[col].fillna(df_train_cleaned[col].mean())

# Select Data for Modeling:
# features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = df_train_cleaned[features]
y = df_train_cleaned.SalePrice

train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

""""
The steps to building and using a model are:

Specify: Define the type of model that will be used, and the parameters of the model.
Fit: Capture patterns from provided data. This is the heart of modeling.
Predict: Predict the values for the prediction target (y)
Evaluate: Determine how accurate the model's predictions are.
"""

# Specify Model:
housingPrice_modelTree = DecisionTreeRegressor(max_leaf_nodes=52, random_state=1)
housingPrice_modelForest = RandomForestRegressor(random_state=1)

housingPrice_modelForest1 = RandomForestRegressor(n_estimators=50, random_state=0)
housingPrice_modelForest2 = RandomForestRegressor(n_estimators=100, random_state=0)
housingPrice_modelForest3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
housingPrice_modelForest4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
housingPrice_modelForest5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [housingPrice_modelForest1, housingPrice_modelForest2, housingPrice_modelForest3, housingPrice_modelForest4,
          housingPrice_modelForest5]

# Fit Model:
housingPrice_modelTree.fit(train_X, train_y)
housingPrice_modelForest.fit(train_X, train_y)

# Predict:
price_predictTree = housingPrice_modelTree.predict(val_X)
price_predictForest = housingPrice_modelForest.predict(val_X)

df_predict = pd.DataFrame()
df_predict['Price'] = val_y.tolist()
df_predict['PricePredicted_Tree'] = price_predictTree.tolist()
df_predict['PricePredicted_Forest'] = price_predictForest.tolist()

df_predict['Error_Tree'] = df_predict['Price'] - df_predict['PricePredicted_Tree']
df_predict['Error_Tree_Norm'] = np.sqrt(df_predict['Error_Tree']**2)

df_predict['Error_Forest'] = df_predict['Price'] - df_predict['PricePredicted_Forest']
df_predict['Error_Forest_Norm'] = np.sqrt(df_predict['Error_Forest']**2)

# Evaluate Decision Tree:
housingPrice_modelTree_mae = mean_absolute_error(price_predictTree, val_y)
print(f'Housing Price Decision Tree MAE: USD {housingPrice_modelTree_mae:.0f} should equal:'
      f' USD {df_predict.Error_Tree_Norm.mean():.0f}')

leaf_nodes = [2**p for p in range(1, 11)]
print_tree_mae([52], train_X, val_X, train_y, val_y)
node_fit = min_tree_mae(leaf_nodes, train_X, val_X, train_y, val_y)

# Evaluate Random Forest:
housingPrice_modelForest_mae = mean_absolute_error(price_predictForest, val_y)
print(f'Housing Price Random Forest MAE: USD {housingPrice_modelForest_mae:.0f} should equal:'
      f' USD {df_predict.Error_Forest_Norm.mean():.0f}')

for m in models:
    print_model_mae(m, train_X, val_X, train_y, val_y)

# Think critically about the data
newest_home_age = datetime.datetime.now().year - df_train['YearBuilt'].max()  # Example of thinking critically
print(f'The most recent house in the training data is #{newest_home_age:.0f} years of age ({df_train.YearBuilt.max()})')