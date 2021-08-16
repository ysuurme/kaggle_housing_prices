from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def missing_val_count_by_column(df):
    print(f'DataFrame shape: {df.shape}', end='\n\n')
    col_missingValues = (df.isnull().sum())
    print(col_missingValues[col_missingValues > 0])

def print_tree_mae(leaf_nodes, train_X, val_X, train_y, val_y):
    """Returns the Mean Absolute Error (MAE) of a Decision Tree for given nodes and training + validation data"""
    for node in leaf_nodes:
        model = DecisionTreeRegressor(max_leaf_nodes=node, random_state=1)
        model.fit(train_X, train_y)
        preds_val = model.predict(val_X)
        mae = mean_absolute_error(val_y, preds_val)
        print(f'Node: {node} - MAE: {mae:.0f}')


def min_tree_mae(leaf_nodes, train_X, val_X, train_y, val_y):
    """Returns the node yielding the minimal Mean Absolute Error (MAE) for given training and validation data"""
    dict_mae = {}
    for node in leaf_nodes:
        model = DecisionTreeRegressor(max_leaf_nodes=node, random_state=1)
        model.fit(train_X, train_y)
        preds_val = model.predict(val_X)
        mae = mean_absolute_error(val_y, preds_val)
        dict_mae[node] = mae
    node = min(dict_mae, key=dict_mae.get)
    print(f'Best Fit Node modelTree: {node} - MAE: {dict_mae[node]:.0f}')
    return node

def print_model_mae(model, train_X, val_X, train_y, val_y):
    """Returns the Mean Absolute Error (MAE) for given model and training + validation data"""
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(preds_val, val_y)
    print(f'Model Mean Average Error (MAE): {mae:.0f}')
    return mae


def get_forest_mae(n_estimators):
    """Return the average MAE over 3 CV folds of random forest model.

    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators, random_state=0))
    ])
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
    return scores.mean()
