import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

def plot_train_test_split(df: pd.DataFrame, split_date: str) -> None:
    """
    Plot the train/test split of the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.
        split_date (str): Date in the format 'YYYY-MM-DD' to split the DataFrame.

    Returns:
        None
    """
    train = df.loc[df.index < split_date]
    test = df.loc[df.index >= split_date]

    fig, ax = plt.subplots(figsize=(15, 5))
    train.plot(ax=ax, label='Training Set', title='Data Train/Test Split', linewidth=1, markersize=5)
    test.plot(ax=ax, label='Test Set', linewidth=1, markersize=5)
    ax.axvline(pd.Timestamp(split_date), color='black', ls='--')
    ax.legend(['Training Set', 'Test Set'])
    plt.show()


# Just want see how a single weekly data looks like
def plot_week_of_data(df: pd.DataFrame, start_date: str, end_date: str) -> None:
    """
    Plot a week of data from the input DataFrame within the specified date range.

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.
        start_date (str): Start date in the format 'YYYY-MM-DD'.
        end_date (str): End date in the format 'YYYY-MM-DD'.

    Returns:
        None
    """
    df.loc[(df.index > start_date) & (df.index < end_date)] \
        .plot(figsize=(15, 5), title='Week Of Data', linewidth=1, markersize=5)
    plt.show()


####### # From DataTime column of the original DF create features for TIme Series  #########

def create_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From DataTime column of the original DF create features for TIme Series

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with additional time series features.

    Raises:
        ValueError: If input DataFrame does not have a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a DatetimeIndex.")

    df = df.copy()

    features = {
        'hour': df.index.hour,
        'dayofweek': df.index.dayofweek,
        'quarter': df.index.quarter,
        'month': df.index.month,
        'year': df.index.year,
        'dayofyear': df.index.dayofyear,
        'dayofmonth': df.index.day,
        'weekofyear': df.index.isocalendar().week
    }
    # df.index.hour: The .hour attribute is used to access the hour component of each datetime in the DatetimeIndex. This returns a new Int64Index object containing the hours.

    for feature_name, feature_values in features.items():
        df[feature_name] = feature_values

    return df


#################################################


def train_xgb_regressor(X_train, y_train, X_test, y_test, use_gpu=False):
    """
    Train an XGBoost Regressor model using the provided training and test data.

    Args:
        X_train: Training feature set.
        y_train: Training target set.
        X_test: Test feature set.
        y_test: Test target set.
        use_gpu (bool): Whether to use GPU for training. Default is False.

    Returns:
        xgb.XGBRegressor: Trained XGBoost Regressor model.
    """
    additional_params = {}
    if use_gpu:
        additional_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}

    xgb_regressor = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                           n_estimators=3000,
                           early_stopping_rounds=50,
                           objective='reg:linear',
                           max_depth=6,
                           learning_rate=0.01,
                           min_child_weight=1,
                           subsample=0.8,
                           colsample_bytree=0.8,
                           gamma=0,
                           reg_alpha=0,
                           reg_lambda=1,
                           **additional_params)
    xgb_regressor.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)


    return xgb_regressor

""" description of each parameter in the xgb.XGBRegressor.

base_score: The initial prediction score for all instances, a global bias. The default value is 0.5. It doesn't have a significant impact on the model's performance, but it can affect the model's convergence speed.

booster: The type of boosting model to use. Choices are 'gbtree', 'gblinear', or 'dart'. The default is 'gbtree'. 'gbtree' and 'dart' use tree-based models, while 'gblinear' uses a linear function. Tree-based models are usually more powerful but may be prone to overfitting, while linear models are simpler and might be suitable for certain problems.

n_estimators: The number of boosting rounds to be run. Increasing the number of rounds can improve the model's performance, but it may also lead to overfitting. It is important to use early_stopping_rounds and cross-validation to find an appropriate value.

early_stopping_rounds: The number of consecutive rounds without a decrease in the evaluation metric before training is stopped. This helps prevent overfitting by stopping the training process when the model's performance starts to degrade on the evaluation set.

objective: The learning task and the corresponding learning objective. In this case, it's 'reg:linear', which means we are solving a regression problem with a linear objective. Other options include 'binary:logistic' for binary classification and 'multi:softmax' for multiclass classification.

max_depth: The maximum depth of a tree. Increasing the depth can capture more complex patterns in the data but may lead to overfitting. It's important to find a good trade-off between model complexity and the risk of overfitting using cross-validation.

learning_rate: Also known as "eta," this parameter controls the step size at each boosting step. Lower values lead to more conservative boosting, but require more boosting rounds to achieve good performance. Higher values can lead to faster convergence but may cause overfitting. It's crucial to find an appropriate balance using cross-validation.

min_child_weight: The minimum sum of instance weights needed in a child node. Higher values result in more conservative tree growth and can help prevent overfitting, while lower values make the algorithm more flexible. Tuning this parameter is useful for controlling the model's complexity.

subsample: The fraction of samples to be used for each boosting round. Lower values can help prevent overfitting by reducing the correlation between the trees in the ensemble. However, very low values may lead to underfitting. Typical values are between 0.5 and 1.

colsample_bytree: The fraction of features to choose for each boosting round. Like subsample, reducing the value can help prevent overfitting, but may lead to underfitting if set too low. Typical values are between 0.5 and 1.

gamma: The minimum loss reduction required to make a split in a tree. Higher values result in more conservative tree growth, while lower values make the algorithm more flexible. Tuning this parameter can help control the model's complexity and prevent overfitting.

reg_alpha: The L1 regularization term on weights. Adding L1 regularization can help prevent overfitting by encouraging sparsity in the model's weights. A higher value results in stronger regularization.

reg_lambda: The L2 regularization term on weights. Adding L2 regularization can help prevent overfitting by penalizing large weight values. A higher value results in stronger regularization.
"""

#######################################################

def plot_feature_importance(model):
    """
    Plot the feature importance of a trained XGBoost model.

    Args:
        model (xgb.XGBRegressor): Trained XGBoost Regressor model.

    Returns:
        None
    """
    feat_importances = pd.DataFrame(data=model.feature_importances_,
                      index=model.feature_names_in_,
                      columns=['importance'])
    feat_importances.sort_values('importance').plot(kind='barh', title='Feature Importance')
    plt.show()

""" The `model.feature_importances_` attribute of the trained XGBoost model contains the feature importances as an array. These importances are computed as the average gain of the feature when it is used in trees. The higher the importance value, the more important the feature is in making predictions.

index=model.feature_names_in_: The `model.feature_names_in_` attribute of the trained XGBoost model contains the names of the features that were used for training. This is assigned to the index of the new DataFrame, which means that each row in the DataFrame will correspond to a feature name.

columns=['importance']: This parameter assigns the name 'importance' to the single column of the DataFrame, which will hold the feature importances.

After this line is executed, `feat_importances` is a DataFrame containing the feature names as the index and their corresponding importances in the 'importance' column. """