import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.load_data import load_csv


def get_desired_vars(df: pd.DataFrame, state_column, action_column):
    states = np.array([np.array(r) for r in df[state_column].values])
    categorical_actions = df[action_column].astype("category").cat
    actions = categorical_actions.codes.values
    action_categories = categorical_actions.categories.values

    return {'states': states, 'actions': actions, 'categories': action_categories}


def load_train_val_test(train_csv_path, test_csv_path, action_col, continuous_var,
                        random_state=40, test_size=.2, standard_scale=True):
    train_df = load_csv(train_csv_path)
    test_df = load_csv(test_csv_path)

    X_test, y_test = test_df[continuous_var].values, test_df[action_col].values
    X_train, X_val, y_train, y_val = train_test_split(train_df[continuous_var].values, train_df[action_col].values,
                                                      shuffle=True, test_size=test_size, stratify=train_df[action_col],
                                                      random_state=random_state)

    if standard_scale:
        st_scaler = StandardScaler()
        st_scaler.fit(train_df[continuous_var])
        X_train = st_scaler.transform(X_train)
        X_val = st_scaler.transform(X_val)
        X_test = st_scaler.transform(X_test)
    else:
        st_scaler = None

    return X_train, X_val, X_test, y_train, y_val, y_test, st_scaler
