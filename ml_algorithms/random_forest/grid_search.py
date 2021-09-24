import os
import pickle
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from ml_algorithms.ann.training.data_utils import load_train_val_test


def get_best_params(params, x_data, y_data, save_folder=""):
    print("Starting search...\n")

    gs = GridSearchCV(RandomForestClassifier(), params)
    gs.fit(x_data, y_data)

    print("Best params: {0}".format(gs.best_params_))
    print("Best score: {0}".format(gs.best_score_))

    model_name = os.path.join(save_folder, "best_random_forest.rf")
    with open(model_name, 'wb') as fd:
        pickle.dump(gs.best_estimator_, fd)
    print("Model saves as {0}".format(model_name))


def main():
    #
    problem_ = "mid_range"
    continuous_var = ['u', 'v', 'omega', 'theta', 'x', 'z']
    action_col = "action_codes"

    # train_csv_path_ = "../../data/landing_train_mlp_format.csv"
    train_csv_path_ = "../../data/{0}/{0}_train_mlp_format.csv".format(problem_)
    # test_csv_path_ = "../../data/landing_test_mlp_format.csv"
    test_csv_path_ = "../../data/{0}/{0}_test_mlp_format.csv".format(problem_)

    #
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_train_val_test(train_csv_path_, test_csv_path_,
                                                                            action_col, continuous_var)
    # the grid search does cross-validation, so there is no need of splitting on training and validation data
    gs_X = np.concatenate([X_train, X_val])
    gs_y = np.concatenate([y_train, y_val])

    get_best_params(
        {'n_estimators': [50, 100, 150, 200], 'criterion': ['gini', 'entropy'], 'max_depth': [25, None]},
        gs_X, gs_y
    )


if __name__ == '__main__':
    main()
