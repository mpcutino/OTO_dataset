import os
import json
import torch
import argparse
import numpy as np

from ml_algorithms.ann.__internal_main__ import do_training, do_eval
from ml_algorithms.ann.training.plots import plot_acc_history, plot_loss_history
from ml_algorithms.ann.training.data_utils import load_train_val_test

from ml_algorithms.random_forest.validation import main_val
from ml_algorithms.random_forest.grid_search import get_best_params


parser = argparse.ArgumentParser(description='Train and test the classifiers on the landing and mid-range task.')

parser.add_argument("--test", help="define if the model is going to be trained or tested. "
                                   "By default, it will be tested", action="store_true")
parser.add_argument("--epochs", type=int, default=5,
                    help="The number of epochs. If the model is not trained, then this argument is not used.")
parser.add_argument("--mid", action="store_true",
                    help="define the task. By default, it will use the landing data.")
parser.add_argument("--rf", help="define the model to use. Random forest, by default.", action="store_true")


if __name__ == '__main__':
    args = parser.parse_args()

    # ---------------------  PARAMETERS
    problem_ = "mid_range" if args.mid else "landing"
    train_ = False if args.test else True

    train_csv_path_ = "data/{0}/{0}_train_mlp_format.csv".format(problem_)
    test_csv_path_ = "data/{0}/{0}_test_mlp_format.csv".format(problem_)
    action_col_ = "action_codes"
    continuous_var = ['u', 'v', 'omega', 'theta', 'x', 'z']

    # mlp_path = "ml_algorithms/ann/model_results/{0}".format(problem_)
    # rf_path = "ml_algorithms/random_forest/{0}".format(problem_)
    # use_path_model = rf_path if args.rf else mlp_path

    save_path = "models/{0}".format(problem_)
    use_cuda_ = torch.cuda.is_available()
    device_ = torch.device('cuda' if use_cuda_ else 'cpu')
    # -------------------------------------------------------

    print("{0} on {1} problem".format("Training" if train_ else "Testing", problem_))
    if not train_:
        print("Using {0}/ as path for evaluation".format(save_path))
    print()

    if train_:
        if args.rf:
            # Train with Random Forest
            X_train, X_val, X_test, y_train, y_val, y_test, _ = load_train_val_test(train_csv_path_, test_csv_path_,
                                                                                    action_col_, continuous_var)
            # the grid search does cross-validation, so there is no need of splitting on training and validation data
            gs_X = np.concatenate([X_train, X_val])
            gs_y = np.concatenate([y_train, y_val])

            get_best_params(
                {'n_estimators': [50, 100, 150, 200], 'criterion': ['gini', 'entropy'], 'max_depth': [25, None]},
                gs_X, gs_y, save_folder=save_path
            )
        else:
            # Train with MLP
            do_training(train_csv_path_, test_csv_path_, action_col_, continuous_var, device_, use_cuda_,
                        epochs=args.epochs, save_folder=save_path)
    else:
        if args.rf:
            # Eval Random Forest
            brf = os.path.join(save_path, "best_random_forest.rf")
            main_val(brf, train_csv_path_, test_csv_path_, action_col_, continuous_var)
        else:
            # Eval MLP
            # plot history on training and validation
            with open(os.path.join(save_path, "direct_snet_history.json")) as fd:
                hist = json.load(fd)
            plot_acc_history(hist, ['train_acc', 'val_acc'],
                             os.path.join(save_path, "training_accuracy.png"), ['green', 'yellow'])
            plot_loss_history(hist, ['train_loss', 'val_loss'],
                              os.path.join(save_path, "training_loss.png"), ['blue', 'red'])

            # eval on test dataset
            do_eval(save_path, train_csv_path_, test_csv_path_, action_col_, continuous_var)
