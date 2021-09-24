import time
import pickle
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.metrics import classification_report

from data.load_data import load_csv

from ospa.flight_state import FlightState

from ml_algorithms.ann.validation.utils import get_action_codes
from ml_algorithms.ann.training.data_utils import load_train_val_test

from ospa.constants import tc
from ospa.utils import get_next_state
from ospa.flight_state import fs2dimensional, fs2Adimensional


def get_rf_values(model_path, mlp_train_data_path, mlp_test_data_path, test_data_path_full_format):
    action_col_ = "action_codes"
    continuous_var = ['u', 'v', 'omega', 'theta', 'x', 'z']

    action_codes_dict = get_action_codes(mlp_test_data_path)
    full_format_df = load_csv(test_data_path_full_format,
                              **{'converters': {"current_state": literal_eval, "target_state": literal_eval}})

    gb_df = full_format_df.groupby(by='id_trajectory')
    # get only the first value in the trajectory
    aux_df = gb_df.agg({c: lambda x: x.iloc[0] for c in full_format_df.columns})

    out = load_train_val_test(mlp_train_data_path, mlp_test_data_path, action_col_, continuous_var)
    scaler = out[-1]

    with open(model_path, 'rb') as fd:
        model = pickle.load(fd)

    dict_as_ospa = {
        'current_state': [], 'target_state': [], 'incremental_cost': [], 'id_trajectory': [], 'id_in_seq': []
    }
    for _, r in aux_df.iterrows():
        cs_fs = FlightState.from_jint_data(*r.current_state)
        ts_fs = FlightState.from_jint_data(*r.target_state)

        path = do_one_path(model, cs_fs, ts_fs, action_codes_dict, r.timestep, scaler=scaler)
        for i, s in enumerate(path):
            dict_as_ospa['current_state'].append([s.u, s.v, s.omega, s.theta, s.x, s.z])
            dict_as_ospa['target_state'].append(r.target_state)
            dict_as_ospa['incremental_cost'].append(s.cost)
            dict_as_ospa['id_trajectory'].append(r.id_trajectory)
            dict_as_ospa['id_in_seq'].append(i)

    pd.DataFrame(dict_as_ospa).to_csv("rf_output_data.csv", index=False)


def do_one_path(rf_model, cs_fs, ts_fs, action_codes_dict, timestep,
                scaler=None, sigma_noise=None, noise_on_target=False, noise_from=5):
    path = []
    input_fs = ts_fs.minus(cs_fs)
    # the input is the difference. Loop while the difference on the x-value is bigger than 0 or the model says stop
    current_traj_point = 0
    while input_fs.x > 0:
        path.append(cs_fs)
        if sigma_noise and noise_on_target and current_traj_point >= noise_from:
            new_v = {}
            for n in sigma_noise:
                new_v[n] = np.random.normal(0, sigma_noise[n], 1)
            noise_applied = ts_fs.sigma_noise(new_v, x_restriction=cs_fs.x)
            if noise_applied:
                # the noise on the target is only added once
                sigma_noise = None
        input = np.array([[input_fs.u, input_fs.v, input_fs.omega, input_fs.theta, input_fs.x, input_fs.z]], dtype=object)
        input = scaler.transform(input) if scaler else input
        if sigma_noise and not noise_on_target:
            noise = np.random.normal(0, sigma_noise, len(input[0]))
            # as input is the difference among the states,
            # the noise added to the current state result in a subtraction of the current input
            input -= noise

        action_code = rf_model.predict(input)
        action = list(action_codes_dict[action_code[0]]) + [timestep/tc]
        action[1] = action[1]*tc

        # the input of the ospa model is a-dimensional
        cs_fs = get_next_state(fs2Adimensional(cs_fs), action, fs2Adimensional(ts_fs))
        # ERROR in the model
        stop = type(cs_fs) is int
        if stop: break

        # the input of the learning algorithm is dimensional, so it must be transformed back
        cs_fs = fs2dimensional(cs_fs)
        input_fs = ts_fs.minus(cs_fs)
        current_traj_point += 1
    return path


def get_sigma_metrics(model_path, mlp_train_data_path, mlp_test_data_path, test_data_path_full_format,
                      action_col, continuous_var, sigma_values, samples=500, sigma_on_target=False):

    action_codes_dict = get_action_codes(mlp_test_data_path)
    full_format_df = load_csv(test_data_path_full_format,
                              **{'converters': {"current_state": literal_eval, "target_state": literal_eval}})

    gb_df = full_format_df.groupby(by='id_trajectory')
    # get only the first value in the trajectory
    aux_df = gb_df.agg({c: lambda x: x.iloc[0] for c in full_format_df.columns})
    aux_df = aux_df.sample(samples)

    out = load_train_val_test(mlp_train_data_path, mlp_test_data_path, action_col, continuous_var)
    scaler = out[-1]

    with open(model_path, 'rb') as fd:
        model = pickle.load(fd)

    sigma_list = []
    for _, r in aux_df.iterrows():
        for s_index, sigma in enumerate(sigma_values):
            # leave it here, because target value can be changed on the path computation
            cs_fs = FlightState.from_jint_data(*r.current_state)
            ts_fs = FlightState.from_jint_data(*r.target_state)
            sigma_list.append(
                {
                    'current_state': [], 'target_state': [], 'incremental_cost': [], 'id_trajectory': [], 'id_in_seq': []
                })
            if sigma_on_target:
                # x position and z position on the mean array
                sigma = {'x': abs(sigma*scaler.mean_[-2]), 'z': abs(sigma*scaler.mean_[-1])}
            path = do_one_path(model, cs_fs, ts_fs, action_codes_dict, r.timestep,
                               scaler=scaler, sigma_noise=sigma, noise_on_target=sigma_on_target)
            for i, s in enumerate(path):
                sigma_list[s_index]['current_state'].append([s.u, s.v, s.omega, s.theta, s.x, s.z])
                sigma_list[s_index]['target_state'].append(r.target_state)
                sigma_list[s_index]['incremental_cost'].append(s.cost)
                sigma_list[s_index]['id_trajectory'].append(r.id_trajectory)
                sigma_list[s_index]['id_in_seq'].append(i)

    for sigma, data in zip(sigma_values, sigma_list):
        pd.DataFrame(data).to_csv("rf_sigma{0}_output_data.csv".format(sigma), index=False)


def main_val(filename, train_csv_path, test_csv_path, action_col, continuous_var):
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_train_val_test(train_csv_path, test_csv_path,
                                                                            action_col, continuous_var)
    gs_X = np.concatenate([X_train, X_val])
    gs_y = np.concatenate([y_train, y_val])

    with open(filename, 'rb') as fd:
        loaded_model = pickle.load(fd)
    start = time.time()
    result = loaded_model.score(X_test, y_test)
    end = time.time()
    print("Testing score {0}".format(result))
    print("Evaluated {0} items in {1:.4f}s".format(len(X_test), end-start))

    result = loaded_model.score(gs_X, gs_y)
    print("Full training dataset score {0}".format(result))

    print('Params: {0}'.format(loaded_model.get_params()))

    print(classification_report(y_test, loaded_model.predict(X_test)))


if __name__ == '__main__':
    #
    continuous_var_ = ['u', 'v', 'omega', 'theta', 'x', 'z']
    action_col_ = "action_codes"
    sigma_vals_ = [.01, .02, .05, .1, .2, .5, 1]
    sigma_on_target_ = True

    problem_ = "mid_range"
    train_csv_path_ = "../../data/{0}/{0}_train_mlp_format.csv".format(problem_)
    test_csv_path_ = "../../data/{0}/{0}_test_mlp_format.csv".format(problem_)
    test_data_path_full_format_ = "../../data/{0}/{0}_test.csv".format(problem_)
    #

    # main_val("best_random_forest.rf", train_csv_path_, test_csv_path_, action_col_, continuous_var_)
    # get_sigma_metrics("best_random_forest.rf", train_csv_path_, test_csv_path_, test_data_path_full_format_,
    #                   action_col_, continuous_var_, sigma_vals_, sigma_on_target=sigma_on_target_)

    get_rf_values(
        "best_random_forest.rf",
        train_csv_path_,
        test_csv_path_,
        test_data_path_full_format_
    )
