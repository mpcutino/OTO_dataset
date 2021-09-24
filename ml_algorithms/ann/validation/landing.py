from ast import literal_eval

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from data.load_data import load_csv

from ospa.flight_state import FlightState

from ml_algorithms.ann.structures.dummy import DirectSNet
from ml_algorithms.ann.training.data_utils import load_train_val_test
from ml_algorithms.ann.validation.utils import get_action_codes, do_one_path


METRICS = ['Cost', 'Time', 'Precision', 'Error', 'V-error', 'P-error']

def get_OSPA_values(ospa_df_path):
    ospa_df = load_csv(ospa_df_path, **{
        'converters': {"current_state": literal_eval, "target_state": literal_eval},
    })
    gb_df = ospa_df.groupby(by='id_trajectory', dropna=True)
    print("Number of trajectories: {0}".format(len(gb_df)))

    if 'incremental_cost' in ospa_df.columns:
        # because we drop na, the last value in the incremental_cost has the total cost of the path.
        # Then we perform a simple aveage to get the average cost per trajectory.
        aux = gb_df.agg({'incremental_cost': lambda x: x.iloc[-2] if len(x) >= 2 else x})['incremental_cost']
        avg_cost = sum(aux)/len(aux)

        print("Avg. cost: {0}".format(avg_cost))

    if 'target_state' in ospa_df.columns and 'current_state' in ospa_df.columns:
        aux = gb_df.agg({'current_state': lambda x: x.iloc[-1], 'target_state': lambda x: x.iloc[-1]})
        cols = ["uc", "vc", "omegac", "thetac", "xc", "zc"]
        aux[cols] = pd.DataFrame(aux["current_state"].to_list(), index=aux.index)
        cols = ["uf", "vf", "omegaf", "thetaf", "xf", "zf"]
        aux[cols] = pd.DataFrame(aux["target_state"].to_list(), index=aux.index)

        aux["u"] = aux["uf"] - aux["uc"]
        aux["v"] = aux["vf"] - aux["vc"]
        aux["omega"] = aux["omegaf"] - aux["omegac"]
        aux["theta"] = aux["thetaf"] - aux["thetac"]
        aux["x"] = aux["xf"] - aux["xc"]
        aux["z"] = aux["zf"] - aux["zc"]

        avg_precision = sum((aux['x']**2 + aux['z']**2)**.5)/len(aux)
        avg_v_error = sum((aux['u']**2 + aux['v']**2)**.5)/len(aux)
        avg_p_error = sum(aux['theta'])/len(aux)

        print("Avg. precision: {0}, Avg. v-error: {1}, Avg. p-error: {2}".format(
            avg_precision, avg_v_error, avg_p_error))


def get_directsnet_values(model_folder, mlp_train_data_path, mlp_test_data_path, test_data_path_full_format):
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

    model = DirectSNet.load_model(model_folder)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model.to(device)

    dict_as_ospa = {
        'current_state': [], 'target_state': [], 'incremental_cost': [], 'id_trajectory': [], 'id_in_seq': []
    }
    for _, r in aux_df.iterrows():
        cs_fs = FlightState.from_jint_data(*r.current_state)
        ts_fs = FlightState.from_jint_data(*r.target_state)

        path = do_one_path(model, cs_fs, ts_fs, action_codes_dict, r.timestep, scaler=scaler, device=device)
        for i, s in enumerate(path):
            dict_as_ospa['current_state'].append([s.u, s.v, s.omega, s.theta, s.x, s.z])
            dict_as_ospa['target_state'].append(r.target_state)
            dict_as_ospa['incremental_cost'].append(s.cost)
            dict_as_ospa['id_trajectory'].append(r.id_trajectory)
            dict_as_ospa['id_in_seq'].append(i)

    pd.DataFrame(dict_as_ospa).to_csv("model_output_data.csv", index=False)


def get_sigma_metrics(model_folder, mlp_train_data_path, mlp_test_data_path, test_data_path_full_format,
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

    model = DirectSNet.load_model(model_folder)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model.to(device)

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
                               scaler=scaler, device=device, sigma_noise=sigma, noise_on_target=sigma_on_target)
            for i, s in enumerate(path):
                sigma_list[s_index]['current_state'].append([s.u, s.v, s.omega, s.theta, s.x, s.z])
                sigma_list[s_index]['target_state'].append(r.target_state)
                sigma_list[s_index]['incremental_cost'].append(s.cost)
                sigma_list[s_index]['id_trajectory'].append(r.id_trajectory)
                sigma_list[s_index]['id_in_seq'].append(i)

    for sigma, data in zip(sigma_values, sigma_list):
        pd.DataFrame(data).to_csv("direct_snet_sigma{0}_output_data.csv".format(sigma), index=False)


def compute_error(model_data_path, ospa_data_path, points=10):
    model_df = load_csv(model_data_path,
                        **{'converters': {"current_state": literal_eval, "target_state": literal_eval}})
    ospa_df = load_csv(ospa_data_path,
                       **{'converters': {"current_state": literal_eval, "target_state": literal_eval}})

    difference, counts = 0, 0
    for id_t in ospa_df.id_trajectory.unique():
        ospa_subset = ospa_df[ospa_df['id_trajectory'] == id_t]
        model_subset = model_df[model_df['id_trajectory'] == id_t]

        if model_subset.empty:
            print("Not model trajectory for {0}".format(id_t))
            continue

        cols = ["u", "v", "omega", "theta", "x", "z"]
        ospa_path = pd.DataFrame(ospa_subset["current_state"].to_list(), columns=cols)
        model_path = pd.DataFrame(model_subset["current_state"].to_list(), columns=cols)

        points = min([points, len(ospa_path), len(model_path)])

        sampler, follower = (ospa_path, model_path) if len(ospa_path) < len(model_path) else (model_path, ospa_path)

        # random sample of points
        ospa_sample = sampler.sample(points)
        # get the follower points according to the sampler sample
        model_sample = follower.loc[ospa_sample.index]

        path_dif = sum(((model_sample['z'] - ospa_sample['z'])**2 + (model_sample['x'] - ospa_sample['x'])**2)**.5)
        difference += path_dif
        counts += points
    print("Avg. difference between ospa and the model ", difference/counts)


def classification_report_like_sklearn(model_folder, train_csv_path, test_csv_path):

    continuous_var = ['u', 'v', 'omega', 'theta', 'x', 'z']
    action_col = "action_codes"
    #
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_train_val_test(train_csv_path, test_csv_path,
                                                                            action_col, continuous_var)

    model = DirectSNet.load_model(model_folder)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model.to(device)
    X_test = torch.from_numpy(X_test).float()

    preds = []
    for m_input in X_test:
        y_pred = model(m_input.to(device).view(1, -1))
        action = y_pred.argmax(dim=1).cpu().item()
        preds.append(action)
    print(classification_report(y_test, np.array(preds)))


if __name__ == '__main__':
    #
    m_folder_ = "../"
    problem_ = "mid_range"
    sigma_vals_ = [.01, .02, .05, .1, .2, .5, 1]

    continuous_var_ = ['u', 'v', 'omega', 'theta', 'x', 'z']
    action_col_ = "action_codes"
    # train_csv_path_ = "../../../data/landing_train_mlp_format.csv"
    # test_csv_path_ = "../../../data/landing_test_mlp_format.csv"
    # test_data_path_full_format_ = "../../../data/landing_test.csv"

    train_csv_path_ = "../../../data/{0}/{0}_train_mlp_format.csv".format(problem_)
    test_csv_path_ = "../../../data/{0}/{0}_test_mlp_format.csv".format(problem_)
    test_data_path_full_format_ = "../../../data/{0}/{0}_test.csv".format(problem_)
    sigma_on_target_ = True
    #
    #

    # get_OSPA_values(test_data_path_full_format_)
    # get_OSPA_values("model_output_data.csv")
    get_OSPA_values("../../random_forest/rf_output_data.csv")

    # for s in sigma_vals_:
    #     print("\n\n++++ SIGMA {0} +++++\n".format(s))
    #     # get_OSPA_values("../../random_forest/rf_sigma{0}_output_data.csv".format(s))
    #     get_OSPA_values("direct_snet_sigma{0}_output_data.csv".format(s))

    # get_sigma_metrics(m_folder_, train_csv_path_, test_csv_path_, test_data_path_full_format_,
    #                   action_col_, continuous_var_, sigma_vals_, sigma_on_target=sigma_on_target_)

    # get_directsnet_values(
    #     m_folder_,
    #     train_csv_path_,
    #     test_csv_path_,
    #     test_data_path_full_format_
    # )
    # classification_report_like_sklearn(m_folder_, train_csv_path_, test_csv_path_)

    # compute_error("model_output_data.csv", test_data_path_full_format_)
    # compute_error("../../random_forest/rf_output_data.csv", test_data_path_full_format_)
