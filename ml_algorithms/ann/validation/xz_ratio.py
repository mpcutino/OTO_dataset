import numpy as np
import pandas as pd

from ast import literal_eval

from data.load_data import load_csv


def get_ratio(x, z, valid_ratios):
    vr = sorted(valid_ratios)
    r = abs(z/x)
    for i in vr[::-1]:
        if i < r:
            return i
    return None


def get_precision_per_ratio(data_path, ratio_step=0.2):
    df = load_csv(data_path, **{
        'converters': {"current_state": literal_eval, "target_state": literal_eval},
    })

    gb_df = df.groupby(by='id_trajectory', dropna=True)
    print("Number of trajectories: {0}".format(len(gb_df)))

    valid_ratios = np.arange(0, 1, ratio_step)
    ratio_dict = {i: {'precision': [], 'cost': []} for i in valid_ratios}
    # we are only interested on the final values
    aux = gb_df.agg({'current_state': lambda x: x.iloc[-1],
                     'target_state': lambda x: x.iloc[-1],
                     'incremental_cost': lambda x: x.iloc[-2] if len(x) >= 2 else x})
    for _, r in aux.iterrows():
        ts = r.target_state
        x_target, z_target = ts[-2], ts[-1]
        ratio = get_ratio(x_target, z_target, valid_ratios)

        cs = r.current_state
        x_final, z_final = cs[-2], cs[-1]
        precision = ((x_target - x_final) ** 2 + (z_target - z_final) ** 2) ** .5
        ratio_dict[ratio]['precision'].append(precision)
        ratio_dict[ratio]['cost'].append(r.incremental_cost)

    for r in ratio_dict:
        p_data = ratio_dict[r]['precision']
        avg_precision = sum(p_data)/float(len(p_data))

        cost_data = ratio_dict[r]['cost']
        avg_cost = sum(cost_data) / float(len(cost_data))

        print("Ratio {0} avg. precision: {1}, avg. cost: {2}".format(r, avg_precision, avg_cost))


def compute_error(model_data_path, ospa_data_path, points=10, ratio_step=0.2):
    model_df = load_csv(model_data_path,
                        **{'converters': {"current_state": literal_eval, "target_state": literal_eval}})
    ospa_df = load_csv(ospa_data_path,
                       **{'converters': {"current_state": literal_eval, "target_state": literal_eval}})

    valid_ratios = np.arange(0, 1, ratio_step)
    ratio_dict = {i: {'error': 0, 'count': 0} for i in valid_ratios}
    for id_t in ospa_df.id_trajectory.unique():
        ospa_subset = ospa_df[ospa_df['id_trajectory'] == id_t]
        model_subset = model_df[model_df['id_trajectory'] == id_t]

        if model_subset.empty:
            print("Not model trajectory for {0}".format(id_t))
            continue

        ts = ospa_subset.iloc[-1].target_state
        x_target, z_target = ts[-2], ts[-1]
        ratio = get_ratio(x_target, z_target, valid_ratios)

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
        ratio_dict[ratio]['error'] += path_dif
        ratio_dict[ratio]['count'] += points

    for r in ratio_dict:
        avg_error = ratio_dict[r]['error']/float(ratio_dict[r]['count'])

        print("Ratio {0} avg. error: {1}".format(r, avg_error))


if __name__ == '__main__':
    problem_ = "mid_range"
    # df_data = "../../random_forest/rf_output_data.csv"
    df_data = "model_output_data.csv"
    test_data_path_full_format_ = "../../../data/{0}/{0}_test.csv".format(problem_)

    # get_precision_per_ratio(df_data)

    compute_error(df_data, test_data_path_full_format_)
