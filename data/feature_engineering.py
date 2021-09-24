from ast import literal_eval

import pandas as pd
from sklearn.model_selection import train_test_split

from data.load_data import load_csv


def transform_df(csv_path, save_name=""):
    df = load_csv(csv_path, **{
        'converters': {"current_state": literal_eval, "target_state": literal_eval},
    })
    df.dropna(inplace=True)
    # assuming u, v, omega, theta, x, z . Checked!! this is the way!!
    cols = ["uc", "vc", "omegac", "thetac", "xc", "zc"]
    df[cols] = pd.DataFrame(df["current_state"].to_list(), index=df.index)
    cols = ["uf", "vf", "omegaf", "thetaf", "xf", "zf"]
    df[cols] = pd.DataFrame(df["target_state"].to_list(), index=df.index)

    df["u"] = df["uf"] - df["uc"]
    df["v"] = df["vf"] - df["vc"]
    df["omega"] = df["omegaf"] - df["omegac"]
    df["theta"] = df["thetaf"] - df["thetac"]
    df["x"] = df["xf"] - df["xc"]
    df["z"] = df["zf"] - df["zc"]
    # maintain cols
    cols = ["id_trajectory", "u", "v", "omega", "theta", "x", "z", "action"]
    rm_cols = [c for c in df.columns if c not in cols]
    df.drop(columns=rm_cols, inplace=True)

    categorical_actions = df["action"].astype("category").cat
    df["action_codes"] = categorical_actions.codes

    if not len(save_name):
        save_name = "filtered"
    df.to_csv(save_name + ".csv", index=False)


def split_train_test(csv_path, **train_split_kwargs):
    df = pd.read_csv(csv_path)
    if 'stratify' in train_split_kwargs:
        train_split_kwargs.pop('stratify')
    df_train, df_test = train_test_split(df, **train_split_kwargs, stratify=df['action_codes'])
    df_train.to_csv("landing_train.csv", index=False)
    df_test.to_csv("landing_test.csv", index=False)


def group_by_split(csv_path):
    df = load_csv(csv_path)
    train, test = train_test_split(df.id_trajectory.unique(), random_state=40)

    train_df = df[df['id_trajectory'].isin(train)]
    test_df = df[df['id_trajectory'].isin(test)]

    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)


def check_same_action_codes(df1_path, df2_path):
    df1 = load_csv(df1_path)
    df2 = load_csv(df2_path)

    # getting the unique ('action_codes', 'action') pairs. Then, order by action and reset the index. If index is not
    # the same, then dataframes are considered as different.
    unique_codes_1 = df1[['action_codes', 'action']].drop_duplicates().sort_values(by='action').reset_index(drop='index')
    unique_codes_2 = df2[['action_codes', 'action']].drop_duplicates().sort_values(by='action').reset_index(drop='index')

    eq = unique_codes_1.equals(unique_codes_2)
    print("They are {0}equals".format("" if eq else "not "))
    return eq


def get_avg_trajectory_length(df_path):
    df = pd.read_csv(df_path)
    vc = df.groupby(by='id_trajectory').size()
    print("Avg. size: ", vc.mean())


if __name__ == '__main__':
    problem = "mid_range"
    # group_by_split(problem + "/data.csv")

    # transform_df("{0}/{0}_train.csv".format(problem), save_name="{0}/{0}_train_mlp_format".format(problem))
    # transform_df("{0}/{0}_test.csv".format(problem), save_name="{0}/{0}_test_mlp_format".format(problem))
    check_same_action_codes("{0}/{0}_train_mlp_format.csv".format(problem),
                            "{0}/{0}_test_mlp_format.csv".format(problem))

    # split_train_test("filtered.csv", **{'test_size': .3, 'shuffle': True, 'random_state': 40})
