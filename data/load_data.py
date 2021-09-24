import pandas as pd


def landing_data(**kwargs):
    return load_csv("data/OSPA_landing_training-test_data.csv", **kwargs)


def load_csv(p, **kwargs):
    df = pd.read_csv(p, **kwargs)
    c = [substitute_space(i, '_') for i in df.columns]
    df.columns = c
    return df


def substitute_space(name_, char_):
    return char_.join(name_.split())


def load_xlsx(p, sheet_name,**kwargs):
    df = pd.read_excel(p, sheet_name=sheet_name, **kwargs)
    bad_cols = [x.startswith("Unnamed") for x in df.columns]
    if any(bad_cols):
        df.drop(columns=df.columns[bad_cols], inplace=True)
    return df


def reformulate_xlsx(p, sheet_name, new_name="data.csv", **kwargs):
    df = load_xlsx(p, sheet_name, **kwargs)
    print(len(df))
    df['action'] = df.apply(lambda x: None if x.name == len(df) - 1 else df.loc[x.name + 1].out_action, axis=1)
    df['incremental_cost'] = df.apply(lambda x: 0 if x.name == len(df) - 1 else df.loc[x.name + 1]['cost(W)'], axis=1)
    # get the cost per action. After moving one place the incremental cost, the difference will get us the action cost,
    # except for the last action, with a negative cost that must be set to 0.
    df['action_cost'] = df['incremental_cost'] - df['cost(W)']
    # # This will show the SettingWithCopyWarning
    # # but the frame values will be set
    # df['action_cost'][df['action_cost'] < 0] = 0
    df.loc[df['action_cost'] < 0, 'action_cost'] = 0

    df.drop(columns=["initial_state", "out_action", "cost(W)"], inplace=True)
    df.to_csv(new_name, index=False)


if __name__ == '__main__':
    # df_ = load_xlsx("outputs_concat.xlsx", "Sheet1")
    # print(df_.columns)
    problem = "mid_range"

    reformulate_xlsx(problem + "/mid_range_data.xlsx", "Sheet1", new_name= problem + "/data.csv")
