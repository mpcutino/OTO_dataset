from ast import literal_eval

import numpy as np
from torch import FloatTensor

from data.load_data import load_csv

from ospa.constants import tc
from ospa.utils import get_next_state
from ospa.flight_state import fs2dimensional, fs2Adimensional


def get_action_codes(mlp_test_data_path):
    df = load_csv(mlp_test_data_path, **{'converters': {"action": literal_eval}})
    codes_df = df[['action_codes', 'action']].drop_duplicates().sort_values(by='action').reset_index(drop='index')

    res = {r.action_codes: r.action for _, r in codes_df.iterrows()}
    return res


def get_new_input(ts_fs, cs_fs):
    return ts_fs.minus(cs_fs)


def do_one_path(model, cs_fs, ts_fs, action_codes_dict, timestep,
                scaler=None, device='cpu', sigma_noise=None, noise_on_target=False, noise_from=5):
    path = []
    input_fs = get_new_input(ts_fs, cs_fs)
    # the input is the difference. Loop while the difference on the x-value is bigger than 0 or the model says stop
    current_traj_point = 0
    while input_fs.x > 0:
        path.append(cs_fs)
        input = [input_fs.u, input_fs.v, input_fs.omega, input_fs.theta, input_fs.x, input_fs.z]
        input = scaler.transform([input]) if scaler else np.array([input])
        if sigma_noise and not noise_on_target:
            noise = np.random.normal(0, sigma_noise, len(input[0]))
            # as input is the difference among the states,
            # the noise added to the current state result in a subtraction of the current input
            input -= noise

        action_code = model(FloatTensor(input).to(device)).argmax(dim=1)
        action = list(action_codes_dict[action_code.item()]) + [timestep/tc]
        action[1] = action[1]*tc

        # the input of the ospa model is a-dimensional
        cs_fs = get_next_state(fs2Adimensional(cs_fs), action, fs2Adimensional(ts_fs))
        # ERROR in the model
        stop = type(cs_fs) is int
        if stop: break

        # the input of the learning algorithm is dimensional, so it must be transformed back
        cs_fs = fs2dimensional(cs_fs)
        if sigma_noise and noise_on_target and current_traj_point >= noise_from:
            new_v = {}
            for n in sigma_noise:
                new_v[n] = np.random.normal(0, sigma_noise[n], 1)
            noise_applied = ts_fs.sigma_noise(new_v, x_restriction=cs_fs.x)
            if noise_applied:
                # the noise on the target is only added once
                sigma_noise = None
        input_fs = get_new_input(ts_fs, cs_fs)
        current_traj_point += 1
    return path
