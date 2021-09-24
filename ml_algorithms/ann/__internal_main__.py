import os
import json
import torch
import numpy as np
import pandas as pd

from ml_algorithms.ann.training.methods import standard
from ml_algorithms.ann.training.data_utils import load_train_val_test

from ml_algorithms.ann.loaders.dataset import StandardDS
from ml_algorithms.ann.loaders.function import load_standard_data

from ml_algorithms.ann.structures.dummy import DirectSNet


def do_training(train_csv_path_, test_csv_path_, action_col_, continuous_var, device_, use_cuda_,
                epochs=300, save_folder=""):
    # Additional Info when using cuda
    if device_.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
    else:
        print("Not using cuda")

    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_train_val_test(train_csv_path_, test_csv_path_,
                                                                            action_col_, continuous_var)

    print("Training samples: {0}".format(X_train.shape))

    loaders_dict = {
        'train': load_standard_data(
            StandardDS(X_train, y_train), use_cuda_, device_, batch_size=128, shuffle=True
        ),
        'val': load_standard_data(
            StandardDS(X_val, y_val), use_cuda_, device_, batch_size=128
        )
    }
    phases = list(loaders_dict.keys())
    m = DirectSNet(X_train.shape[1], len(np.unique(y_train)))
    m.to(device_)

    vc = pd.Series(y_train).value_counts()
    loss_w = torch.FloatTensor((1 - vc / vc.sum()) ** 5).to(device_)
    m, history = standard(
        epochs, m, lambda phase: loaders_dict[phase](),
        torch.optim.RMSprop(m.parameters()),
        torch.nn.CrossEntropyLoss(weight=loss_w),
        accuracy_function=lambda y_hat, y: (y_hat.argmax(dim=1) == y).sum().float().item(),
        print_every=5,
        phases=phases
    )
    m.save_model(save_folder=save_folder)

    h_file = os.path.join(save_folder, "{0}_history.json".format(m.name()))
    with open(h_file, "w") as fd:
        json.dump(history, fd)


def do_eval(model_path, train_csv_path_, test_csv_path_, action_col_, continuous_var):
    model = DirectSNet.load_model(model_path)
    device = next(model.parameters()).device
    use_cuda = device.type == 'cuda'

    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_train_val_test(train_csv_path_, test_csv_path_,
                                                                         action_col_, continuous_var)

    print("Testing samples: {0}".format(X_test.shape))

    loaders_dict = {
        'val': load_standard_data(
            StandardDS(X_test, y_test), use_cuda, device, batch_size=1
        )
    }
    phases = ['val']

    vc = pd.Series(y_train).value_counts()
    loss_w = torch.FloatTensor((1 - vc / vc.sum()) ** 5).to(device)
    m, history = standard(
        1, model, lambda phase: loaders_dict[phase](),
        torch.optim.RMSprop(model.parameters()),
        torch.nn.CrossEntropyLoss(weight=loss_w),
        accuracy_function=lambda y_hat, y: (y_hat.argmax(dim=1) == y).sum().float().item(),
        print_every=5,
        phases=phases
    )
    print(history['val_loss'][-1])
    print(history['val_acc'][-1])
