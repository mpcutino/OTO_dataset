from torch.utils.data import DataLoader


def load_standard_data(dataset, use_cuda, device, **kwargs):
    data_loader = DataLoader(dataset, pin_memory=use_cuda, **kwargs)

    def iterate_function():
        for x, y in data_loader:
            x = x.to(device).float()
            y = y.to(device).long()
            yield x, y

    return iterate_function
