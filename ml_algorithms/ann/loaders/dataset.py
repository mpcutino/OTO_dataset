from torch.utils.data import Dataset


class StandardDS(Dataset):

    def __init__(self, x_data, y_data):
        super(StandardDS, self).__init__()
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]
