from torch import nn


class BasicNN(nn.Module):

    def __init__(self):
        super(BasicNN, self).__init__()

    @staticmethod
    def name():
        return "basic_nn"
