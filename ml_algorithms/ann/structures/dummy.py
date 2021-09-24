import os
import json
import torch
from torch import nn

from ml_algorithms.ann.structures.basic import BasicNN


class DirectSNet(BasicNN):

    def __init__(self, in_size, out_size):
        super(DirectSNet, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.init_feat, self.res_feat, self.classifier = self.__net_structure__(in_size, out_size)

    def forward(self, x):
        x = self.init_feat(x)
        for i, f in enumerate(self.res_feat):
            # apply residual connection if not in the last feature sequential layer
            x = f(x) + x if i != len(self.res_feat) - 1 else f(x)
        return self.classifier(x)

    def __net_structure__(self, in_size, out_size):
        residual_size = 350
        ending_size = 512

        init_features = nn.Sequential(
            self.__module__(in_size, 1024),
            self.__module__(1024, residual_size)
        )
        residual_features = nn.ModuleList([
            nn.Sequential(
                self.__module__(residual_size, 200),
                self.__module__(200, residual_size),
            ),
            # apply residual connection, then:
            nn.Sequential(
                self.__module__(residual_size, 100),
                self.__module__(100, residual_size)
            ),
            # apply residual connection, then:
            self.__module__(residual_size, ending_size)
        ])
        classifier = nn.Linear(ending_size, out_size)

        return init_features, residual_features, classifier

    def save_model(self, save_folder=""):
        if len(save_folder) and not os.path.exists(save_folder):
            os.makedirs(save_folder)

        with open(os.path.join(save_folder, "{0}_params.json".format(self.name())), "w") as fd:
            json.dump(self.params_dict(), fd)
        torch.save(self.state_dict(), os.path.join(save_folder, "{0}.pth".format(self.name())))

    def params_dict(self):
        return {
            'in_size': self.in_size,
            'out_size': self.out_size
        }

    @staticmethod
    def load_model(save_folder=""):
        with open(os.path.join(os.path.join(save_folder, "{0}_params.json".format(DirectSNet.name())))) as fd:
            d = json.load(fd)
        model = DirectSNet(**d)
        model.load_state_dict(
            torch.load(os.path.join(save_folder, "{0}.pth".format(DirectSNet.name())))
        )
        model.eval()
        return model

    @staticmethod
    def name():
        return "direct_snet"

    @staticmethod
    def __module__(in_size, out_size, dropout=.15):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.LayerNorm(out_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )


if __name__ == '__main__':
    import torch
    import time

    n_ = DirectSNet(6, 30)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    n_.to(torch.device(device))
    n_.eval()
    count, total_ellapsed = 10000, 0.0
    for i in range(count):
        start = time.time()
        inp_ = torch.rand((1, 6), dtype=torch.float, device=device)
        out_ = n_(inp_)
        total_ellapsed += time.time() - start
    print("Total ellapsed {0}s in {1} iterations, with and average of {2}it/s".
          format(total_ellapsed, count, total_ellapsed/count))

    print(sum([p.numel() for p in n_.parameters()]))
