import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn(nn.Module):
    def __init__(self, args):
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_size
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, kernel_size=(K, args.feature_dim)) for K in Ks])


        mlp_hidden = args.mlp_hidden
        mlp_hidden.insert(0, args.cnn_out)
        self.mlp = nn.Sequential()
        for i in range(len(mlp_hidden) - 1):
            self.mlp.add_module('mlp' + str(i), nn.Linear(mlp_hidden[i], mlp_hidden[i + 1]))
            self.mlp.add_module('activ' + str(i), nn.Sigmoid())
            self.mlp.add_module('dropout' + str(i), nn.Dropout(args.dropout))
        self.mlp.add_module('mlp' + str(i), nn.Linear(mlp_hidden[i], args.num_class))
        self.mlp.add_module('softmax', nn.Softmax())

    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, feature_dim)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, Co, seq_len)]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(batch_size, Co)]*len(Ks)

        x = torch.cat(x, 1)  # (batch_size, len(Ks) * Co)
        x = self.mlp(x)   # (batch_size, num_class)

        return x
