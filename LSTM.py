import torch
import torch.nn as nn
import torch.nn.Functional as F


class lstm(nn.Module):
    def __init__(self, args):

        self.lstm = nn.LSTM(args.feature_dim, args.lstm_out, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(args.dropout)

        mlp_hidden = args.mlp_hidden
        mlp_hidden.insert(0, args.lstm_out*2)
        self.mlp = nn.Sequential()
        for i in range(len(mlp_hidden)-1):
            self.mlp.add_module('mlp'+str(i), nn.Linear(mlp_hidden[i], mlp_hidden[i+1]))
            self.mlp.add_module('activ'+str(i), nn.Sigmoid())
            self.mlp.add_module('dropout'+str(i), nn.Dropout(args.dropout))
        self.mlp.add_module('mlp'+str(i), nn.Linear(mlp_hidden[i], args.num_class))
        self.mlp.add_module('softmax', nn.Softmax())

    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)

        _, x = self.lstm(self.dropout(x))  # (batch_size, 1, lstm_out*2)

        x = self.mlp(self.dropout(x[0].squeeze(1)))

        return x
