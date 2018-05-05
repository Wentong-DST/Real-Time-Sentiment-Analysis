import torch
import torch.nn as nn
import torch.nn.functional as F


class lstm(nn.Module):
    def __init__(self, args):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(args.feature_dim, args.lstm_out, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(args.dropout)

        # Attention Mechanism
        self.attn = nn.Linear(args.lstm_out*2, 1)
        self.attn_softmax = nn.Softmax(dim=1)

        mlp_hidden = args.mlp_hidden
        mlp_hidden.insert(0, args.lstm_out*2)
        self.mlp = nn.Sequential()
        for i in range(len(mlp_hidden)-1):
            self.mlp.add_module('mlp'+str(i), nn.Linear(mlp_hidden[i], mlp_hidden[i+1]))
            self.mlp.add_module('activ'+str(i), nn.Sigmoid())
            self.mlp.add_module('dropout'+str(i), nn.Dropout(args.dropout))
        self.mlp.add_module('mlp'+str(i+1), nn.Linear(mlp_hidden[i], args.num_classes))
        self.mlp.add_module('softmax', nn.Softmax())

    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)

        # lstm
        x, _ = self.lstm(x)  # (batch_size, 1, lstm_out*2)

        # attention
        attn_weights = self.attn_softmax(self.attn(self.dropout(x)))
        x = torch.sum(attn_weights*x, dim=1)

        # mlp prediction
        x = self.mlp(self.dropout(x[0].squeeze(1)))

        return x
