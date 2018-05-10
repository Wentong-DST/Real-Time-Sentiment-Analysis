import torch
import torch.nn as nn
import torch.nn.functional as F


class lstm_attn_cnn(nn.Module):
    def __init__(self, args):
        super(lstm_attn_cnn, self).__init__()
        self.embed = nn.Embedding(args.vocab_size, args.feature_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(args.feature_dim, args.lstm_out, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(args.dropout)

        # Attention Mechanism
        self.attn = nn.Linear(args.lstm_out*2, 1)
        self.attn_softmax = nn.Softmax(dim=1)

        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_size
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, kernel_size=(K, args.lstm_out * 2)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        
        mlp_hidden = args.mlp_hidden
        mlp_hidden.insert(0, Co * len(Ks))
        self.mlp = nn.Sequential()
        for i in range(len(mlp_hidden) - 1):
            self.mlp.add_module('mlp' + str(i), nn.Linear(mlp_hidden[i], mlp_hidden[i + 1]))
            self.mlp.add_module('activ' + str(i), nn.Sigmoid())
            self.mlp.add_module('dropout' + str(i), nn.Dropout(args.dropout))
        self.mlp.add_module('mlp' + str(i+1), nn.Linear(mlp_hidden[i+1], args.num_classes))
        self.mlp.add_module('softmax', nn.Softmax())

    def forward(self, x, embed=0):
        '''
        if embed, x with shape (batch_size, seq_len)
        otherwise, x with shape (batch_size, seq_len, feature_dim)
        '''

        # whether embedding, 
        if embed:
            x = self.embed(x)  
        # x: (batch_size, seq_len, feature_dim)
        # lstm
        x, _ = self.lstm(x)  # (batch_size, seq, lstm_out*2)

        # attention
        attn_weights = self.attn_softmax(self.attn(self.dropout(x)))
        x = attn_weights*x      # (batch_size, seq, lstm_out*2)

        # cnn
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, lstm_out*2)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, Co, seq_len)]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(batch_size, C0)]*len(Ks)

        x = torch.cat(x, 1)  # (batch_size, Co * len(Ks))

        # mlp prediction
        x = self.mlp(self.dropout(x))   # (batch_size, num_class)

        return x
