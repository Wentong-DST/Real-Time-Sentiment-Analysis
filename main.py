import torch
import argparse
from CNN import cnn
from LSTM import lstm


def main():
    parser = argparse.ArgumentParser(description='PyTorch  Training')
    parser.add_argument('--feature_dim', default=400, type=int, help='feature dimension')
    parser.add_argument('--min_count', default=5, type=int, help='minimal frequency to filter words')
    parser.add_argument('--model_path', default='word2vec_twitter_model/word2vec_twitter_model.bin', type=str, help='word2vector model')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--kernel_num', default=32, type=int, help='number of each kernel')
    parser.add_argument('--kernel_size', default='3,4,5', type=str, help='kernel size')
    parser.add_argument('--cnn_out', default=300, type=int, help='cnn output size')
    parser.add_argument('--lstm_out', default=100, type=int, help='lstm hidden size')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout probability')
    parser.add_argument('--ratio', default=0.8, type=float, help='ratio of training dataset')
    parser.add_argument('--mlp_hidden', default='64,32', type=str, help='mlp hidden size')
    parser.add_argument('--num_classes', default=3, type=int, help='the number of classes')
    parser.add_argument('--max_epochs', default=50, type=int, help='the number of epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    args = parser.parse_args()

    args.kernel_size = [int(k) for k in args.kernel_size.split(',')]
    args.mlp_hidden = [int(k) for k in args.mlp_hidden.split(',')]
    args.use_cuda = torch.cuda.is_available() and args.cuda_able
    # args.use_cuda = False
    print 'use_cuda = ', args.use_cuda

    return args


if __name__ == '__main__':
    main()