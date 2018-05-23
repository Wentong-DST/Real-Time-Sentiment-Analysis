import time
import cPickle
from models import CNN, LSTM, LSTM_CNN
from word2vec_twitter_model.word2vecReader import Word2Vec
from Preprocessing import divide_dataset, texts_preprocessing, model_update
from train import train, test
from main import setup
import os
import numpy as np

"""
This file is used to evaluate the external model
"""

os.environ['CUDA_VISIBLE_DEVICES']='1'


def train_model(args, model_choice):

    # loading data and divide
    filename = 'Sentiment140/training.1600000.processed.noemoticon.csv'
    # filename = 'Sentiment140/testdata.manual.2009.06.14.csv'
    train_set, test_set = divide_dataset(filename, args.ratio, args.sample)
    print 'trainset = %d, testset = %d' % (len(train_set[0]), len(test_set[0]))

    # loading word2vec model
    start = time.time()
    model_path = 'new_model.pkl'
    with open(model_path, 'r') as f:
        model = cPickle.load(f)
    # model= []
    print 'loading model successfully. Time spend = ', time.time() - start

    all_info = 'Train model %s \n' % model_choice

    xtrain, ytrain = train_set
    xtest, ytest = test_set

    batchinfo = ''
    for batch_size in [128, 256, 512, 1024, 2048,4096]:
        print 'batch_size = ', batch_size
        net = CNN.cnn(args)
        if args.use_cuda:
            net = net.cuda()
        for epoch in range(args.max_epochs):
            info = 'Epoch %d \n' % epoch
            train_loss, train_correct = 0, 0

            train_num_batch = len(xtrain) / batch_size
            for i in range(train_num_batch):
                # get data for each batch
                x = xtrain[i * batch_size: (i + 1) * batch_size]
                y = ytrain[i * batch_size: (i + 1) * batch_size]

                # process data to get word embedding
                try:
                    texts, x = texts_preprocessing(x, model, args.max_len, preprocess_choice="vector")
                except:
                    print 'epoch = ', epoch
                    print type(x)
                # simulate one batch
                if epoch < 1:
                    loss, correct = train(net, (x,y), args, fine_tune=False)
                else:
                    loss, correct, x = train(net, (x, y), args, fine_tune=True)
                    model = model_update(model, texts, x)
                    # print 'the differences = ', np.square(new_x[i]-x).sum()
                train_loss += loss
                train_correct += correct
                # print 'train model %s, time spend = %d' % (model_choice, time.time()-start)


            info += 'Train loss: %.3f | Acc: %.3f%% (%d/%d) \n' % \
                    (train_loss / train_num_batch, 100.0 * train_correct / train_num_batch / batch_size, train_correct,
                     train_num_batch * batch_size)

            test_loss, test_correct = 0, 0
            process_time, test_time = 0, 0
            test_num_batch = len(xtest) / batch_size
            for i in range(test_num_batch):
                # get data for each batch
                x = xtest[i*batch_size: (i+1)*batch_size]
                y = ytest[i*batch_size: (i+1)*batch_size]

                # process data
                start = time.time()
                _, x = texts_preprocessing(x, model, args.max_len, preprocess_choice="vector")
                process_time += (time.time() - start)

                # simulate one batch
                start = time.time()
                loss, correct = test(net, (x,y), args.use_cuda)
                test_time += (time.time() - start)
                test_loss += loss
                test_correct += correct


            info += 'Test loss: %.3f | Acc: %.3f%% (%d/%d) \n' % \
                    (test_loss/test_num_batch, 100.0 * test_correct / test_num_batch / batch_size, test_correct, test_num_batch * batch_size)
            info += 'batch_size = %d, for each batch, avg_process_time = %.6f, avg_test_time = %.6f \n' % \
                    (batch_size, float(process_time) / test_num_batch, float(test_time) / test_num_batch)

            print info
            all_info += info

        save_file = 'results/%s_bs%d_info.txt' % (model_choice, batch_size)
        with open(save_file,'w') as f:
            f.writelines(all_info)

        batchinfo += 'batch_size = %d, for each batch, avg_process_time = %.6f, avg_test_time = %.6f \n' % \
                     (batch_size, float(process_time) / test_num_batch, float(test_time) / test_num_batch) + \
                     '                 for each tweet, avg_process_time = %.6f, avg_test_time = %.6f \n' % \
                     (float(process_time) / test_num_batch / batch_size, float(test_time) / test_num_batch / batch_size)
    with open('results/batch-info.txt','w') as f:
        f.writelines(batchinfo)

if __name__ == '__main__':
    args = setup()
    model_choice = {0: "cnn", 1: "lstm", 2: "lstm_attn_cnn"}
    # for k,v in model_choice.items():
        # train_model(args, v)
    train_model(args, model_choice[0])
