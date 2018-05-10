import time
import cPickle
from word2vec_twitter_model.word2vecReader import Word2Vec
from Preprocessing import divide_dataset, tweet_preprocessing, Texts2Index, build_vocab
from main import setup
import os
from keras_model import LSTM_Keras, CNN_LSTM

os.environ['CUDA_VISIBLE_DEVICES']="1"

def train_model(args):

	# loading data and divide
	filename = 'Sentiment140/training.1600000.processed.noemoticon.csv'
	# filename = 'Sentiment140/testdata.manual.2009.06.14.csv'
	train_set, test_set = divide_dataset(filename, 0.8, 0.1)
	print 'trainset = %d, testset = %d' % (len(train_set[0]), len(test_set[0]))

	# loading word2vec model
	start = time.time()
	model_path = 'word2vec_twitter_model/word2vec_twitter_model.bin'
	model = Word2Vec.load_word2vec_format(model_path, binary=True)
	#model= []
	print 'loading model successfully. Time spend = ', time.time() - start

	# # loading vocab
	# start = time.time()
	# vocab_path = 'vocab.pkl'
	# if os.path.exists(vocab_path):
	# 	with open(vocab_path, 'r') as f:
	# 		vocab = cPickle.load(f)
	# else:
	# 	vocab = build_vocab(filename, args.min_freq)

	# args.vocab_size = len(vocab)

	xtrain, ytrain = train_set
	xtest, ytest = test_set

	all_info = ''
	batchinfo = ''
	for batch_size in [128, 256, 512, 1024]:
		net = CNN_LSTM(args)
		for epoch in range(args.max_epochs):
			info = 'Epoch %d \n' % epoch
			train_loss, train_acc = 0.0, 0.0

			train_num_batch = len(xtrain) / batch_size
			for i in range(train_num_batch):
				# get data for each batch
				x = xtrain[i*batch_size: (i+1)*batch_size]
				y = ytrain[i*batch_size: (i+1)*batch_size]

				# process data
				# x = Texts2Index(x, vocab, args.max_len)
				x = tweet_preprocessing(x, model, args.max_len)

				# simulate one batch
				loss, acc = net.train_on_batch(x, y)
				train_loss += loss
				train_acc += acc

			info += 'Train loss: %.3f | Acc: %.3f%% (%d/%d) \n' % \
					(train_loss/train_num_batch, train_acc / train_num_batch, int(train_acc*batch_size), train_num_batch * batch_size)

			test_loss, test_acc = 0.0, 0.0
			process_time, test_time = 0, 0
			test_num_batch = len(xtest) / batch_size
			for i in range(test_num_batch):
				# get data for each batch
				x = xtest[i*batch_size: (i+1)*batch_size]
				y = ytest[i*batch_size: (i+1)*batch_size]

				# process data
				start = time.time()
				# x = Texts2Index(x, vocab, args.max_len)
				x = tweet_preprocessing(x, model, args.max_len)
				process_time += (time.time() - start)

				# simulate one batch
				start = time.time()
				loss, acc = net.test_on_batch(x, y)
				test_time += (time.time() - start)
				test_loss += loss
				test_acc += acc

			info += 'Test loss: %.3f | Acc: %.3f%% (%d/%d) \n' %	\
					(test_loss/test_num_batch, test_acc / test_num_batch, int(test_acc*batch_size), test_num_batch * batch_size)
			info += 'batch_size = %d, for each batch, avg_process_time = %.6f, avg_test_time = %.6f \n' %	\
					(batch_size, float(process_time) / test_num_batch, float(test_time) / test_num_batch)

			print info
			all_info += info

		save_file = 'results/cnn_lstm_bs%d_info.txt' % (batch_size)
		with open(save_file,'w') as f:
			f.writelines(all_info)

		batchinfo += 'batch_size = %d, for each batch, avg_process_time = %.6f, avg_test_time = %.6f \n' % \
		             (batch_size, float(process_time) / test_num_batch, float(test_time) / test_num_batch)
	with open('results/cnn_lstm_batch-info.txt','w') as f:
		f.writelines(batchinfo)

if __name__ == '__main__':
    args = setup()

    train_model(args)
