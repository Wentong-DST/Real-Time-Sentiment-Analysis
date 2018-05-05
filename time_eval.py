import time
import cPickle
from CNN import cnn
from LSTM import lstm
from word2vec_twitter_model.word2vecReader import Word2Vec
from Preprocessing import divide_dataset, tweet_preprocessing
from train import train, test
from main import main

"""
def build_models(filename):
    print 'start to loading wiki corpus, ',
    start = time.time()
    wiki = wikicorpus.WikiCorpus(filename, lemmatize=False, dictionary={})
    sentences = list(wiki.get_texts())
    print 'time spent =', (time.time() - start)/60.0
    print 'get wiki sentences successfully.'
    start = time.time()
    with open('wikis.pkl','w') as f:
        cPickle.dump(sentences, f)
    print 'save wiki sentences into file. ',
    print 'time spent =', (time.time() - start)/60.0

    # to build word2vec model
    start = time.time()
    params = {'size':200, 'window':10, 'min_count':10,
              'workers': max(1, multiprocessing.cpu_count()-1)}
    w2v = Word2Vec(sentences, **params)
    w2v.save('w2v_model')
    print 'save w2v model into file. ',
    print 'time spent =', (time.time() - start)/60.0

    # to build glove model
    start = time.time()
    glv_corpus = Corpus()
    glv_corpus.fit(sentences, window=10)
    glove = Glove(no_components=200)
    glove.fit(glv_corpus.matrix, epochs=100,no_threads=max(1, multiprocessing.cpu_count()-1), verbose=True)
    glove.save('glove_model')
    print 'save glove model into file. ',
    print 'time spent =', (time.time() - start)/60.0
"""

def load_model(filepath):
    pass

def train_model(args, model_choice):
    if model_choice == "cnn":
        net = cnn(args)
    elif model_choice == "lstm":
        net = lstm(args)
    else:
        print "Wrong model_choice, please correct and try again."
        return

    all_info = 'Train model %s \n' % model_choice

    # loading data and divide
    # filename = 'Sentiment140/training.1600000.processed.noemoticon.csv'
    filename = 'Sentiment140/testdata.manual.2009.06.14.csv'
    train_set, test_set = divide_dataset(filename)

    # loading word2vec model
    model_path = 'word2vec_twitter_model/word2vec_twitter_model.bin'
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    #model= []

    batch_size = args.batch_size
    xtrain, ytrain = train_set
    xtest, ytest = test_set
    xtest = tweet_preprocessing(xtest, model)
    num_batch = len(xtrain) / batch_size

    for epoch in range(args.max_epochs):
        info = 'Epoch %d \n' % epoch
        train_loss, train_correct = 0, 0
        process_time, train_time = 0, 0
        for i in range(num_batch):
            # get data for each batch
            x = xtrain[i*batch_size: (i+1)*batch_size]
            y = ytrain[i*batch_size: (i+1)*batch_size]

            # process data
            start = time.time()
            x = tweet_preprocessing(x, model)
            process_time += (time.time() - start)
            # print 'processing batch_size = %d, time spend = %d' % (batch_size, time.time()-start),

            # simulate one batch
            start = time.time()
            loss, correct = train(net, (x,y), False)
            train_time += (time.time() - start)
            train_loss += loss
            train_correct += correct
            # print 'train model %s, time spend = %d' % (model_choice, time.time()-start)
        info += 'batch_size = %d, for each batch, avg_process_time = %.2f, avg_train_time = %.2f \n' % \
                        (batch_size, float(process_time)/num_batch, float(train_time)/num_batch)
        info += 'Train loss: %.3f | Acc: %.3f%% (%d/%d) \n' % \
                (train_loss/num_batch, 100.0 * correct / num_batch / batch_size, correct, num_batch * batch_size)

        info += test(net, (xtest, ytest), False)
        print info
        all_info += info

    save_file = 'results/%s_bs%d_info.txt' % (model_choice, batch_size)
    with open(save_file,'w') as f:
        f.writelines(all_info)


if __name__ == '__main__':
    args = main()

    train_model(args, "lstm")
