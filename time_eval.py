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

    print 'Train model %s' % model_choice

    # loading data and divide
    # filename = 'Sentiment140/training.1600000.processed.noemoticon.csv'
    filename = 'Sentiment140/testdata.manual.2009.06.14.csv'
    train_set, test_set = divide_dataset(filename)

    # loading word2vec model
    model_path = 'word2vec_twitter_model/word2vec_twitter_model.bin'
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    # train_set[0] = tweet_preprocessing(train_set[0], model)

    batch_size = args.batch_size
    xtrain, ytrain = train_set
    num_batch = len(train_set[1]) / batch_size
    for i in range(num_batch):
        # get data for each batch
        x = xtrain[i*batch_size: (i+1)*batch_size]
        y = ytrain[i*batch_size: (i+1)*batch_size]

        # process data
        start = time.time()
        x = tweet_preprocessing(x, model)
        print 'processing batch_size = %d, time spend = %d' % (batch_size, time.time()-start),

        # simulate one batch
        start = time.time()
        train(net, (x,y), False)
        print 'train model %s, time spend = %d' % (model_choice, time.time()-start)

    test(net, test_set, False)


if __name__ == '__main__':
    args = main()

    train_model(args, "cnn")
