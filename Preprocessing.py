import  os
import numpy as np
import cPickle
import pandas as pd
from twokenize import simple_tokenize
from word2vec_twitter_model.word2vecReader import Word2Vec
from datetime import timedelta
from Utils import *
import time

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy.sparse import hstack

def divide_dataset(filename, ratio, sample):
    '''
    divide dataset into train and test dataset, ratio is the rate of train/test, sample is the (train+test)/all_data
    '''
    if not os.path.exists(filename):
        print 'file %s does not exist. Please correct the name and try again.' % filename

    parser = lambda date: pd.datetime.strptime(date[:20]+date[24:], '%c')
    columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
    df = pd.read_csv(filename, names=columns, parse_dates=[2], date_parser=parser) # tweet id as index
    categories = len(set(df['polarity']))
    if categories == 2:
        df['polarity'] = df['polarity'].apply(lambda x: x/4.0)
    elif categories == 3:
        df['polarity'] = df['polarity'].apply(lambda x: x/2.0)

    texts = df['text']
    labels = df['polarity']
    # get index for negative and positive samples
    neg_idx = labels.index[labels==0]
    if categories == 2:
        pos_idx = labels.index[labels==1]
    elif categories == 3:
        pos_idx = labels.index[labels==2]

    num_neg = len(neg_idx)
    num_pos = len(pos_idx)
    train_num_neg, train_num_pos = int(num_neg * ratio * sample), int(num_pos * ratio * sample)
    test_num_neg, test_num_pos = int(num_neg * sample - train_num_neg), int(num_pos * sample - train_num_pos)

    train_idx = neg_idx.tolist()[:train_num_neg] + pos_idx.tolist()[:train_num_pos]
    test_idx = neg_idx.tolist()[train_num_neg: train_num_neg+test_num_neg] + pos_idx.tolist()[train_num_pos: train_num_pos+test_num_pos]
    train_idx = train_idx[:int(len(train_idx))]
    test_idx = test_idx[:int(len(test_idx))]
    # shuffle train samples
    tmp = np.arange(len(train_idx))
    np.random.shuffle(tmp)
    train_idx = np.array(train_idx)[tmp]
    train_texts = texts[train_idx]
    train_labels = labels[train_idx]
    # train_labels = np.array(map(lambda l: label_transfer(l), train_labels.tolist()))
    train_labels = np.array(train_labels.tolist())

    # shuffle test samples
    tmp = np.arange(len(test_idx))
    np.random.shuffle(tmp)
    test_idx = np.array(test_idx)[tmp]
    test_texts = texts[test_idx]
    test_labels = labels[test_idx]
    # test_labels = np.array(map(lambda l: label_transfer(l), test_labels.tolist()))
    test_labels = np.array(test_labels.tolist())
    return (train_texts, train_labels), (test_texts, test_labels)

def Texts2Matrix(texts, model, max_len):
    '''
        preprocess tweets by tokenize, embedding and padding,
        return list of word2vector matrix
        '''
    # tokenization and replace URL, NUMBERs and MENTION with special tokens
    # start = time.time()
    texts = list(map(lambda t: simple_tokenize(t), texts))  # list: (#tweets, list_of_words)
    # print 'tokenize time = ', time.time() - start

    # embedding
    # start = time.time()
    embeddings = list(map(lambda tweet: np.array(list(map(lambda w: w2v(w, model), tweet))), texts))  # list: (#tweets, np.array(#words, #dim))
    # print 'embedding time = ', time.time() - start

    # padding
    # start = time.time()
    if max_len == 0:
        max_len = max(list(map(lambda t: len(t), texts)))
    paddings = np.array(list(map(lambda x: padding_2D(x, max_len), embeddings)))
    # print 'padding2D time = ', time.time() - start

    return texts, paddings

def build_vocab(filename, min_freq = 5):
    '''
    build vocab from texts in filename, with minimum frequency (5 by default)
    '''
    if not os.path.exists(filename):
        print 'file %s does not exist. Please correct the name and try again.' % filename

    parser = lambda date: pd.datetime.strptime(date[:20]+date[24:], '%c')
    columns = ['text']
    df = pd.read_csv(filename, names=columns, usecols=[5]) # tweet id as index

    df = df['text'].tolist()
    # texts to words, words: list of list of words
    words = list(map(lambda tweet: simple_tokenize(tweet), df))
    # counting words to get vocab with (word: freq)
    freq_vocab = word2FreqVocab(words)
    # transfer freq_vocab to index vocab with (word: index)
    vocab = freq2IndexVocab(freq_vocab, min_freq)
    with open('vocab.pkl','w') as f:
        cPickle.dump(vocab, f)

    return vocab

def Texts2Index(texts, vocab, max_len):
    '''
    transfer list of text to list of indexes by looking up the vocab,
    return list of indexes
    '''
    # list of list of words: (#texts, #words)
    # start = time.time()
    texts = list(map(lambda text: simple_tokenize(text), texts))
    # print 'tokenize time = ', time.time() - start

    # list of list of indexs: (#texts, #idxes)
    # start = time.time()
    idxes = list(map(lambda words: list(map(lambda word: word2Index(word, vocab), words)), texts))
    # print 'word2index time = ', time.time() - start

    # padding
    # start = time.time()
    if max_len == 0:
        max_len = max(list(map(lambda t: len(t), texts)))
    idxes = list(map(lambda idx: padding_1D(idx, max_len), idxes))
    # print 'padding1D time = ', time.time() - start
    return np.array(idxes)



def Texts2SVM_Feature(texts, **kwargs):#, tfidf, stopwords_list, SentiWords):
    '''
    transfer list of text to list of features for SVM,
    return list of features
    '''
    tfidf = kwargs['tfidf']
    stopwords_list = kwargs['stopwords']
    SentiWords = kwargs['SentiWords']

    # list of list of words: (#texts, #words)
    # start = time.time()
    texts = list(map(lambda text: simple_tokenize(text), texts))
    # print 'tokenize time = ', time.time() - start

    # filter useless stopwords and stemming
    # list of [#noun, #adj, #adv, #verb, #url, #hashtag, #mentions, #number, #cap, #strong_neg, #strong_pos, #weak_neg, #weak_pos]
    other_values = np.array(list(map(lambda text: svm_text_feature(text, stopwords_list, SentiWords), texts)))

    # get tfidf value
    # texts = list(map(lambda text: ' '.join(text), texts))
    # tfidf_values = tfidf.fit_transform(texts)
    # print tfidf_values.shape

    # all_features = hstack([tfidf_values, other_values])
    # print all_features.shape
    # return all_features
    return other_values


def texts_preprocessing(texts, model, max_len, preprocess_choice = "vector", **kwargs):
    '''
    texts preprocessing based on preporcess_choice
    :param tweets:
    :param model:
    :param max_len:
    :param preprocess_choice:
    :return:
    '''
    if preprocess_choice == "vector":
        return Texts2Matrix(texts, model, max_len)
    elif preprocess_choice == "index":
        return Texts2Index(texts, model, max_len)
    elif preprocess_choice == "svm":
        return Texts2SVM_Feature(texts, **kwargs)
    else:
        print 'Please input correct preprocess_choice'
        return

def model_update(model, texts, embeddings):
    """
    update model with fine-tuned embeddings
    :param model:
    :param texts: list of list of words, (#msgs, #words_in_each_msg)    #msg = batch_size
    :param embeddings: fine-tuned matrix, (#msg, #words, #feature_dim)
    :return: updated model
    """
    for text, embedding in zip(texts, embeddings):
        for i, word in enumerate(text):
            try:
                model[word] = embedding[i]
            except:
                pass
    return model


if __name__ == '__main__':
    # loading data and divide
    filename = 'Sentiment140/training.1600000.processed.noemoticon.csv'
    # filename = 'Sentiment140/testdata.manual.2009.06.14.csv'
    # train_set, test_set = divide_dataset(filename, 0.8, 0.1)

    # # loading word2vec model
    # model_path = 'word2vec_twitter_model/word2vec_twitter_model.bin'
    # model = Word2Vec.load_word2vec_format(model_path, binary=True)

    # build vocabulary
    # build_vocab(filename)
    # with open('vocab.pkl','r') as f:
    #     vocab = cPickle.load(f)

    # test textw2index function
    # texts = ['today is a lovely day.', 'what\' your name?']
    # idx = Texts2Index(texts, vocab, 10)
    # print idx

