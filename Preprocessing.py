import  os
import numpy as np
import cPickle
import pandas as pd
from twokenize import simple_tokenize
#from word2vec_twitter_model.word2vecReader import Word2Vec
from datetime import timedelta

def label_transfer(label):
    if label == 0:
        return [1, 0]
    else:
        return [0, 1]

def divide_dataset(filename, ratio, sample):
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


def w2v(word, model):
    try:
        return model[word]
    except:
        return np.zeros((400,))

def padding_2D(word_embedding, max_len):
    if word_embedding.shape[0] <= max_len:
        return np.pad(word_embedding, ((0, max_len - word_embedding.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
    else:
        return np.array(word_embedding[:max_len])

def tweet_preprocessing(tweets, model, max_len):
    '''
    :param tweets: list of tweets
    :return:
    '''
    # tokenization and replace URL, NUMBERs and MENTION with special tokens
    tweets = list(map(lambda t: simple_tokenize(t), tweets))  # list: (#tweets, list_of_words)
    # embedding
    embeddings = list(map(lambda tweet: np.array(list(map(lambda w: w2v(w, model), tweet))), tweets))   # list: (#tweets, np.array(#words, #dim))
    # padding
    #max_words = max(list(map(lambda t: len(t), tweets)))
    paddings = np.array(list(map(lambda x: padding_2D(x, max_len), embeddings)))

    return paddings

def lookup(vocab, word):
    if vocab.has_key(word):
        vocab[word] += 1
    else:
        vocab[word] = 1

def word2FreqVocab(list_of_words):
    vocab = dict()
    map(lambda words: map(lambda word: lookup(vocab, word), words), list_of_words)
    print 'Before filter less freq words, vocab_size = ', len(vocab)
    return vocab

def freq2IndexVocab(freq_vocab, min_freq):
    vocab = {'_UNKNOWN_': 0}
    idx = 1
    for k, v in freq_vocab.items():
        if v < min_freq:
            continue
        else:
            vocab[k] = idx
            idx += 1
    print 'After filter less freq words, vocab_size = ', len(vocab)
    return vocab

def build_vocab(filename, min_freq = 5):
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

def word2Index(word, vocab):
    try:
        return vocab[word]
    except:
        return 0

def padding_1D(idx, max_len):
    idx += [0] * max(0, max_len-len(idx))
    return idx[:max_len]

def Texts2Index(texts, vocab, max_len):
    # list of list of words: (#texts, #words)
    texts = list(map(lambda text: simple_tokenize(text), texts))
    # list of list of indexs: (#texts, #idxes)
    idxes = list(map(lambda words: list(map(lambda word: word2Index(word, vocab), words)), texts))
    # padding
    idxes = list(map(lambda idx: padding_1D(idx, max_len), idxes))
    return np.array(idxes)


def same_users(users):
    user_dict = dict()
    for user in users:
        if user_dict.has_key(user):
            user_dict[user] += 1
        else:
            user_dict[user] = 1
    print '#user = ', len(user_dict)
    users = list()
    for k, v in user_dict.items():
        if v > 1:
            users.append(k)
    print 'after filtering, #user = ', len(users)
    return users

def save_msg_to_csv(data, file, flag):
    ''' save data into csv file, 
        flag: 1: create a new file and keep header
              0: append to file
    '''
    if flag:
        if os.path.exists(file):
            os.remove(file)
            print 'remove %s' % file
        data.to_csv(file)
    else:
        with open(file, 'a') as f:
            data.to_csv(f, header=flag)


def evaluate_same_user(filename):
    parser = lambda date: pd.datetime.strptime(date[:20]+date[24:], '%c')
    columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
    df = pd.read_csv(filename, names=columns, parse_dates=[2], date_parser=parser)
    users = same_users(df['user'].tolist())
    diff = timedelta(minutes=10)
    same_senti, oppo_senti = 0, 0
    same_file, oppo_file = 'same.csv', 'oppo.csv'
    same_ = pd.DataFrame(columns = columns)
    oppo_ = pd.DataFrame(columns = columns)
    save_msg_to_csv(same_, same_file, flag=1)
    save_msg_to_csv(oppo_, oppo_file, flag=1)
    # return
    for uidx, user in enumerate(users):
        if uidx != 0 and uidx % 200 == 0:
            print '%d_th user. %d same polarity, %d opposity polarity' % (uidx, same_senti, oppo_senti)
            save_msg_to_csv(same_, same_file, flag=0)
            save_msg_to_csv(oppo_, oppo_file, flag=0)
            same_ = pd.DataFrame(columns = columns)
            oppo_ = pd.DataFrame(columns = columns)
        user_msgs = df[df['user']==user]
        idxes = user_msgs.index
        for i in range(len(idxes)-1):
            idx = idxes[i]
            next_idx = idxes[i+1]
            if user_msgs.loc[next_idx, 'date'] - user_msgs.loc[idx, 'date'] < diff :
                if user_msgs.loc[next_idx, 'polarity'] == user_msgs.loc[idx, 'polarity']:
                    same_senti += 1
                    same_ = pd.concat([same_, user_msgs[i: i+2]])
                else:
                    oppo_senti += 1
                    oppo_ = pd.concat([same_, user_msgs[i: i+2]])
    save_msg_to_csv(same_, same_file, flag=0)
    save_msg_to_csv(oppo_, oppo_file, flag=0)
    print '#msg with same polarity in one hour = ', same_senti
    print '#msg with oppo polarity in one hour = ', oppo_senti

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

    # generate same polarity within short time
    evaluate_same_user(filename)



