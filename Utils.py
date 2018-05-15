import  os
import numpy as np
import cPickle
import pandas as pd
from twokenize import simple_tokenize
from word2vec_twitter_model.word2vecReader import Word2Vec
from datetime import timedelta


def label_transfer(label):
    if label == 0:
        return [1, 0]
    else:
        return [0, 1]

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


def word2Index(word, vocab):
    try:
        return vocab[word]
    except:
        return 0

def padding_1D(idx, max_len):
    idx += [0] * max(0, max_len-len(idx))
    return idx[:max_len]


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
        user_msgs.sort_value('date', inplace=True)
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
                    oppo_ = pd.concat([oppo_, user_msgs[i: i+2]])
    save_msg_to_csv(same_, same_file, flag=0)
    save_msg_to_csv(oppo_, oppo_file, flag=0)
    print '#msg with same polarity in one hour = ', same_senti
    print '#msg with oppo polarity in one hour = ', oppo_senti

def delete_unused_word():
    print 'delete starting...'
    model_path = 'word2vec_twitter_model/word2vec_twitter_model.bin'
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    print 'Load model successfully.'

    with open('vocab.pkl', 'r') as f:
        vocab = cPickle.load(f)
    keys = vocab.keys()
    print 'Load vocab successfully.'

    big_v = model.vocab
    big_k = big_v.keys()

    for i, k in enumerate(big_k):
        if k not in keys:
            del big_v[k]
        if i%10000 == 0:
            print 'finish %d_th words.' % i

    new_dict = dict()
    keys = model.vocab.keys()
    for k in keys:
        new_dict[k] = model[k]

    with open('new_model.pkl','w') as f:
        cPickle.dump(new_dict, f)

def evaluate_polarity_from_same_user(filename):
    parser = lambda date: pd.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    df = pd.read_csv(filename, usecols=[3], parse_dates=[0], date_parser=parser)
    df = df['date']
    diff_list = []
    for i in range(0,len(df),2):
        diff = (df[i+1]-df[i]).seconds
        diff_list.append(diff)

    import matplotlib.pyplot as plt
    plt.hist(diff_list, bins=10, label = filename[:-4])
    plt.legend()
    plt.title('Histogram of same/opposity polarity')
    plt.xlabel('Seconds')
    plt.ylabel('Numbers')
    plt.savefig(filename[:-4])
