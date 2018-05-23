import  os
import numpy as np
import cPickle
import pandas as pd
from word2vec_twitter_model.word2vecReader import Word2Vec
from datetime import timedelta
# import matplotlib.pyplot as plt
from twokenize import simple_tokenize
from nltk import pos_tag
from collections import OrderedDict


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


def user_vs_msg(users):
    '''
    to get dict for {user_name: #msg_they_post}
    :param users: list of username
    '''
    user_dict = dict()
    for user in users:
        if user_dict.has_key(user):
            user_dict[user] += 1
        else:
            user_dict[user] = 1
    print 'In the dataset, #user = ', len(user_dict)
    return user_dict

def user_filter_by_msg_number(user_dict, threshold=1):
    '''
    to filter users whose posts not greater than threshold
    '''
    users = list()
    for k, v in user_dict.items():
        if v > threshold:
            users.append(k)
    print 'After filtering #msgs not greater than %d, #user = %d' % (threshold ,len(users))
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


def evaluate_same_user_msg(filename):
    '''
    get users and save their msgs with same or opposite polarity in short interval into file
    '''
    parser = lambda date: pd.datetime.strptime(date[:20]+date[24:], '%c')
    columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
    df = pd.read_csv(filename, names=columns, parse_dates=[2], date_parser=parser)
    user_dict = user_vs_msg(df['user'].tolist())
    users = user_filter_by_msg_number(user_dict)
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
    '''
    delete unused word in external model
    :return:
    '''
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


def load_polarity_dict(filename = 'SentiWords_1.1.txt'):
    if not os.path.exists(filename):
        print 'The polarity file %s does not exist. Please correct and try again.' % filename
        return

    polarity_dict = dict()
    with open(filename,'r') as f:
        for line in f:
            word, polarity = line.split()
            polarity_dict[word] = float(polarity)
    return polarity_dict

def word2Polarity(word, polar_dict):
    try:
        return polar_dict[word]
    except:
        return 0.0

def svm_text_feature(text, stopwords_list, SentiWords):
    '''
    text is a list of word
    '''
    tags = pos_tag(text)
    tags = list(filter(lambda word: word[0] in stopwords_list, tags))
    # #pos for noun, adj, adv, verb
    num_pos = [0, 0, 0, 0]
    # #url, #mention, #hashtags, #number
    num_icon = OrderedDict({'_URL_':0, '_MENTION_':0, '_HASHTAG_':0, '_NUMBER_':0})
    icon = num_icon.keys()
    num_cap = 0     # # of captialization
    new_word_list = []
    for (word, tag) in tags:
        if word in icon:
            num_icon[word] += 1
            num_pos[0] += 1
            continue
        else:
            if word.isupper():
                num_cap += 1

        if tag == 'NN':
            num_pos[0] += 1
            new_word_list.append(word+'#n')
        elif tag == 'JJ':
            num_pos[1] += 1
            new_word_list.append(word + '#a')
        elif tag == 'RB':
            num_pos[2] += 1
            new_word_list.append(word + '#r')
        elif tag == 'VB':
            num_pos[3] += 1
            new_word_list.append(word + '#v')
        else:
            new_word_list.append(word + '#r')

    polarities = np.array(list(map(lambda word: word2Polarity(word, SentiWords), new_word_list)))
    # #polarity: strong neg, strong pos, weak neg, weak pos
    num_polarity = []
    num_polarity.append(len(np.where(polarities<-0.5)[0]))
    num_polarity.append(len(np.where(polarities > 0.5)[0]))
    num_polarity.append(len(np.where(polarities < 0)[0]) - num_polarity[0])
    num_polarity.append(len(np.where(polarities > 0)[0]) - num_polarity[1])

    return num_pos + num_icon.values() + [num_cap] + num_polarity


"""
def plot_user_vs_time_interval_for_polarity(filenames):
    '''
    plot histogram for users vs time_interval for same or opposite polarity in the file
    '''
    data, labels = [], []
    for filename in filenames:
        if filename.startswith('same'):
            labels.append('same polarity')
        elif filename.startswith('oppo'):
            labels.append('opposity polarity')
        else:
            print 'get wrong filename %s' % filename
            return
        parser = lambda date: pd.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        df = pd.read_csv(filename, usecols=[3], parse_dates=[0], date_parser=parser)
        df = df['date']
        diff_list = []
        for i in range(0,len(df),2):
            diff = (df[i+1]-df[i]).seconds
            diff_list.append(diff)
        data.append(np.array(diff_list))
    plt.hist(data, bins=10, label = labels)
    plt.legend()
    plt.xticks(range(30,600,60), range(1,11))
    plt.title('Histogram of same/opposity polarity')
    plt.xlabel('Minutes')
    plt.ylabel('Numbers')
    plt.savefig('Histogram_of_#user_vs_#minutes_for_polarity')


def plot_user_vs_msg(filename):
    '''
    plot histogram for users vs msgs they posted in the datset
    :param filename:
    :return:
    '''
    parser = lambda date: pd.datetime.strptime(date[:20] + date[24:], '%c')
    columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
    df = pd.read_csv(filename, names=columns, parse_dates=[2], date_parser=parser)
    user_dict = user_vs_msg(df['user'].tolist())
    num_post = user_dict.values()
    plt.hist(num_post, bins=range(25))
    plt.title('Histogram of the number of tweets posted by each user')
    plt.xlabel('#Tweets')
    plt.ylabel('#Users')
    plt.savefig('Hist_of_#tweets_vs_#users')


def evaluate_overlap_of_msg(filenames, overlap_choice=0):
    '''
    :param overlap_choice: 0: rate = #overlap_words/#all_words_in_two_msgs
                           1: rate = #overlap_words/#all_words_in_first_msgs
    :return:
    '''
    data, labels = [], []
    for filename in filenames:
        if filename.startswith('same'):
            labels.append('same polarity')
        elif filename.startswith('oppo'):
            labels.append('opposity polarity')
        else:
            print 'get wrong filename %s' % filename
            return
        df = pd.read_csv(filename, usecols=[6])
        df = df['text']
        rate_list = []
        for i in range(0, len(df), 2):
            msg1, msg2 = df[i], df[i+1]
            words1, words2 = simple_tokenize(msg1), simple_tokenize(msg2)
            if overlap_choice == 0:
                rate = 1.0 * len(set(words1)&set(words2)) / len(set(words1)|set(words2))
            elif overlap_choice == 1:
                rate = 1.0 * len(set(words1)&set(words2)) / len(set(words1))
            rate_list.append(rate)
        data.append(np.array(rate_list))
    plt.hist(data, bins=10, label=labels)
    plt.legend()
    plt.xticks(np.arange(0,1.1,0.1))
    plt.title('Histogram of overlap rate for same/opposity polarity')
    plt.xlabel('Overlap Rate')
    plt.ylabel('Numbers')
    plt.savefig('Histogram_of_#tweets_vs_overlap_rate_for_polarity_'+str(overlap_choice))


"""