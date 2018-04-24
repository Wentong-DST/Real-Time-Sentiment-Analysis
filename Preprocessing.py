import nltk
from openpyxl import load_workbook
import  os
import cPickle
from gensim.models import Word2Vec

def tweet_file_to_pkl(filename = 'Sentiment Analysis Dataset.csv'):
    if not os.path.exists(filename):
        print 'file %s does not exist. Please correct the name and try again.' % filename

    wb = load_workbook(filename)
    ws = wb.active
    row = ws.max_row
    tweets = list()
    labels = list()
    possible_content_columns = ['D', 'E', 'F']
    for i in range(2, row+1):
        labels.append(ws['B'+str(i)].value)
        tweet = ''
        for c in possible_content_columns:
            tmp = ws[c+str(i)].value
            if tmp is not None:
                tweet += tmp + ' '
        tweets.append(tweet)

    with open('dataset.pkl', 'w') as f:
        cPickle.dump((tweets, labels), f)

def remove_repeated_letter(word):
    result = word[:2]
    i = 2
    while i < len(word):
        a, b, c = word[i-2], word[i-1], word[i]
        if b == a and c == b:
            i += 1
            while i < len(word):
                if word[i] != c:
                    result += word[i]
                    break
                else:
                    i += 1
        else:
            result += c
        i += 1
    return result


def tweet_preporcessing(tweet):
    # remove urls
    url_idx = 0
    while url_idx >= 0:
        url_idx = tweet.find('http')
        whitespace = tweet.find(' ', start=url_idx)
        tweet = tweet.replace(tweet[url_idx: whitespace], 'url')

    # remove emoticons
    tweet = tweet.lower().replace(':)', 'smile').replace(':-)','smile').replace(': )','smile').replace(':D','smile').replace('=)','smile').replace(':(','sadface').replace(':-(','sadface').replace(': (','sadface')

    # remove repeated letters
    words = nltk.word_tokenize(tweet)
    words = list(map(lambda word: remove_repeated_letter(word), words))

def WordEmbedding(words, embedding_method='word2vec'):
    '''
    return word embeddings of words
    :param words: list of words
    :param embedding_method: the methdo for word embedding, fasttext, GloVe, Word2Vec
    :return:
    '''
    if embedding_method == 'fasttext':
        pass
    elif embedding_method == 'glove':
        pass
    elif embedding_method == 'word2vec':
        model = Word2Vec.load('w2v.model')

    words = list(map(lambda w: model.wv[w], words))     # list of (#words, #dim)
    return words


def process(tweets):
    '''

    :param tweets: list of tweets
    :return:
    '''
    # remove url, emoticons and repeated letters in tweet_preporcessing
    words = list(map(lambda t: tweet_preporcessing(t), tweets))  # list: (#tweets, list_of_words)
    embeddings = list(map(lambda w: WordEmbedding(w), words))   # list: (#tweets, (#words, #dim))



