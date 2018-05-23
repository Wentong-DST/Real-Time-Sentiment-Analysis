# Real-Time-Sentiment-Analysis

The real-time sentiment analysis system takes in tweets, classify tweets into two polarities (positive or negative) after preprocessing. 

## Pre-Requirement  
### Dependencies  
* Pytorch
* scipy
* numpy
* NLTK
* pandas 

### Dataset and WordEmbedding model
The dataset is a public dataset from [Stanford](http://help.sentiment140.com/for-students), which contains abount 1.6m data.
```Bash
wget http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip  
mkdir Sentiment140 | unzip trainingandtestdata.zip -d Sentiment140/
```

The word embedding model is pre-trained on 400m tweets.
```Bash
wget http://yuca.test.iminds.be:8900/fgodin/downloads/word2vec_twitter_model.tar.gz
tar zxvf word2vec_twitter_model.tar.gz
```

## Preprocessing  

* Tokenization  
The first step of preprocessing part is tokenization. In this system, we utilized [TweetMotif](https://github.com/brendano/tweetmotif) to tokenize tweet and replace urls, mentions(starts with symbol '@') , numbers and hashtags(starts with symbol '#') with special tokens '_URL_', '_MENTION_', '_NUMBER_' and '_HASHTAG_' respectively.  

* Word processing  
After tokenization, each tweet is split into a list of words. Then three word preprocess_choices could be chosen:    

1. Word Embedding: The word embedding model is a [model](https://www.fredericgodin.com/software/) pre-trained on 400m tweets.    

2. One-hot Vector: Use all words in datasets to build a vocabulary in [build_vocab](Preprocessing.py) function. And then lookup the vocabulary and transfer word to its index in the vocabulary as [Texts2Index](Preprocessing.py).  

3. SVM: To count the number of polarity words and other meaningful features in each messages as [Texts2SVM_Feature](Preprocessing.py).   


## Classification  

We design three deep learning models to do binary classification, which are [CNN+MLP](models/CNN.py), [LSTM+Attention+MLP](models/LSTM.py) and [LSTM+Attention+CNN+MLP](models/LSTM_CNN.py).  



## Execution  
* time_eval.py: use word embedding to train/test model  
* idx_time_eval.py: use one-hot vector to train/test model  
* fine_tune.py: to fine-tune the word embeddings from existing model  
Use `python *.py` to run relative file, and results would be saved into results/ folder.
