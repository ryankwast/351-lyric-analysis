"""
Lyric Data Processing
CMPE 351 Group Project
Spring 2021
"""


#%% Import actual data

import pandas as pd

ld = pd.read_csv('./data/track_features.csv')
ld = ld[ld["lyrics"]!="''"]

#%% Encode labels as 0 or 1

ld.valence = round(ld.valence)
ld.danceability = round(ld.danceability)

#%% Language filter

import nltk
import os

nltk.download('words')
def eng_ratio(text):
    ''' Returns the ratio of non-English to English words from a text '''

    english_vocab = set(w.lower() for w in nltk.corpus.words.words()) 
    text_vocab = set(w.lower() for w in text.split() if w.lower().isalpha()) 
    unusual = text_vocab.difference(english_vocab)
    diff = len(unusual)/len(text_vocab)
    return diff


before = ld.shape[0]
for row_id in ld.index:
    text = ld.loc[row_id]['lyrics']
    try:
        diff = eng_ratio(text)
    except:
        ld = ld[ld.index != row_id]
        print('row %s is causing problems' %row_id)
    if diff >= 0.5:
        ld = ld[ld.index != row_id]
after = ld.shape[0]
rem = before - after
print('%s have been removed.' %rem)
print('%s songs remain in the dataset.' %after)

dataPath1 = "/Users/Ryan/Documents/GitHub/351-lyric-analysis/data/filtered_data.csv"

# ld.to_csv(os.path.join(dataPath1), index=False)

#%% Split into training, test
import numpy as np

msk = np.random.rand(len(ld)) < 0.8

train = ld[msk]
test = ld[~msk]
                 

#%% Porter-Stemmer Tokenizer, suffix stripper

import nltk
import string
import re

porter_stemmer = nltk.stem.porter.PorterStemmer()

def porter_tokenizer(text, stemmer=porter_stemmer):
    """
    A Porter-Stemmer-Tokenizer hybrid to splits sentences into words (tokens) 
    and applies the porter stemming algorithm to each of the obtained token. 
    Tokens that are only consisting of punctuation characters are removed as well.
    Only tokens that consist of more than one letter are being kept.
    
    Parameters
    ----------
        
    text : `str`. 
      A sentence that is to split into words.
        
    Returns
    ----------
    
    no_punct : `str`. 
      A list of tokens after stemming and removing Sentence punctuation patterns.
    
    """
    lower_txt = text.lower()
    tokens = nltk.wordpunct_tokenize(lower_txt)
    stems = [porter_stemmer.stem(t) for t in tokens]
    no_punct = [s for s in stems if re.match('^[a-zA-Z]+$', s) is not None]
    return no_punct

#%% Stop words

# # One-time download of stop words file:
# nltk.download('stopwords')
# stp = nltk.corpus.stopwords.words('english')
# with open('./stopwords_eng.txt', 'w') as outfile:
#     outfile.write('\n'.join(stp))
    
    
with open('./stopwords_eng.txt', 'r') as infile:
    stop_words = infile.read().splitlines()
print('stop words %s ...' %stop_words[:5])

#%% Count Vectorizer

from sklearn.feature_extraction.text import CountVectorizer

# can try different values for ngram_range
countVec = CountVectorizer(
            encoding='utf-8',
            decode_error='replace',
            strip_accents='unicode',
            analyzer='word',
            binary=False,
            stop_words=stop_words,
            tokenizer=porter_tokenizer,
            ngram_range=(1,1)
    )

valenceTrain = train["valence"]
valenceTest = test["valence"]
danceTrain = train["danceability"]
danceTest = test["danceability"]
countVecTrain = countVec.fit(train["lyrics"].values)
countVecTest = countVec.fit(test["lyrics"].values)
# print('Vocabulary size: %s' %len(countVecTrain.get_feature_names()))

#%% Save Vect

import pickle
 
with open('countVecTrain', 'wb') as count_vector_file:
  pickle.dump(countVecTrain, count_vector_file)
  
with open('countVecTest', 'wb') as count_vector_file:
  pickle.dump(countVecTest, count_vector_file)


#%% Load Vect (test)

with open('countVecTrain', 'rb') as count_vector_file:
    countVecTrain = pickle.load(count_vector_file)
with open('countVecTest', 'rb') as count_vector_file:
    countVecTest = pickle.load(count_vector_file)


#%% Model

