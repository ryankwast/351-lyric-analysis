#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from __future__ import division, print_function
from gensim import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import collections
import re
import string
nltk.download('punkt')
from nltk import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[68]:


df = pd.read_csv ('spotify_songs.csv')


# In[69]:


df1 = df[['lyrics', 'valence']]
df1.valence = pd.cut(df1.valence,bins=[0,0.5,1],labels=[0,1])
df1.lyrics = df1['lyrics'].values.astype(str)


# In[70]:


df1.valence.value_counts()


# In[71]:


def remove_punct(lyrics):#preprocessing to be replaced
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', '', lyrics)
    return text_nopunct
df1['lyrics'] = df1['lyrics'].apply(lambda x: remove_punct(x))


# In[72]:


tokens = [word_tokenize(sen) for sen in df1.lyrics]


# In[73]:


def lower_token(tokens): 
    return [w.lower() for w in tokens]    
    
lower_tokens = [lower_token(token) for token in tokens]


# In[74]:


stoplist = stopwords.words('english')
def removeStopWords(tokens): 
    return [w for w in tokens if w not in stoplist]
filtered_words = [removeStopWords(i) for i in lower_tokens]
df1['lyrics'] = [' '.join(i) for i in filtered_words]
df1['tokens'] = filtered_words


# In[10]:


pos = []
neg = []
for l in df1.valence:
    if l == 0:
        pos.append(0)
        neg.append(1)
    elif l == 1:
        pos.append(1)
        neg.append(0)
df1['Pos']= pos
df1['Neg']= neg

data = df1[['lyrics', 'tokens', 'valence', 'Pos', 'Neg']]
data.head()


# In[11]:


from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(data, test_size=0.10, random_state=42)


# In[12]:


all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in data_train["tokens"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
print("Max sentence length is %s" % max(training_sentence_lengths))


# In[14]:


all_test_words = [word for tokens in data_test['tokens'] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in data_test['tokens']]
TEST_VOCAB = sorted(list(set(all_test_words)))
print('%s words total, with a vocabulary size of %s' % (len(all_test_words), len(TEST_VOCAB)))
print('Max sentence length is %s' % max(test_sentence_lengths))


# In[30]:


import gensim
from gensim import models
from gensim.models import Word2Vec


# In[32]:


word2vec_path = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


# In[36]:


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments['tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)


# In[46]:


training_embeddings = get_word2vec_embeddings(word2vec, data_train, generate_missing=True)


# In[79]:


MS_LENGTH = 50
EMBEDDING_DIM = 300


# In[49]:


tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(data_train["lyrics"].tolist())
training_sequences = tokenizer.texts_to_sequences(data_train["lyrics"].tolist())

train_word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(train_word_index))


# In[50]:


train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)


# In[77]:


tew = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))
for word,index in train_word_index.items():
    tew[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)


# In[53]:


test_sequences = tokenizer.texts_to_sequences(data_test["lyrics"].tolist())
test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)


# In[75]:


label_names = ['Pos', 'Neg']


# In[76]:


y_train = data_train[label_names].values

x_train = train_cnn_data
y_tr = y_train


# In[90]:


embedding_layer = Embedding(len(train_word_index)+1,
                         300,
                         weights=[tew],
                         input_length=MS_LENGTH,
                          trainable=False)
    
sequence_input = Input(shape=(MS_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
epochs_count = 5
b_size = 36

convs = []
filter_sizes = [2,3,4,5,6]

for filter_size in filter_sizes:
    l_conv = Conv1D(filters=250, kernel_size=filter_size, activation='relu')(embedded_sequences)
    l_pool = GlobalMaxPooling1D()(l_conv)
    convs.append(l_pool)


lm = concatenate(convs, axis=1)

x = Dropout(0.1)(lm)  
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
preds = Dense(len(list(label_names)), activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.summary()


# In[91]:


hist = model.fit(x_train, y_tr, epochs=epochs_count, validation_split=0.1, shuffle=True, batch_size=b_size)


# In[92]:


predictions = model.predict(test_cnn_data, batch_size=1024, verbose=1)


# In[93]:


labels = [1, 0]


# In[94]:


prediction_labels=[]
for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])


# In[95]:



sum(data_test.valence==prediction_labels)/len(prediction_labels)

