#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[4]:


df = pd.read_csv ('spotify_songs.csv')


# In[6]:


lyrics = df['lyrics'].values.astype(str)


# In[41]:


y = pd.cut(df.valence,bins=[0,0.5,1],labels=[0,1])


# In[42]:


lyrics_train, lyrics_test, y_train, y_test = train_test_split(lyrics, y, random_state=1)


# In[44]:


vectorizer = CountVectorizer()
vectorizer.fit(lyrics_train)


# In[45]:


cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
lyrics_train_cv = cv.fit_transform(lyrics_train)
lyrics_test_cv = cv.transform(lyrics_test)


# In[46]:


naive_bayes = MultinomialNB()
naive_bayes.fit(lyrics_train_cv, y_train)
predictions = naive_bayes.predict(lyrics_test_cv)


# In[47]:


print('Accuracy score:' , accuracy_score(y_test, predictions))

