#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[ ]:


df = pd.read_csv ('spotify_songs.csv')


# In[ ]:


lyrics = df['lyrics'].values.astype(str)


# In[ ]:


y = pd.cut(df.valence,bins=[0,0.5,1],labels=['Sad','Happy'])


# In[ ]:


lyrics_train, lyrics_test, y_train, y_test = train_test_split(lyrics, y, test_size=0.25, random_state=1000)


# In[ ]:


vectorizer = CountVectorizer()
vectorizer.fit(lyrics_train)


# In[ ]:


x_train = vectorizer.transform(lyrics_train)
x_test  = vectorizer.transform(lyrics_test)


# In[ ]:


classifier = LogisticRegression()
classifier.fit(x_train, y_train)
acc = classifier.score(x_test, y_test)
print("Accuracy:", acc)

